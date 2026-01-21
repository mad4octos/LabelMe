from __future__ import annotations

from pathlib import Path
import enum
import functools
import html
import math
import os
import os.path as osp
import re
import types
import typing
import webbrowser

import imgviz
import natsort
import numpy as np
import osam
from loguru import logger
from numpy.typing import NDArray
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from labelme import __appname__
from labelme import __version__
from labelme._automation import bbox_from_text
from labelme._label_file import LabelFile
from labelme._label_file import LabelFileError
from labelme._label_file import ShapeDict
from labelme._label_file import convert_coco_detections_to_shapes
from labelme.config import get_config
from labelme.shape import Shape
from labelme.widgets import AiPromptWidget
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import StatusStats
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget
from labelme.widgets import download_ai_model
from labelme.widgets.guided_review_widget import GuidedReviewWidget
from labelme.widgets.guided_review_widget import MissedAnnotationDialog
from labelme.widgets.guided_review_widget import ReviewSummaryDialog
from labelme.guided_review_mode import GuidedReviewManager, AnnotationPair
from . import utils
from labelme.coco_dataset import LazyCOCODataset

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".

# handle high-dpi scaling issue
# https://leomoon.com/journal/python/high-dpi-scaling-in-pyqt5
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


LABEL_COLORMAP: NDArray[np.uint8] = imgviz.label_colormap()


class _ZoomMode(enum.Enum):
    FIT_WINDOW = enum.auto()
    FIT_WIDTH = enum.auto()
    MANUAL_ZOOM = enum.auto()


class MainWindow(QtWidgets.QMainWindow):
    filename: str | None
    _config: dict
    _is_changed: bool = False
    _copied_shapes: list[Shape]
    _zoom_mode: _ZoomMode
    _zoom_values: dict[str, tuple[_ZoomMode, int]]
    _brightness_contrast_values: dict[str, tuple[int | None, int | None]]
    _prev_opened_dir: str | None
    _other_data: dict | None

    # NB: this tells Mypy etc. that `actions` here
    #     is a different type cf. the parent class
    #     (where it is Callable[[QWidget], list[QAction]]).
    actions: types.SimpleNamespace  # type: ignore[assignment]

    def __init__(
        self,
        config: dict | None = None,
        filename: str | None = None,
        output: str | None = None,
        output_file: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        if output is not None:
            logger.warning("argument output is deprecated, use output_file instead")
            if output_file is None:
                output_file = output
        del output

        # see labelme/config/default_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super().__init__()
        self.setWindowTitle(__appname__)

        self._copied_shapes = []

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self._prev_opened_dir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self._load_flags(flags={k: False for k in config["flags"]})
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.setDirty)

        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        self.labelList.itemDoubleClicked.connect(self._edit_label)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Polygon Labels"), self)
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr("Select label to start annotating for it. Press 'Esc' to deselect.")
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                self.uniqLabelList.add_label_item(
                    label=label, color=self._get_rgb_by_label(label=label)
                )
        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))
        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(self.fileSelectionChanged)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
            bbox_padding=self._config["canvas"]["bbox_padding"],
        )
        self.canvas.zoomRequest.connect(self._zoom_requested)
        self.canvas.mouseMoved.connect(self._update_status_stats)
        self.canvas.statusUpdated.connect(lambda text: self.status_left.setText(text))

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)

        # Guided review mode
        self._review_manager = GuidedReviewManager(self)
        self._review_widget = GuidedReviewWidget(self)
        self.review_dock = QtWidgets.QDockWidget(self.tr("Guided Review"), self)
        self.review_dock.setObjectName("GuidedReviewDock")
        self.review_dock.setWidget(self._review_widget)
        self.review_dock.setVisible(False)

        # Connect review manager signals
        self._review_manager.reviewModeChanged.connect(self._on_review_mode_changed)
        self._review_manager.currentPairChanged.connect(self._on_current_pair_changed)
        self._review_manager.progressUpdated.connect(
            self._review_widget.update_progress
        )
        self._review_manager.frameReviewCompleted.connect(
            self._on_frame_review_completed
        )

        # Connect review widget signals
        self._review_widget.confirmClicked.connect(self._review_confirm)
        self._review_widget.editClicked.connect(self._review_edit)
        self._review_widget.deleteClicked.connect(self._review_delete)
        self._review_widget.exitReviewClicked.connect(self._exit_review_mode)
        self._review_widget.viewSummaryClicked.connect(self._show_review_summary)
        self._review_widget.resetFrameClicked.connect(self._reset_frame_review)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.review_dock)

        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit"),
            self.close,
            shortcuts["quit"],
            icon=None,
            tip=self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open\n"),
            self._open_file_with_dialog,
            shortcuts["open"],
            icon="folder-open.svg",
            tip=self.tr("Open image or label file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self._open_dir_with_dialog,
            shortcuts["open_dir"],
            icon="folder-open.svg",
            tip=self.tr("Open Dir"),
        )
        openNextImg = action(
            self.tr("&Next Image"),
            self._open_next_image,
            shortcuts["open_next"],
            icon="arrow-fat-right.svg",
            tip=self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image"),
            self._open_prev_image,
            shortcuts["open_prev"],
            icon="arrow-fat-left.svg",
            tip=self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openNextEntity = action(
            self.tr("&Next Entity"),
            self._select_next_entity,
            shortcuts["open_next_entity"],
            icon="arrow-fat-right.svg",
            tip=self.tr("Select next entity"),
            enabled=False,
        )
        openPrevEntity = action(
            self.tr("&Prev Entity"),
            self._select_prev_entity,
            shortcuts["open_prev_entity"],
            icon="arrow-fat-left.svg",
            tip=self.tr("Select prev entity"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n"),
            self.saveFile,
            shortcuts["save"],
            icon="floppy-disk.svg",
            tip=self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            icon="floppy-disk.svg",
            tip=self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            icon="file-x.svg",
            tip=self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="folders.svg",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text=self.tr("Save With Image Data"),
            slot=self.enableSaveImageWithData,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        exportCOCO = action(
            text=self.tr("Export COCO Annotations"),
            slot=self.exportCOCOAnnotations,
            shortcut=shortcuts.get("export_coco"),
            icon="floppy-disk.svg",
            tip=self.tr("Export annotations to COCO JSON file"),
            enabled=False,
        )

        close = action(
            self.tr("&Close"),
            self.closeFile,
            shortcuts["close"],
            icon="x-circle.svg",
            tip=self.tr("Close current file"),
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep previous annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(
            self.tr("Create Polygons"),
            lambda: self._switch_canvas_mode(edit=False, createMode="polygon"),
            shortcuts["create_polygon"],
            "polygon.svg",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        createRectangleMode = action(
            self.tr("Create Rectangle"),
            lambda: self._switch_canvas_mode(edit=False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "rectangle.svg",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self._switch_canvas_mode(edit=False, createMode="circle"),
            shortcuts["create_circle"],
            "circle.svg",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line"),
            lambda: self._switch_canvas_mode(edit=False, createMode="line"),
            shortcuts["create_line"],
            "line-segment.svg",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point"),
            lambda: self._switch_canvas_mode(edit=False, createMode="point"),
            shortcuts["create_point"],
            icon="circles-four.svg",
            tip=self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self._switch_canvas_mode(edit=False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "line-segments.svg",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self._switch_canvas_mode(edit=False, createMode="ai_polygon"),
            None,
            "ai-polygon.svg",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiMaskMode = action(
            self.tr("Create AI-Mask"),
            lambda: self._switch_canvas_mode(edit=False, createMode="ai_mask"),
            None,
            "ai-mask.svg",
            self.tr("Start drawing ai_mask. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        editMode = action(
            self.tr("Edit Polygons"),
            lambda: self._switch_canvas_mode(edit=True),
            shortcuts["edit_polygon"],
            icon="note-pencil.svg",
            tip=self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            icon="trash.svg",
            tip=self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            icon="copy.svg",
            tip=self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons"),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons"),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            icon="arrow-u-up-left.svg",
            tip=self.tr("Undo last drawn point"),
            enabled=False,
        )
        removePoint = action(
            text=self.tr("Remove Selected Point"),
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="trash.svg",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n"),
            self.undoShapeEdit,
            shortcuts["undo"],
            icon="arrow-u-up-left.svg",
            tip=self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            shortcuts["hide_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            shortcuts["show_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )
        toggleAll = action(
            self.tr("&Toggle\nPolygons"),
            functools.partial(self.togglePolygons, None),
            shortcuts["toggle_all_polygons"],
            icon="eye.svg",
            tip=self.tr("Toggle all polygons"),
            enabled=False,
        )

        help = action(
            self.tr("&Tutorial"),
            self.tutorial,
            icon="question.svg",
            tip=self.tr("Show tutorial page"),
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomLabel = QtWidgets.QLabel(self.tr("Zoom"))
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomBoxLayout.addWidget(zoomLabel)
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(f"{shortcuts['zoom_in']},{shortcuts['zoom_out']}"),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            lambda _: self._add_zoom(increment=1.1),
            shortcuts["zoom_in"],
            icon="magnifying-glass-minus.svg",
            tip=self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            lambda _: self._add_zoom(increment=0.9),
            shortcuts["zoom_out"],
            icon="magnifying-glass-plus.svg",
            tip=self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            self._set_zoom_to_original,
            shortcuts["zoom_to_original"],
            icon="image-square.svg",
            tip=self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            icon="frame-corners.svg",
            tip=self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            icon="frame-arrows-horizontal.svg",
            tip=self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            self.tr("&Brightness Contrast"),
            self.brightnessContrast,
            None,
            "brightness-contrast.svg",
            self.tr("Adjust brightness and contrast"),
            enabled=False,
        )
        self._zoom_mode = _ZoomMode.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            _ZoomMode.FIT_WINDOW: self.scaleFitWindow,
            _ZoomMode.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            _ZoomMode.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self._edit_label,
            shortcuts["edit_label"],
            icon="note-pencil.svg",
            tip=self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            icon="paint-bucket.svg",
            tip=self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        toggleGuidedReview = action(
            self.tr("Guided Review"),
            self._toggle_guided_review_mode,
            shortcuts.get("guided_review", "Ctrl+G"),
            icon="magnifying-glass.svg",
            tip=self.tr("Toggle guided annotation review mode (Ctrl-G)"),
            checkable=True,
            enabled=False,
        )

        # Guided review mode actions (only active during review)
        reviewConfirm = action(
            self.tr("Confirm Annotation"),
            self._review_confirm,
            shortcuts.get("review_confirm", "C"),
            tip=self.tr("Confirm current annotation (C)"),
            enabled=False,
        )
        reviewEdit = action(
            self.tr("Edit Annotation"),
            self._review_edit,
            shortcuts.get("review_edit", "E"),
            tip=self.tr("Edit current annotation (E)"),
            enabled=False,
        )
        reviewDelete = action(
            self.tr("Delete Annotation"),
            self._review_delete,
            shortcuts.get("review_delete", "Delete"),
            tip=self.tr("Delete current annotation (Del)"),
            enabled=False,
        )
        reviewExit = action(
            self.tr("Exit Review"),
            self._exit_review_mode,
            shortcuts.get("review_exit", "Escape"),
            tip=self.tr("Exit guided review mode (Esc)"),
            enabled=False,
        )
        reviewResetFrame = action(
            self.tr("Reset Frame Review"),
            self._reset_frame_review,
            shortcuts.get("review_reset_frame", "R"),
            tip=self.tr("Reset review progress for current frame (R)"),
            enabled=False,
        )

        # Label list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = types.SimpleNamespace(
            about=action(
                text=f"&About {__appname__}",
                slot=functools.partial(
                    QMessageBox.about,
                    self,
                    f"About {__appname__}",
                    f"""
<h3>{__appname__}</h3>
<p>Image Polygonal Annotation with Python</p>
<p>Version: {__version__}</p>
<p>Author: Kentaro Wada</p>
<p>
    <a href="https://labelme.io">Homepage</a> |
    <a href="https://labelme.io/docs">Documentation</a> |
    <a href="https://labelme.io/docs/troubleshoot">Troubleshooting</a>
</p>
<p>
    <a href="https://github.com/wkentaro/labelme">GitHub</a> |
    <a href="https://x.com/labelmeai">Twitter/X</a>
</p>
""",
                ),
            ),
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            exportCOCO=exportCOCO,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            toggle_keep_prev_brightness_contrast=action(
                text=self.tr("Keep Previous Brightness/Contrast"),
                slot=lambda: self._config.__setitem__(
                    "keep_prev_brightness_contrast",
                    not self._config["keep_prev_brightness_contrast"],
                ),
                checkable=True,
                checked=self._config["keep_prev_brightness_contrast"],
            ),
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            createAiMaskMode=createAiMaskMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,
            openNextEntity=openNextEntity,
            openPrevEntity=openPrevEntity,
            toggleGuidedReview=toggleGuidedReview,
            reviewConfirm=reviewConfirm,
            reviewEdit=reviewEdit,
            reviewDelete=reviewDelete,
            reviewExit=reviewExit,
            reviewResetFrame=reviewResetFrame,
        )

        # Add review actions to main window and review widget so shortcuts work
        # when either window is focused
        for review_action in (
            reviewConfirm,
            reviewEdit,
            reviewDelete,
            reviewExit,
            reviewResetFrame,
        ):
            self.addAction(review_action)
            self._review_widget.addAction(review_action)

        self.on_shapes_present_actions = (saveAs, hideAll, showAll, toggleAll)

        self.draw_actions: list[tuple[str, QtWidgets.QAction]] = [
            ("polygon", createMode),
            ("rectangle", createRectangleMode),
            ("circle", createCircleMode),
            ("point", createPointMode),
            ("line", createLineMode),
            ("linestrip", createLineStripMode),
            ("ai_polygon", createAiPolygonMode),
            ("ai_mask", createAiMaskMode),
        ]

        # Group zoom controls into a list for easier toggling.
        self.zoom_actions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.on_load_active_actions = (
            close,
            createMode,
            createRectangleMode,
            createCircleMode,
            createLineMode,
            createPointMode,
            createLineStripMode,
            createAiPolygonMode,
            createAiMaskMode,
            brightnessContrast,
            toggleGuidedReview,
        )
        # menu shown at right click
        self.context_menu_actions = (
            *[draw_action for _, draw_action in self.draw_actions],
            editMode,
            edit,
            duplicate,
            copy,
            paste,
            delete,
            undo,
            undoLastPoint,
            removePoint,
        )
        # XXX: need to add some actions here to activate the shortcut
        self.edit_menu_actions = (
            edit,
            duplicate,
            copy,
            paste,
            delete,
            None,
            undo,
            undoLastPoint,
            None,
            removePoint,
            None,
            toggle_keep_prev_mode,
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = types.SimpleNamespace(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions(
            self.menus.file,
            (
                open_,
                openNextImg,
                openPrevImg,
                openNextEntity,
                openPrevEntity,
                opendir,
                self.menus.recentFiles,
                save,
                saveAs,
                saveAuto,
                changeOutputDir,
                saveWithImageData,
                exportCOCO,
                close,
                deleteFile,
                None,
                quit,
            ),
        )
        utils.addActions(self.menus.help, (help, self.actions.about))
        utils.addActions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                fill_drawing,
                None,
                hideAll,
                showAll,
                toggleAll,
                None,
                zoomIn,
                zoomOut,
                zoomOrg,
                keepPrevScale,
                None,
                fitWindow,
                fitWidth,
                None,
                brightnessContrast,
                self.actions.toggle_keep_prev_brightness_contrast,
            ),
        )

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        #
        selectAiModelLabel = QtWidgets.QLabel(self.tr("AI Mask Model"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)
        #
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        selectAiModel.defaultWidget().layout().addWidget(self._selectAiModelComboBox)
        MODEL_NAMES: list[tuple[str, str]] = [
            ("efficientsam:10m", "EfficientSam (speed)"),
            ("efficientsam:latest", "EfficientSam (accuracy)"),
            ("sam:100m", "Sam (speed)"),
            ("sam:300m", "Sam (balanced)"),
            ("sam:latest", "Sam (accuracy)"),
            ("sam2:small", "Sam2 (speed)"),
            ("sam2:latest", "Sam2 (balanced)"),
            ("sam2:large", "Sam2 (accuracy)"),
        ]
        for model_name, model_ui_name in MODEL_NAMES:
            self._selectAiModelComboBox.addItem(model_ui_name, userData=model_name)
        model_ui_names: list[str] = [model_ui_name for _, model_ui_name in MODEL_NAMES]
        if self._config["ai"]["default"] in model_ui_names:
            model_index = model_ui_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda index: self.canvas.set_ai_model_name(
                model_name=self._selectAiModelComboBox.itemData(index)
            )
        )
        self._selectAiModelComboBox.setCurrentIndex(model_index)

        self._ai_prompt_widget: AiPromptWidget = AiPromptWidget(
            on_submit=self._submit_ai_prompt, parent=self
        )
        ai_prompt_action = QtWidgets.QWidgetAction(self)
        ai_prompt_action.setDefaultWidget(self._ai_prompt_widget)

        self.addToolBar(
            Qt.TopToolBarArea,
            ToolBar(
                title="Tools",
                actions=[
                    open_,
                    opendir,
                    openPrevImg,
                    openNextImg,
                    save,
                    deleteFile,
                    None,
                    editMode,
                    duplicate,
                    delete,
                    undo,
                    brightnessContrast,
                    None,
                    fitWindow,
                    zoom,
                    None,
                    toggleGuidedReview,
                    None,
                    selectAiModel,
                    None,
                    ai_prompt_action,
                ],
                font_base=self.font(),
            ),
        )
        self.addToolBar(
            Qt.LeftToolBarArea,
            ToolBar(
                title="CreateShapeTools",
                actions=[a for _, a in self.draw_actions],
                orientation=Qt.Vertical,
                button_style=Qt.ToolButtonTextUnderIcon,
                font_base=self.font(),
            ),
        )

        self.status_left = QtWidgets.QLabel(self.tr("%s started.") % __appname__)
        self.status_right = StatusStats()
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addWidget(self.status_right, 0)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.labelFile: LabelFile | None = None
        self.imagePath: str | None = None
        self.recentFiles: list[str] = []
        self.maxRecent = 7
        self._other_data = None
        self.zoom_level = 100
        self.fit_window = False
        self._zoom_values = {}
        self._brightness_contrast_values = {}
        self.scroll_values = {  # type: ignore[var-annotated]
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(900, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry'])
        self.restoreState(state)

        # Ensure review dock is hidden after state restoration
        # (restoreState may have set it visible from a previous session)
        self.review_dock.setVisible(False)

        if filename:
            if osp.isdir(filename):
                self._import_images_from_dir(root_dir=filename)
                self._open_next_image()
            else:
                self._load_file(filename=filename)
        else:
            self.filename = None

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self._paint_canvas)

        self.populateModeActions()

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], self.context_menu_actions)
        self.menus.edit.clear()
        actions = (
            *[draw_action for _, draw_action in self.draw_actions],
            self.actions.editMode,
            *self.edit_menu_actions,
        )
        utils.addActions(self.menus.edit, actions)

    def _get_window_title(self, dirty: bool) -> str:
        window_title: str = __appname__
        if self.imagePath:
            window_title = f"{window_title} - {self.imagePath}"
            if self.fileListWidget.count() and self.fileListWidget.currentItem():
                window_title = (
                    f"{window_title} "
                    f"[{self.fileListWidget.currentRow() + 1}"
                    f"/{self.fileListWidget.count()}]"
                )
        if dirty:
            window_title = f"{window_title}*"
        return window_title

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            assert self.imagePath
            label_file = f"{osp.splitext(self.imagePath)[0]}.json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self._is_changed = True
        self.actions.save.setEnabled(True)
        self.setWindowTitle(self._get_window_title(dirty=True))

    def setClean(self):
        self._is_changed = False
        self.actions.save.setEnabled(False)
        for _, action in self.draw_actions:
            action.setEnabled(True)
        self.setWindowTitle(self._get_window_title(dirty=False))

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.zoom_actions:
            z.setEnabled(value)
        for action in self.on_load_active_actions:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def show_status_message(self, message, delay=500):
        self.statusBar().showMessage(message, delay)

    def _submit_ai_prompt(self, _) -> None:
        texts = self._ai_prompt_widget.get_text_prompt().split(",")

        model_name: str = "yoloworld"
        model_type = osam.apis.get_model_type_by_name(model_name)
        model_type = typing.cast(type[osam.types.Model], model_type)
        if not (_is_already_downloaded := model_type.get_size() is not None):
            if not download_ai_model(model_name=model_name, parent=self):
                return

        boxes, scores, labels = bbox_from_text.get_bboxes_from_texts(
            model=model_name,
            image=utils.img_qt_to_arr(self.image)[:, :, :3],
            texts=texts,
        )

        for shape in self.canvas.shapes:
            if shape.shape_type != "rectangle" or shape.label not in texts:
                continue
            box = np.array(
                [
                    shape.points[0].x(),
                    shape.points[0].y(),
                    shape.points[1].x(),
                    shape.points[1].y(),
                ],
                dtype=np.float32,
            )
            boxes = np.r_[boxes, [box]]
            scores = np.r_[scores, [1.01]]
            labels = np.r_[labels, [texts.index(shape.label)]]

        boxes, scores, labels = bbox_from_text.nms_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=self._ai_prompt_widget.get_iou_threshold(),
            score_threshold=self._ai_prompt_widget.get_score_threshold(),
            max_num_detections=100,
        )

        keep = scores != 1.01
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        shape_dicts: list[dict] = bbox_from_text.get_shapes_from_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            texts=texts,
        )

        shapes: list[Shape] = []
        for shape_dict in shape_dicts:
            shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                description=shape_dict["description"],
            )
            for point in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(*point))
            shapes.append(shape)

        self.canvas.storeShapes()
        self._load_shapes(shapes, replace=False)
        self.setDirty()

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self._other_data = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self._load_shapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def tutorial(self):
        url = "https://github.com/labelmeai/labelme/tree/main/examples/tutorial"  # NOQA
        webbrowser.open(url)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def _switch_canvas_mode(
        self, edit: bool = True, createMode: str | None = None
    ) -> None:
        self.canvas.setEditing(edit)
        if createMode is not None:
            self.canvas.createMode = createMode
        if edit:
            for _, draw_action in self.draw_actions:
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in self.draw_actions:
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, f"&{i + 1} {QtCore.QFileInfo(f).fileName()}", self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)  # type: ignore[attr-defined,union-attr]
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def validateGroupId(
        self, group_id: int | None, shape_type: str, exclude_shape=None
    ) -> tuple[bool, str]:
        """
        Validate that a group_id can be assigned to a shape of the given type.

        Rules:
        - Only one rectangle and one polygon can share the same group_id.
        - Returns (is_valid, error_message).

        Args:
            group_id: The group_id to validate (None always passes).
            shape_type: The shape type being assigned this group_id.
            exclude_shape: Shape to exclude from validation (for editing).
        """
        if group_id is None:
            return True, ""

        # Count shapes with the same group_id by type
        rectangle_count = 0
        polygon_count = 0

        for shape in self.canvas.shapes:
            if shape.group_id == group_id:
                # Skip the shape being edited
                if exclude_shape is not None and shape is exclude_shape:
                    continue
                if shape.shape_type == "rectangle":
                    rectangle_count += 1
                elif shape.shape_type == "polygon":
                    polygon_count += 1

        # Check if adding this shape would violate the constraint
        if shape_type == "rectangle" and rectangle_count >= 1:
            return False, self.tr(
                "ObjID {} already has a rectangle. "
                "Only one rectangle and one polygon can share the same ObjID."
            ).format(group_id)
        elif shape_type == "polygon" and polygon_count >= 1:
            return False, self.tr(
                "ObjID {} already has a polygon. "
                "Only one rectangle and one polygon can share the same ObjID."
            ).format(group_id)

        return True, ""

    def _edit_label(self, value=None):
        if not self.canvas.editing():
            return

        items = self.labelList.selectedItems()
        if not items:
            logger.warning("No label is selected, so cannot edit label.")
            return

        shape = items[0].shape()

        if len(items) == 1:
            edit_text = True
            edit_flags = True
            edit_group_id = True
            edit_description = True
        else:
            edit_text = all(item.shape().label == shape.label for item in items[1:])
            edit_flags = all(item.shape().flags == shape.flags for item in items[1:])
            edit_group_id = all(
                item.shape().group_id == shape.group_id for item in items[1:]
            )
            edit_description = all(
                item.shape().description == shape.description for item in items[1:]
            )

        if not edit_text:
            self.labelDialog.edit.setDisabled(True)
            self.labelDialog.labelList.setDisabled(True)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(True)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(True)

        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label if edit_text else "",
            flags=shape.flags if edit_flags else None,
            group_id=shape.group_id if edit_group_id else None,
            description=shape.description if edit_description else None,
            flags_disabled=not edit_flags,
        )

        if not edit_text:
            self.labelDialog.edit.setDisabled(False)
            self.labelDialog.labelList.setDisabled(False)
        if not edit_group_id:
            self.labelDialog.edit_group_id.setDisabled(False)
        if not edit_description:
            self.labelDialog.editDescription.setDisabled(False)

        if text is None:
            assert flags is None
            assert group_id is None
            assert description is None
            return

        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        # Validate group_id for each shape before applying changes
        if edit_group_id:
            for item in items:
                shape: Shape = item.shape()
                is_valid, error_msg = self.validateGroupId(
                    group_id, shape.shape_type, exclude_shape=shape
                )
                if not is_valid:
                    self.errorMessage(self.tr("Invalid ObjID"), error_msg)
                    return

        self.canvas.storeShapes()
        for item in items:
            shape: Shape = item.shape()  # type: ignore[no-redef]
            original_group_id = shape.group_id

            if edit_text:
                shape.label = text
            if edit_flags:
                shape.flags = flags
            if edit_group_id:
                shape.group_id = group_id
            if edit_description:
                shape.description = description

            self._update_shape_color(shape)
            self._update_label_list_item_text(item, shape)

            # Update paired shape (polygon/rectangle with same group_id)
            if (
                shape.group_id is not None
                and group_id is not None
                and original_group_id is not None
            ):
                paired_type = (
                    "rectangle" if shape.shape_type == "polygon" else "polygon"
                )
                for other_item in self.labelList:
                    other_shape: Shape = other_item.shape()
                    if (
                        other_shape.group_id == original_group_id
                        and other_shape.shape_type == paired_type
                    ):
                        if edit_group_id:
                            other_shape.group_id = group_id
                        if edit_text:
                            other_shape.label = text
                            self._update_shape_color(other_shape)
                        self._update_label_list_item_text(other_item, other_shape)
                        break

            self.setDirty()
            if self.uniqLabelList.find_label_item(shape.label) is None:
                self.uniqLabelList.add_label_item(
                    label=shape.label, color=self._get_rgb_by_label(label=shape.label)
                )

    def fileSearchChanged(self):
        self._import_images_from_dir(
            root_dir=self._prev_opened_dir, pattern=self.fileSearch.text()
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self._can_continue():
            return

        filepath = item.data(Qt.UserRole)
        if filepath:
            self._load_file(filepath)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self.labelList.itemSelectionChanged.disconnect(self._label_selection_changed)
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected)

    def addLabel(self, shape: Shape):
        if shape.group_id is None:
            text = f"{shape.label} [{shape.shape_type}]"
        else:
            text = f"{shape.label} ({shape.group_id}) [{shape.shape_type}]"
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.find_label_item(shape.label) is None:
            self.uniqLabelList.add_label_item(
                label=shape.label, color=self._get_rgb_by_label(label=shape.label)
            )
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.on_shapes_present_actions:
            action.setEnabled(True)

        self._update_shape_color(shape)
        r, g, b = shape.fill_color.getRgb()[:3]
        label_list_item.setText(
            f'{html.escape(text)}<font color="#{r:02x}{g:02x}{b:02x}">●</font>'
        )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _update_label_list_item_text(self, item, shape: Shape):
        if shape.group_id is None:
            text = f"{shape.label} [{shape.shape_type}]"
        else:
            text = f"{shape.label} ({shape.group_id}) [{shape.shape_type}]"
        r, g, b = shape.fill_color.getRgb()[:3]
        item.setText(
            f'{html.escape(text)}<font color="#{r:02x}{g:02x}{b:02x}">●</font>'
        )

    def _get_rgb_by_label(self, label: str) -> tuple[int, int, int]:
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.find_label_item(label)
            item_index: int = (
                self.uniqLabelList.indexFromItem(item).row()
                if item
                else self.uniqLabelList.count()
            )
            label_id: int = (
                1  # skip black color by default
                + item_index
                + self._config["shift_auto_shape_color"]
            )
            rgb: tuple[int, int, int] = tuple(
                LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)].tolist()
            )
            return rgb
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            if not (
                len(self._config["label_colors"][label]) == 3
                and all(0 <= c <= 255 for c in self._config["label_colors"][label])
            ):
                raise ValueError(
                    "Color for label must be 0-255 RGB tuple, but got: "
                    f"{self._config['label_colors'][label]}"
                )
            return tuple(self._config["label_colors"][label])
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def _load_shapes(self, shapes: list[Shape], replace: bool = True) -> None:
        self.labelList.itemSelectionChanged.disconnect(self._label_selection_changed)
        shape: Shape
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.labelList.itemSelectionChanged.connect(self._label_selection_changed)
        self.canvas.loadShapes(shapes=shapes, replace=replace)

    def _load_shape_dicts(self, shape_dicts: list[ShapeDict]) -> None:
        shapes: list[Shape] = []
        shape_dict: ShapeDict
        for shape_dict in shape_dicts:
            shape: Shape = Shape(
                label=shape_dict["label"],
                shape_type=shape_dict["shape_type"],
                group_id=shape_dict["group_id"],
                description=shape_dict["description"],
                mask=shape_dict["mask"]
            )
            for x, y in shape_dict["points"]:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if not isinstance(shape.label, str):
                        logger.warning("shape.label is not str: {}", shape.label)
                        continue
                    if re.match(pattern, shape.label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(shape_dict["flags"])
            shape.other_data = shape_dict["other_data"]

            shapes.append(shape)
        self._load_shapes(shapes=shapes)

    def _load_flags(self, flags: dict[str, bool]) -> None:
        self.flag_widget.clear()  # type: ignore[union-attr]
        key: str
        flag: bool
        for key, flag in flags.items():
            item: QtWidgets.QListWidgetItem = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)  # type: ignore[union-attr]

    def saveLabels(self, filename, coco=False):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        for i in range(self.flag_widget.count()):  # type: ignore[union-attr]
            item = self.flag_widget.item(i)  # type: ignore[union-attr]
            assert item
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            imageData = self.imageData if self._config["store_data"] else None
            image_height = self.image.height()
            image_width = self.image.width()

            if coco:
                image_filename = Path(self.imagePath).name
                image_id = self.dataset.image_id_by_filename[image_filename]
                lf._sync_labelme_shapes_to_coco_dataset(
                    self.dataset,
                    image_id,
                    shapes=shapes,
                    im_height=image_height,
                    im_width=image_width,
                    imageData=imageData,
                )
            else:
                assert self.imagePath
                imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
                if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                    os.makedirs(osp.dirname(filename))
                lf.save(
                    filename=filename,
                    shapes=shapes,
                    imagePath=imagePath,
                    imageData=imageData,
                    imageHeight=image_height,
                    imageWidth=image_width,
                    otherData=self._other_data,
                    flags=flags,
                )

            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicateSelectedShape(self):
        self.copySelectedShape()
        self.pasteSelectedShape()

    def pasteSelectedShape(self):
        self._load_shapes(shapes=self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def _label_selection_changed(self) -> None:
        if not self.canvas.editing():
            logger.warning("canvas is not editing mode, cannot change label selection")
            return

        selected_shapes: list[Shape] = []
        for item in self.labelList.selectedItems():
            selected_shapes.append(item.shape())
        if selected_shapes:
            self.canvas.selectShapes(selected_shapes)
        else:
            self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        if self._config["display_label_popup"] or not text:
            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)

            # Validate group_id before assignment
            is_valid, error_msg = self.validateGroupId(group_id, shape.shape_type)
            if not is_valid:
                self.errorMessage(self.tr("Invalid ObjID"), error_msg)
                self.canvas.undoLastLine()
                self.canvas.shapesBackups.pop()
                return

            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)

            # Automatically create bbox with padding around polygon
            if shape.shape_type == "polygon" and shape.points:
                # Calculate bounding box from polygon points
                xs = [p.x() for p in shape.points]
                ys = [p.y() for p in shape.points]
                padding = self._config["canvas"]["bbox_padding"]

                # Create rectangle shape with the same label and group_id
                bbox_shape = Shape(
                    label=shape.label,
                    shape_type="rectangle",
                    flags=flags,
                    group_id=shape.group_id,
                    description=shape.description
                )
                bbox_shape.addPoint(
                    QtCore.QPointF(min(xs) - padding, min(ys) - padding)
                )
                bbox_shape.addPoint(
                    QtCore.QPointF(max(xs) + padding, max(ys) + padding)
                )
                bbox_shape.close()

                # Add the bbox to canvas and label list
                self.canvas.shapes.append(bbox_shape)
                self.addLabel(bbox_shape)
                self.canvas.storeShapes()

            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def _set_zoom(self, value: int, pos: QtCore.QPointF | None = None) -> None:
        if self.filename is None:
            logger.warning("filename is None, cannot set zoom")
            return

        if pos is None:
            pos = QtCore.QPointF(self.canvas.visibleRegion().boundingRect().center())
        canvas_width_old: int = self.canvas.width()

        self.actions.fitWidth.setChecked(self._zoom_mode == _ZoomMode.FIT_WIDTH)
        self.actions.fitWindow.setChecked(self._zoom_mode == _ZoomMode.FIT_WINDOW)
        self.canvas.enableDragging(
            enabled=value > int(self.scalers[_ZoomMode.FIT_WINDOW]() * 100)
        )
        self.zoomWidget.setValue(value)  # triggers self._paint_canvas
        self._zoom_values[self.filename] = (self._zoom_mode, value)

        canvas_width_new: int = self.canvas.width()
        if canvas_width_old == canvas_width_new:
            return
        canvas_scale_factor = canvas_width_new / canvas_width_old
        x_shift: float = pos.x() * canvas_scale_factor - pos.x()
        y_shift: float = pos.y() * canvas_scale_factor - pos.y()
        self.setScroll(
            Qt.Horizontal,
            self.scrollBars[Qt.Horizontal].value() + x_shift,
        )
        self.setScroll(
            Qt.Vertical,
            self.scrollBars[Qt.Vertical].value() + y_shift,
        )

    def _set_zoom_to_original(self):
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(value=100)

    def _add_zoom(self, increment: float, pos: QtCore.QPointF | None = None) -> None:
        zoom_value: int
        if increment > 1:
            zoom_value = math.ceil(self.zoomWidget.value() * increment)
        else:
            zoom_value = math.floor(self.zoomWidget.value() * increment)
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(value=zoom_value, pos=pos)

    def _zoom_requested(self, delta: int, pos: QtCore.QPointF) -> None:
        self._add_zoom(increment=1.1 if delta > 0 else 0.9, pos=pos)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self._zoom_mode = _ZoomMode.FIT_WINDOW if value else _ZoomMode.MANUAL_ZOOM
        self._adjust_scale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self._zoom_mode = _ZoomMode.FIT_WIDTH if value else _ZoomMode.MANUAL_ZOOM
        self._adjust_scale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage), clear_shapes=False)

    def brightnessContrast(self, value: bool, is_initial_load: bool = False):
        if self.filename is None:
            logger.warning("filename is None, cannot set brightness/contrast")
            return

        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData).convert("RGB"),
            self.onNewBrightnessContrast,
            parent=self,
        )

        brightness: int | None
        contrast: int | None
        brightness, contrast = self._brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if is_initial_load:
            prev_filename: str = self.recentFiles[0] if self.recentFiles else ""
            if self._config["keep_prev_brightness_contrast"] and prev_filename:
                brightness, contrast = self._brightness_contrast_values.get(
                    prev_filename, (None, None)
                )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)

        if is_initial_load:
            dialog.onNewValue(None)
        else:
            dialog.exec_()
            brightness = dialog.slider_brightness.value()
            contrast = dialog.slider_contrast.value()

        self._brightness_contrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        flag = value
        for item in self.labelList:
            if value is None:
                flag = item.checkState() == Qt.Unchecked
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)

    def _load_file(self, filename=None):
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file
        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        prev_shapes: list[Shape] = (
            self.canvas.shapes
            if self._config["keep_prev"]
            or QtWidgets.QApplication.keyboardModifiers()
            == (Qt.ControlModifier | Qt.ShiftModifier)
            else []
        )
        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.show_status_message(self.tr("Loading %s...") % osp.basename(str(filename)))
        label_file = f"{osp.splitext(filename)[0]}.json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p><p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.show_status_message(self.tr("Error reading %s") % label_file)
                return False
            assert self.labelFile is not None
            self.imageData = self.labelFile.imageData
            assert self.labelFile.imagePath
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self._other_data = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        assert self.imageData is not None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                f"*.{fmt.data().decode()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.show_status_message(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self._load_shape_dicts(shape_dicts=self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        else:
            self.curr_frame, self.curr_frame_annotations = self.dataset[
                self.fileListWidget.currentRow()
            ]

            # Get the original COCO annotations for this image
            image_filename = Path(filename).name
            image_id = self.dataset.image_id_by_filename[image_filename]
            image_annotations = self.dataset.annotations_by_image_id.get(image_id, [])

            shapes: list[ShapeDict] = []

            # Create a shape for each bounding box
            shapes.extend(
                convert_coco_detections_to_shapes(
                    detections=self.curr_frame_annotations,
                    classes_names=self.dataset.classes,
                    image_annotations=image_annotations,
                )
            )

            # Create a shape for each mask
            shapes.extend(
                convert_coco_detections_to_shapes(
                    detections=self.curr_frame_annotations,
                    classes_names=self.dataset.classes,
                    image_annotations=image_annotations,
                    mask=True,
                )
            )

            self._load_shape_dicts(shape_dicts=shapes)

        self._load_flags(flags=flags)
        if prev_shapes and self.noShapes():
            self._load_shapes(shapes=prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self._zoom_values
        if self.filename in self._zoom_values:
            self._zoom_mode = self._zoom_values[self.filename][0]
            self._set_zoom(self._zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self._zoom_mode = _ZoomMode.FIT_WINDOW
            self._adjust_scale()
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        self.brightnessContrast(value=False, is_initial_load=True)
        self._paint_canvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.show_status_message(self.tr("Loaded %s") % osp.basename(filename))
        logger.debug("loaded file: {!r}", filename)

        # Update guided review for the new frame if review mode is active
        if self._review_manager.is_active:
            self._start_guided_review()

        return True

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if (
            self.canvas
            and not self.image.isNull()
            and self._zoom_mode != _ZoomMode.MANUAL_ZOOM
        ):
            self._adjust_scale()
        super().resizeEvent(a0)

    def _paint_canvas(self) -> None:
        if self.image.isNull():
            logger.warning("image is null, cannot paint canvas")
            return
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def _adjust_scale(self) -> None:
        self._set_zoom(value=int(self.scalers[self._zoom_mode]() * 100))

    def scaleFitWindow(self) -> float:
        EPSILON_TO_HIDE_SCROLLBAR: float = 2.0
        w1: float = self.centralWidget().width() - EPSILON_TO_HIDE_SCROLLBAR
        h1: float = self.centralWidget().height() - EPSILON_TO_HIDE_SCROLLBAR
        a1: float = w1 / h1

        w2: float = self.canvas.pixmap.width()
        h2: float = self.canvas.pixmap.height()
        a2: float = w2 / h2

        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        EPSILON_TO_HIDE_SCROLLBAR: float = 15.0
        w = self.centralWidget().width() - EPSILON_TO_HIDE_SCROLLBAR
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if not self._can_continue():
            a0.ignore()
        self.settings.setValue("filename", self.filename if self.filename else "")
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if a0.mimeData().hasUrls():
            items = [i.toLocalFile() for i in a0.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                a0.accept()
        else:
            a0.ignore()

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        if not self._can_continue():
            a0.ignore()
            return
        items = [i.toLocalFile() for i in a0.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self._can_continue():
            self._load_file(filename)

    def _open_adjacent_image(self, direction: int) -> None:
        """Open the previous or next image in the file list.

        Args:
            direction: -1 for previous image, +1 for next image.
        """
        # _can_continue() must be called BEFORE changing the row, because it may
        # trigger a save dialog.
        if not self._can_continue():
            return

        target_row: int = self.fileListWidget.currentRow() + direction
        if target_row < 0 or target_row >= self.fileListWidget.count():
            logger.debug("there is no {} image", "prev" if direction < 0 else "next")
            return

        # Block signals to prevent fileSelectionChanged() from being triggered.
        # That method also calls _can_continue(), which would show the "Save
        # annotations?" dialog a second time. Loading is handled directly below.
        self.fileListWidget.blockSignals(True)
        self.fileListWidget.setCurrentRow(target_row)
        self.fileListWidget.blockSignals(False)
        self.fileListWidget.repaint()
        item = self.fileListWidget.item(target_row)
        if item and (filepath := item.data(Qt.UserRole)):
            self._load_file(filepath)

    def _open_prev_image(self, _value=False) -> None:
        self._open_adjacent_image(-1)

    def _open_next_image(self, _value=False) -> None:
        self._open_adjacent_image(+1)

    def _select_entity_by_offset(self, row_offset: int):
        """
        Select an entity in the polygon label list by applying a row offset
        relative to the current selection.
        """

        selection_model = self.labelList.selectionModel()
        current_index = selection_model.currentIndex()
        row_count = self.labelList._model.rowCount()

        if row_count == 0:
            self.canvas.deSelectShape()
            return

        # Determine target row with bounds clamping
        if current_index.isValid():
            target_row = max(0, min(current_index.row() + row_offset, row_count - 1))
        else:
            target_row = 0

        # Select and scroll to target
        target_index = self.labelList._model.index(target_row, 0)
        selection_model.setCurrentIndex(
            target_index,
            QtCore.QItemSelectionModel.ClearAndSelect,
        )
        self.labelList.scrollTo(target_index)

        # Update canvas selection
        if target_index.isValid():
            item = self.labelList._model.itemFromIndex(target_index)
            shape: Shape = item.shape()
            self.canvas.selectShapes([shape])
            return shape
        else:
            self.canvas.deSelectShape()

    def _center_content_point(self, shape) -> None:
        """Center the viewport on the given shape."""
        rect = shape.boundingRect()
        center = rect.center()  # unscaled image coordinates

        # Get actual image dimensions from pixmap
        pixmap = self.canvas.pixmap
        if pixmap and not pixmap.isNull():
            image_width = pixmap.width()
            image_height = pixmap.height()
        else:
            # Fallback if no pixmap available
            image_width = self.canvas.width() / self.canvas.scale
            image_height = self.canvas.height() / self.canvas.scale

        # Calculate scaled image dimensions
        scaled_image_width = image_width * self.canvas.scale
        scaled_image_height = image_height * self.canvas.scale

        # Calculate the offset (padding) where the image starts on the canvas
        offset_x = (self.canvas.width() - scaled_image_width) / 2
        offset_y = (self.canvas.height() - scaled_image_height) / 2

        # Convert image coordinates to canvas coordinates
        canvas_x = offset_x + center.x() * self.canvas.scale
        canvas_y = offset_y + center.y() * self.canvas.scale

        # Get the visible viewport dimensions
        hbar = self.scrollBars[Qt.Horizontal]
        vbar = self.scrollBars[Qt.Vertical]
        viewport_width = hbar.pageStep()
        viewport_height = vbar.pageStep()

        # Calculate scroll position to center the point in viewport
        target_x = canvas_x - viewport_width / 2
        target_y = canvas_y - viewport_height / 2

        self.setScroll(Qt.Horizontal, target_x)
        self.setScroll(Qt.Vertical, target_y)

    def _zoom_to_shapes(self, shapes: list, padding_factor: float = 0.2) -> None:
        """
        Zoom and center the viewport to fit the given shapes.

        Args:
            shapes: List of Shape objects to fit in view
            padding_factor: Extra padding around shapes (0.2 = 20% padding)
        """
        if not shapes:
            return

        # Compute combined bounding box of all shapes
        combined_rect = shapes[0].boundingRect()
        for shape in shapes[1:]:
            combined_rect = combined_rect.united(shape.boundingRect())

        # Get viewport dimensions
        viewport_width = self.centralWidget().width()
        viewport_height = self.centralWidget().height()

        # Get shape dimensions in image coordinates
        shape_width = combined_rect.width()
        shape_height = combined_rect.height()

        if shape_width <= 0 or shape_height <= 0:
            return

        # Calculate zoom to fit shape with padding
        padding = 1.0 + padding_factor
        scale_x = viewport_width / (shape_width * padding)
        scale_y = viewport_height / (shape_height * padding)
        target_scale = min(scale_x, scale_y)

        # Convert scale to zoom percentage
        target_zoom = int(target_scale * 100)

        # Clamp zoom to reasonable bounds
        min_zoom = 10
        max_zoom = 500
        target_zoom = max(min_zoom, min(max_zoom, target_zoom))

        # Set the zoom level
        self._zoom_mode = _ZoomMode.MANUAL_ZOOM
        self._set_zoom(target_zoom)

        # Center on the combined bounding box
        self._center_content_point(shapes[0])

    def _select_next_entity(self, _value=False):
        """ """
        if self._review_manager.is_active:
            return  # Block entity navigation in review mode
        shape = self._select_entity_by_offset(1)
        if shape is not None:
            self._center_content_point(shape)

    def _select_prev_entity(self, _value=False):
        """ """
        if self._review_manager.is_active:
            return  # Block entity navigation in review mode
        shape = self._select_entity_by_offset(-1)
        if shape is not None:
            self._center_content_point(shape)

    # ========== Guided Review Mode Methods ==========

    def _toggle_guided_review_mode(self, checked: bool) -> None:
        """Toggle guided review mode on/off."""
        if checked:
            self._start_guided_review()
        else:
            self._exit_review_mode()

    def _start_guided_review(self) -> None:
        """Initialize and start guided review for current frame."""
        logger.debug("Starting guided review...")

        if not self.canvas.shapes:
            QMessageBox.information(
                self,
                self.tr("No Annotations"),
                self.tr("No annotations found on this frame to review."),
            )
            self.actions.toggleGuidedReview.setChecked(False)
            return

        success = self._review_manager.start_review(
            shapes=self.canvas.shapes,
            filename=self.filename,
        )

        if not success:
            QMessageBox.information(
                self,
                self.tr("No Annotation Pairs"),
                self.tr(
                    "No bbox/polygon pairs with group IDs found.\n"
                    "Shapes must have a group ID (ObjID) to be reviewed."
                ),
            )
            self.actions.toggleGuidedReview.setChecked(False)
        else:
            # Update overall progress when review starts
            self._update_overall_progress()

    def _exit_review_mode(self) -> None:
        """Exit guided review mode."""
        self._review_manager._clear_state()
        self.actions.toggleGuidedReview.setChecked(False)

    def _on_review_mode_changed(self, active: bool) -> None:
        """Handle review mode activation/deactivation."""
        # Enable/disable review shortcut actions
        self.actions.reviewConfirm.setEnabled(active)
        self.actions.reviewEdit.setEnabled(active)
        self.actions.reviewDelete.setEnabled(active)
        self.actions.reviewExit.setEnabled(active)
        self.actions.reviewResetFrame.setEnabled(active)

        if active:
            # Show the review dock and block shape editing
            self.review_dock.setVisible(True)
            self.canvas.set_review_editing_blocked(True)
        else:
            # Restore normal canvas state
            self.canvas.set_review_highlight(False)
            self.canvas.set_review_editing_blocked(False)
            self.review_dock.setVisible(False)

    def _on_current_pair_changed(self, pair: AnnotationPair | None) -> None:
        """Handle change to current review pair."""
        self._review_widget.update_current_pair(pair)

        if pair:
            # Update canvas highlighting
            self.canvas.set_review_highlight(True, pair.group_id)

            # Select shapes in the pair
            self.canvas.selectShapes(pair.shapes)

            # Zoom and center view on the pair
            if pair.shapes:
                self._zoom_to_shapes(pair.shapes)
        else:
            # Clear visual highlighting but keep editing blocked.
            # Editing is unblocked when exiting review mode via _on_review_mode_changed
            self.canvas.set_review_highlight(False)
            self.canvas.deSelectShape()

    def _on_frame_review_completed(self) -> None:
        """Handle completion of all pairs on current frame."""
        # Reset zoom to fit window so user can see the whole picture
        self.setFitWindow(True)

        # Show missed annotation dialog
        dialog = MissedAnnotationDialog(self)
        summary = self._review_manager.get_review_summary()
        dialog.set_summary(summary)

        # Position dialog in top-right corner, out of the way
        dialog.adjustSize()
        main_geo = self.geometry()
        dialog.move(
            main_geo.x() + main_geo.width() - dialog.width() - 20,
            main_geo.y() + 50,
        )

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # User confirmed no missed annotations - mark frame as completed
            self._review_manager.complete_frame_review()
            self._update_overall_progress()
            self._proceed_to_next_frame_after_review()
        else:
            # User wants to add more annotations - exit review mode
            self._exit_review_mode()

    def _proceed_to_next_frame_after_review(self) -> None:
        """Save current frame and advance to next."""
        # Save the current frame
        if self._is_changed:
            self.saveFile()

        # Exit current review
        # TODO: maybe I should refactor this pattern of entering/exiting the review mode between frames
        self._exit_review_mode()

        # Move to next frame
        row_next = self.fileListWidget.currentRow() + 1
        if row_next >= self.fileListWidget.count():
            QMessageBox.information(
                self,
                self.tr("Review Complete"),
                self.tr("You have reached the last frame."),
            )
            return

        self.fileListWidget.setCurrentRow(row_next)
        self.fileListWidget.repaint()

        # Auto-start review on next frame if it has annotations with group_ids
        QtCore.QTimer.singleShot(100, self._auto_start_review_if_applicable)

    def _auto_start_review_if_applicable(self) -> None:
        """Auto-start guided review if current frame has reviewable annotations."""
        # Check if there are shapes with group_ids
        has_grouped_shapes = any(
            shape.group_id is not None for shape in self.canvas.shapes
        )
        if has_grouped_shapes:
            self.actions.toggleGuidedReview.setChecked(True)
            self._start_guided_review()

    def _review_confirm(self) -> None:
        """Handle confirm action in review mode."""
        logger.debug("Confirming review")
        self._review_manager.confirm_current()

    def _review_edit(self) -> None:
        """Handle edit action - exits review mode for manual editing."""
        # Mark as TO_EDIT so we return to this annotation after editing
        self._review_manager.mark_for_edit()

        # Exit review mode so user can edit freely
        self._exit_review_mode()

        QMessageBox.information(
            self,
            self.tr("Edit Mode"),
            self.tr(
                "Review mode paused for editing.\n"
                "Make your changes, then press Ctrl+G to restart review."
            ),
        )

    def _review_delete(self) -> None:
        """Handle delete action in review mode."""
        pair = self._review_manager.current_pair
        if pair:
            # Delete all shapes in the pair
            for shape in pair.shapes:
                if shape in self.canvas.shapes:
                    self.canvas.shapes.remove(shape)
                    item = self.labelList.findItemByShape(shape)
                    if item:
                        self.labelList.removeItem(item)

            self.canvas.storeShapes()
            self.setDirty()

        self._review_manager.mark_deleted()

    def _reset_frame_review(self) -> None:
        """Reset review progress for the current frame."""
        self._review_manager.reset_frame_review()

    def _get_all_frame_names(self) -> list[str]:
        """Get list of all frame filenames in the dataset."""
        return [osp.basename(f) for f in self.imageList]

    def _update_overall_progress(self) -> None:
        """Update the overall progress display in the review widget."""
        if not self._review_manager._persistence:
            return

        all_frames = self._get_all_frame_names()
        summary = self._review_manager._persistence.get_summary(all_frames)
        self._review_widget.update_overall_progress(
            completed=summary["completed"],
            total=summary["total"],
        )

    def _show_review_summary(self) -> None:
        """Show the review summary dialog."""
        if not self._review_manager._persistence:
            return

        all_frames = self._get_all_frame_names()
        summary = self._review_manager._persistence.get_summary(all_frames)

        # Get current frame summary if in review mode
        current_frame_summary = None
        if self._review_manager.is_active:
            current_frame_summary = self._review_manager.get_review_summary()

        # Build frames status list
        frames_status = []
        for frame_name in all_frames:
            frame_state = self._review_manager._persistence.get_frame_state(frame_name)
            frames_status.append(
                {
                    "name": frame_name,
                    "status": frame_state.status.value,
                }
            )

        dialog = ReviewSummaryDialog(
            summary, current_frame_summary, frames_status, self
        )
        dialog.exec_()

    # ========== End Guided Review Mode Methods ==========

    def _open_file_with_dialog(self, _value: bool = False) -> None:
        if not self._can_continue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            f"*.{fmt.data().decode()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + [f"*{LabelFile.suffix}"]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self._load_file(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self._import_images_from_dir(root_dir=self._prev_opened_dir)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(self.imageList.index(current_filename))
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if isinstance(self.dataset, LazyCOCODataset):
            self._saveFile(None, coco=True)
        else:
            if self.labelFile:
                # DL20180323 - overwrite when in directory
                self._saveFile(self.labelFile.filename)
            elif self.output_file:
                self._saveFile(self.output_file)
                self.close()
            else:
                self._saveFile(self.saveFileDialog())


    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def exportCOCOAnnotations(self):
        """Export all annotations to COCO JSON file."""

        if not hasattr(self, "dataset") or not isinstance(
            self.dataset, LazyCOCODataset
        ):
            QMessageBox.warning(
                self,
                "Export COCO Annotations",
                "No COCO dataset loaded. This feature only works when a COCO "
                "annotations file is loaded.",
                QMessageBox.Ok,
            )
            return

        # Ask user for output path
        caption = self.tr("Export COCO Annotations")
        filters = self.tr("JSON files (*.json)")
        default_path = str(
            self.dataset.annotations_file_path.parent
            / f"{self.dataset.annotations_file_path.stem}_export.json"
        )

        filename = QtWidgets.QFileDialog.getSaveFileName(
            self, caption, default_path, filters
        )

        if isinstance(filename, tuple):
            filename = filename[0]

        if not filename:
            return

        try:
            self.dataset.export_annotations(Path(filename))
            QMessageBox.information(
                self,
                "Export Successful",
                f"COCO annotations exported to:\n{filename}",
                QMessageBox.Ok,
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export COCO annotations:\n{str(e)}",
                QMessageBox.Ok,
            )

    def saveFileDialog(self):
        assert self.filename is not None
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(self, caption, self.output_dir, filters)
        else:
            dlg = QtWidgets.QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            return filename[0]
        return filename

    def _saveFile(self, filename, coco=False):
        if filename is None and coco:
            self.saveLabels(filename, coco)
            self.setClean()
        else:
            if filename and self.saveLabels(filename):
                self.addRecentFile(filename)
                self.setClean()

    def closeFile(self, _value=False):
        if not self._can_continue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.fileListWidget.setFocus()
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        assert self.filename is not None
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = f"{osp.splitext(self.filename)[0]}.json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info(f"Label file is removed: {label_file}")

            item = self.fileListWidget.currentItem()
            if item:
                item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def _can_continue(self) -> bool:
        if not self._is_changed:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(self.filename)
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, f"<p><b>{title}</b></p>{message}"
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if self.canvas.hShape and not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.on_shapes_present_actions:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):
        if self._review_manager.is_active:
            return  # Use review widget's delete action instead
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = self.tr(
            "You are about to permanently delete {} polygons, proceed anyway?"
        ).format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(
            self, self.tr("Attention"), msg, yes | no, yes
        ):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.on_shapes_present_actions:
                    action.setEnabled(False)

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def _open_dir_with_dialog(self, _value: bool = False) -> None:
        if not self._can_continue():
            return

        defaultOpenDirPath: str
        if self._prev_opened_dir and osp.exists(self._prev_opened_dir):
            defaultOpenDirPath = self._prev_opened_dir
        else:
            defaultOpenDirPath = osp.dirname(self.filename) if self.filename else "."

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )

        images_dir = Path(targetDirPath, "images", "train")
        annotations_file = Path(targetDirPath, "annotations", "instances_train.json")

        if not images_dir.exists():
            print(f"Couldn't find expected route: '{images_dir}'")
        if not annotations_file.exists():
            print(f"Couldn't find expected annotations file: '{annotations_file}'")
        
        if images_dir.exists() and annotations_file.exists():
            self.dataset = LazyCOCODataset(images_dir, annotations_file)
            self.actions.exportCOCO.setEnabled(True)
            self._import_images_from_annotations_file()
        else:
            self._import_images_from_dir(root_dir=targetDirPath)
            self._open_next_image()

    @property
    def imageList(self) -> list[str]:
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            assert item
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(tuple(extensions)):
                continue
            label_file = f"{osp.splitext(file)[0]}.json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)

        self._open_next_image()

    def _import_images_from_annotations_file(self):
        """ """
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)
        self.actions.openNextEntity.setEnabled(True)
        self.actions.openPrevEntity.setEnabled(True)

        if not self._can_continue():
            return

        self._review_manager.set_dataset_dir(self.dataset.annotations_file_path.parent)

        # Populate the sidebar File List
        for im_path in self.dataset.image_filepaths:
            item = QtWidgets.QListWidgetItem(str(im_path.name))
            item.setData(Qt.UserRole, str(im_path))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if im_path.exists():
                item.setCheckState(Qt.Checked)
            self.fileListWidget.addItem(item)

    def _import_images_from_dir(
        self, root_dir: str | None, pattern: str | None = None
    ) -> None:
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self._can_continue() or not root_dir:
            return

        self._prev_opened_dir = root_dir
        self.filename = None
        self.fileListWidget.clear()

        filenames = _scan_image_files(root_dir=root_dir)
        if pattern:
            try:
                filenames = [f for f in filenames if re.search(pattern, f)]
            except re.error:
                pass
        for filename in filenames:
            label_file = f"{osp.splitext(filename)[0]}.json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(label_file):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

    def _update_status_stats(self, mouse_pos: QtCore.QPointF) -> None:
        stats: list[str] = []
        stats.append(f"mode={self.canvas.mode.name}")
        stats.append(f"x={mouse_pos.x():6.1f}, y={mouse_pos.y():6.1f}")
        self.status_right.setText(" | ".join(stats))


def _scan_image_files(root_dir: str) -> list[str]:
    extensions: list[str] = [
        f".{fmt.data().decode().lower()}"
        for fmt in QtGui.QImageReader.supportedImageFormats()
    ]

    images: list[str] = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                relativePath = os.path.normpath(osp.join(root, file))
                images.append(relativePath)

    logger.debug("found {:d} images in {!r}", len(images), root_dir)
    return natsort.os_sorted(images)
