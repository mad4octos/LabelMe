"""Guided Review Widget - UI components for annotation review workflow."""

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor

from labelme.guided_review_mode import AnnotationPair
from labelme.review_persistence import ReviewStatus


class ReviewSummaryDialog(QtWidgets.QDialog):
    """Dialog showing detailed review progress summary."""

    def __init__(
        self,
        summary: dict,
        current_frame_summary: dict | None = None,
        frames_status: list[dict] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Review Progress Summary"))
        self.setModal(True)
        self._setup_ui(summary, current_frame_summary, frames_status)

    def _setup_ui(
        self,
        summary: dict,
        current_frame_summary: dict | None,
        frames_status: list[dict] | None,
    ):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Title
        title = QtWidgets.QLabel(self.tr("Review Progress"))
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        # Separator
        line1 = QtWidgets.QFrame()
        line1.setFrameShape(QtWidgets.QFrame.HLine)
        line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line1)

        # Current Frame section (if available)
        if current_frame_summary:
            self._add_current_frame_section(layout, current_frame_summary)

            # Separator
            line_frame = QtWidgets.QFrame()
            line_frame.setFrameShape(QtWidgets.QFrame.HLine)
            line_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout.addWidget(line_frame)

        # Frame progress
        total = summary.get("total", 0)
        completed = summary.get("completed", 0)
        in_progress = summary.get("in_progress", 0)
        pending = summary.get("pending", 0)

        frames_title = QtWidgets.QLabel(self.tr("All Frames"))
        frames_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(frames_title)

        percent = (completed / total * 100) if total > 0 else 0
        frame_text = (
            f" - {self.tr('Total')}: {total}\n"
            f" - {self.tr('Completed')}: {completed} ({percent:.1f}%)\n"
            f" - {self.tr('In Progress')}: {in_progress}\n"
            f" - {self.tr('Pending')}: {pending}"
        )
        frame_label = QtWidgets.QLabel(frame_text)
        frame_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(frame_label)

        # Frames list (scrollable table)
        if frames_status:
            self._add_frames_list_section(layout, frames_status)

        # Separator
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line2)

        # Annotation counts (overall)
        confirmed = summary.get("confirmed", 0)
        edited = summary.get("edited", 0)
        deleted = summary.get("deleted", 0)

        ann_title = QtWidgets.QLabel(self.tr("All Annotations"))
        ann_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(ann_title)

        ann_text = (
            f" - {self.tr('Confirmed')}: {confirmed}\n"
            f" - {self.tr('Edited')}: {edited}\n"
            f" - {self.tr('Deleted')}: {deleted}"
        )
        ann_label = QtWidgets.QLabel(ann_text)
        ann_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(ann_label)

        # Close button
        layout.addStretch()
        close_btn = QtWidgets.QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setMinimumWidth(400)

    def _add_frames_list_section(
        self, layout: QtWidgets.QVBoxLayout, frames_status: list[dict]
    ):
        """Add scrollable list of all frames with their review status."""
        # Create table widget
        table = QtWidgets.QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels([self.tr("Image"), self.tr("Status")])
        table.setRowCount(len(frames_status))
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)

        # Status display mapping
        status_display = {
            "pending": (self.tr("Pending"), "#999999"),
            "in_progress": (self.tr("In Progress"), "#FF9800"),
            "completed": (self.tr("Completed"), "#4CAF50"),
        }

        for row, frame_info in enumerate(frames_status):
            # Image name
            name_item = QtWidgets.QTableWidgetItem(frame_info["name"])
            table.setItem(row, 0, name_item)

            # Status with color
            status = frame_info["status"]
            display_text, color = status_display.get(
                status, (status, "#999999")
            )
            status_item = QtWidgets.QTableWidgetItem(display_text)
            status_item.setForeground(QColor(color))
            table.setItem(row, 1, status_item)

        # Adjust column widths
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents
        )

        # Set fixed height for scrollable area
        table.setMaximumHeight(200)

        layout.addWidget(table)

    def _add_current_frame_section(
        self, layout: QtWidgets.QVBoxLayout, frame_summary: dict
    ):
        """Add current frame summary section."""
        current_title = QtWidgets.QLabel(self.tr("Current Frame"))
        current_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(current_title)

        # Count entities by status
        pending = frame_summary.get(ReviewStatus.PENDING.name, 0)
        to_edit = frame_summary.get(ReviewStatus.TO_EDIT.name, 0)
        confirmed = frame_summary.get(ReviewStatus.CONFIRMED.name, 0)
        edited = frame_summary.get(ReviewStatus.EDITED.name, 0)
        deleted = frame_summary.get(ReviewStatus.DELETED.name, 0)

        total_entities = pending + to_edit + confirmed + edited + deleted

        current_text = (
            f" - {self.tr('Total entities')}: {total_entities}\n"
            f" - {self.tr('Pending')}: {pending + to_edit}\n"
            f" - {self.tr('Confirmed')}: {confirmed}\n"
            f" - {self.tr('Edited')}: {edited}\n"
            f" - {self.tr('Deleted')}: {deleted}"
        )
        current_label = QtWidgets.QLabel(current_text)
        current_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(current_label)


class GuidedReviewWidget(QtWidgets.QWidget):
    """
    Floating widget showing review controls and progress.

    Designed to be non-modal and non-blocking.
    """

    # Signals for actions
    confirmClicked = QtCore.pyqtSignal()
    editClicked = QtCore.pyqtSignal()
    deleteClicked = QtCore.pyqtSignal()
    exitReviewClicked = QtCore.pyqtSignal()
    viewSummaryClicked = QtCore.pyqtSignal()
    resetFrameClicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setWindowTitle(self.tr("Guided Review"))
        # self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header with title and exit button
        header = QtWidgets.QHBoxLayout()
        self._title_label = QtWidgets.QLabel(self.tr("Guided Review Mode"))
        self._title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(self._title_label)
        header.addStretch()

        self._exit_btn = QtWidgets.QPushButton(self.tr("Exit"))
        self._exit_btn.setToolTip(self.tr("Exit review mode (Esc)"))
        self._exit_btn.clicked.connect(self.exitReviewClicked.emit)
        header.addWidget(self._exit_btn)
        layout.addLayout(header)

        # OVERALL PROGRESS section
        overall_progress_label = QtWidgets.QLabel(self.tr("Overall Progress"))
        overall_progress_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(overall_progress_label)

        overall_progress_layout = QtWidgets.QHBoxLayout()
        self._overall_progress_label = QtWidgets.QLabel("0 / 0 frames")
        self._overall_progress_label.setMinimumWidth(80)
        overall_progress_layout.addWidget(self._overall_progress_label)
        self._overall_progress_bar = QtWidgets.QProgressBar()
        self._overall_progress_bar.setMinimum(0)
        self._overall_progress_bar.setMaximum(100)
        self._overall_progress_bar.setTextVisible(False)
        overall_progress_layout.addWidget(self._overall_progress_bar)
        layout.addLayout(overall_progress_layout)

        # View Summary button
        self._view_summary_btn = QtWidgets.QPushButton(self.tr("View Summary"))
        self._view_summary_btn.setToolTip(self.tr("View detailed review progress"))
        self._view_summary_btn.clicked.connect(self.viewSummaryClicked.emit)
        layout.addWidget(self._view_summary_btn)

        # Separator line
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.HLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep1)

        # FRAME PROGRESS section
        frame_progress_label = QtWidgets.QLabel(self.tr("Frame Progress"))
        frame_progress_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(frame_progress_label)

        frame_progress_layout = QtWidgets.QHBoxLayout()
        self._progress_label = QtWidgets.QLabel("0 / 0")
        self._progress_label.setMinimumWidth(50)
        frame_progress_layout.addWidget(self._progress_label)
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setTextVisible(False)
        frame_progress_layout.addWidget(self._progress_bar)
        layout.addLayout(frame_progress_layout)

        # Separator line
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep2)

        # Current annotation info
        info_frame = QtWidgets.QFrame()
        info_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Raised)
        info_layout = QtWidgets.QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(4)

        self._group_id_label = QtWidgets.QLabel(self.tr("ObjID: --"))
        self._label_info = QtWidgets.QLabel(self.tr("Label: --"))

        info_layout.addWidget(self._group_id_label)
        info_layout.addWidget(self._label_info)
        layout.addWidget(info_frame)

        # Action buttons in a grid
        btn_layout = QtWidgets.QGridLayout()
        btn_layout.setSpacing(6)

        self._confirm_btn = QtWidgets.QPushButton(self.tr("Confirm"))
        self._confirm_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self._confirm_btn.setToolTip(self.tr("Confirm this annotation (C)"))
        self._confirm_btn.clicked.connect(self.confirmClicked.emit)
        btn_layout.addWidget(self._confirm_btn, 0, 0)

        self._edit_btn = QtWidgets.QPushButton(self.tr("Edit"))
        self._edit_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self._edit_btn.setToolTip(self.tr("Edit this annotation (E)"))
        self._edit_btn.clicked.connect(self.editClicked.emit)
        btn_layout.addWidget(self._edit_btn, 0, 1)

        self._delete_btn = QtWidgets.QPushButton(self.tr("Delete"))
        self._delete_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self._delete_btn.setToolTip(self.tr("Delete this annotation (Del)"))
        self._delete_btn.clicked.connect(self.deleteClicked.emit)
        btn_layout.addWidget(self._delete_btn, 1, 0)

        self._reset_frame_btn = QtWidgets.QPushButton(self.tr("Reset Frame"))
        self._reset_frame_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #F57C00; }"
        )
        self._reset_frame_btn.setToolTip(
            self.tr("Reset review progress for current frame (R)")
        )
        self._reset_frame_btn.clicked.connect(self.resetFrameClicked.emit)
        btn_layout.addWidget(self._reset_frame_btn, 1, 1)

        layout.addLayout(btn_layout)

        # Keyboard hints
        hints = QtWidgets.QLabel(
            self.tr("C=Confirm  E=Edit  Del=Delete  R=Reset  Esc=Exit")
        )
        hints.setStyleSheet("color: gray; font-size: 10px;")
        hints.setWordWrap(True)
        hints.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(hints)

        # Set fixed width
        self.setFixedWidth(280)

    def update_progress(self, current: int, total: int):
        """Update progress display."""
        self._progress_label.setText(f"{current} / {total}")
        if total > 0:
            self._progress_bar.setValue(int((current / total) * 100))
        else:
            self._progress_bar.setValue(0)

    def update_current_pair(self, pair: AnnotationPair | None):
        """Update display for current annotation pair."""
        if pair is None:
            self._group_id_label.setText(self.tr("ObjID: --"))
            self._label_info.setText(self.tr("Label: --"))
            self.set_buttons_enabled(False)
            return

        self._group_id_label.setText(f"ObjID: {pair.group_id}")

        # Get label from first shape
        if pair.shapes:
            label = pair.shapes[0].label or "--"
            self._label_info.setText(f"{self.tr('Label')}: {label}")

        self.set_buttons_enabled(True)

    def set_buttons_enabled(self, enabled: bool):
        """Enable/disable action buttons."""
        self._confirm_btn.setEnabled(enabled)
        self._edit_btn.setEnabled(enabled)
        self._delete_btn.setEnabled(enabled)

    def update_overall_progress(self, completed: int, total: int):
        """Update overall dataset progress display."""
        self._overall_progress_label.setText(f"{completed} / {total} frames")
        if total > 0:
            self._overall_progress_bar.setValue(int((completed / total) * 100))
        else:
            self._overall_progress_bar.setValue(0)

    def closeEvent(self, event):
        """Handle window close (X button) by exiting review mode."""
        self.exitReviewClicked.emit()
        event.accept()


class MissedAnnotationDialog(QtWidgets.QDialog):
    """
    Dialog shown at end of frame review to check for missed annotations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Frame Review Complete"))
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Summary
        summary_title = QtWidgets.QLabel(self.tr("Review Summary"))
        summary_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(summary_title)

        self._summary_label = QtWidgets.QLabel()
        self._summary_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self._summary_label)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)

        # Question
        question = QtWidgets.QLabel(
            self.tr(
                "Did you miss any annotations on this frame?\n"
                "(e.g., objects that should have been annotated but weren't)"
            )
        )
        question.setWordWrap(True)
        question.setStyleSheet("font-weight: bold;")
        layout.addWidget(question)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(8)

        self._add_annotation_btn = QtWidgets.QPushButton(
            self.tr("Yes, Add Annotations")
        )
        self._add_annotation_btn.setToolTip(
            self.tr("Exit review mode to add missed annotations")
        )
        self._add_annotation_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._add_annotation_btn)

        self._no_missed_btn = QtWidgets.QPushButton(
            self.tr("No, Continue to Next Frame")
        )
        self._no_missed_btn.setToolTip(self.tr("Save and proceed to the next frame"))
        self._no_missed_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 8px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self._no_missed_btn.clicked.connect(self.accept)
        self._no_missed_btn.setDefault(True)
        btn_layout.addWidget(self._no_missed_btn)

        layout.addLayout(btn_layout)

        # Keyboard hint
        hint = QtWidgets.QLabel(self.tr("Press C or Enter to continue"))
        hint.setStyleSheet("color: gray; font-size: 10px;")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(hint)

        self.setMinimumWidth(350)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts - C to continue (matches review_confirm)."""
        if event.key() == QtCore.Qt.Key_C:
            self.accept()
        else:
            super().keyPressEvent(event)

    def set_summary(self, summary: dict[str, int]):
        """Set the review summary to display."""
        confirmed = summary.get(ReviewStatus.CONFIRMED.name, 0)
        edited = summary.get(ReviewStatus.EDITED.name, 0)
        deleted = summary.get(ReviewStatus.DELETED.name, 0)
        text = (
            f" - {self.tr('Confirmed')}: {confirmed}\n"
            f" - {self.tr('Edited')}: {edited}\n"
            f" - {self.tr('Deleted')}: {deleted}"
        )
        self._summary_label.setText(text)
