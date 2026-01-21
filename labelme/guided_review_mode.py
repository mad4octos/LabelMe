"""Guided Review Mode - State management for annotation review workflow."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from loguru import logger
from PyQt5 import QtCore

from labelme.review_persistence import ReviewPersistence
from labelme.review_persistence import ReviewStatus
from labelme.shape import Shape


@dataclass
class AnnotationPair:
    """Represents a bbox/polygon pair with same group_id."""

    group_id: int
    shapes: list[Shape] = field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING


class GuidedReviewManager(QtCore.QObject):
    """
    Manage the guided review mode state and progression.

    Emit signals for UI updates and mode transitions.
    Each review session is scoped to a single frame.
    """

    reviewModeChanged = QtCore.pyqtSignal(bool)  # active/inactive
    currentPairChanged = QtCore.pyqtSignal(object)  # AnnotationPair or None
    progressUpdated = QtCore.pyqtSignal(int, int)  # current_idx (1-based), total
    frameReviewCompleted = QtCore.pyqtSignal()
    editConfirmed = QtCore.pyqtSignal(int)  # group_id - emitted when TO_EDIT -> EDITED

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active: bool = False
        self._annotation_pairs: list[AnnotationPair] = []
        self._current_index: int = 0
        self._frame_filename: str | None = None
        self._persistence: ReviewPersistence | None = None

    def set_dataset_dir(self, dataset_dir: str | Path | None) -> None:
        """Set dataset directory and initialize persistence."""
        if dataset_dir is None:
            self._persistence = None
            return
        self._persistence = ReviewPersistence(dataset_dir)
        self._persistence.load()

    @property
    def is_active(self) -> bool:
        """Check if review mode is currently active."""
        return self._active

    @property
    def current_pair(self) -> AnnotationPair | None:
        """Get the current annotation pair being reviewed."""
        if not self._active or not self._annotation_pairs:
            return None
        if 0 <= self._current_index < len(self._annotation_pairs):
            return self._annotation_pairs[self._current_index]
        return None

    @property
    def total_pairs(self) -> int:
        """Get total number of annotation pairs."""
        return len(self._annotation_pairs)

    def start_review(self, shapes: list[Shape], filename: str) -> bool:
        """
        Initialize review mode for current frame.

        Groups shapes by group_id and starts iteration.
        Restores previous review state if available.
        Returns False if no annotation pairs found.
        """
        self._frame_filename = filename
        self._annotation_pairs = self._group_shapes_by_id(shapes)

        if not self._annotation_pairs:
            return False

        # Restore previous review state from persistence
        if self._persistence:
            frame_name = Path(filename).name
            frame_state = self._persistence.get_frame_state(frame_name)

            # Restore annotation statuses
            for pair in self._annotation_pairs:
                if (gid_str := str(pair.group_id)) in frame_state.annotations:
                    pair.status = frame_state.annotations[gid_str].status

        # Find first pending annotation (skip already reviewed)
        self._current_index = self._find_first_pending_index()

        self._active = True
        self.reviewModeChanged.emit(True)
        self._emit_current_state()
        return True

    def _find_first_pending_index(self) -> int:
        """Find index of first annotation needing review (PENDING or TO_EDIT)."""
        for i, pair in enumerate(self._annotation_pairs):
            if pair.status in (ReviewStatus.PENDING, ReviewStatus.TO_EDIT):
                return i
        # All reviewed - position at end to trigger completion
        return len(self._annotation_pairs)

    def _clear_state(self) -> None:
        """Clear internal state and emit deactivation signals."""
        self._active = False
        self._annotation_pairs = []
        self._current_index = 0
        self._frame_filename = None
        self.reviewModeChanged.emit(False)
        self.currentPairChanged.emit(None)

    def _group_shapes_by_id(self, shapes: list[Shape]) -> list[AnnotationPair]:
        """Group shapes by group_id into annotation pairs."""
        groups: dict[int, list[Shape]] = {}
        for shape in shapes:
            if shape.group_id is not None:
                if shape.group_id not in groups:
                    groups[shape.group_id] = []
                groups[shape.group_id].append(shape)

        pairs = []
        for gid in sorted(groups.keys()):
            pairs.append(
                AnnotationPair(
                    group_id=gid, shapes=groups[gid], status=ReviewStatus.PENDING
                )
            )
        return pairs

    def confirm_current(self) -> None:
        """Mark current pair as confirmed and advance."""
        logger.debug(f"@confirm_current | self.current_pair: {self.current_pair} ")
        if self.current_pair:
            logger.debug("Good")
            # If user was editing (TO_EDIT), mark as EDITED; otherwise CONFIRMED
            if self.current_pair.status == ReviewStatus.TO_EDIT:
                self.current_pair.status = ReviewStatus.EDITED
                # Emit signal so incorrect predictions can be finalized
                self.editConfirmed.emit(self.current_pair.group_id)
            else:
                self.current_pair.status = ReviewStatus.CONFIRMED
            self._persist_current_status()
            self._advance()
        else:
            logger.debug("Bad")

    def mark_for_edit(self) -> None:
        """Mark current pair as needing edit (user will edit manually)."""
        if self.current_pair:
            self.current_pair.status = ReviewStatus.TO_EDIT
            self._persist_current_status()

    def mark_deleted(self) -> None:
        """Mark current pair as deleted and advance."""
        if self.current_pair:
            self.current_pair.status = ReviewStatus.DELETED
            self._persist_current_status()
            self._advance()

    def _persist_current_status(self) -> None:
        """Save current annotation status to disk."""
        logger.debug(
            f"@_persist_current_status: "
            f"self._persistence: {self._persistence}"
            f"self._frame_filename: {self._frame_filename}"
            f"self.current_pair: {self.current_pair}"
        )
        if self._persistence and self._frame_filename and self.current_pair:
            frame_name = Path(self._frame_filename).name
            self._persistence.set_annotation_status(
                frame_name=frame_name,
                group_id=self.current_pair.group_id,
                status=self.current_pair.status,
            )

    def _advance(self) -> None:
        """Move to next pair or trigger frame completion."""
        self._current_index += 1
        if self._current_index >= len(self._annotation_pairs):
            self._emit_current_state()
            self.frameReviewCompleted.emit()
        else:
            self._emit_current_state()

    def _emit_current_state(self) -> None:
        """Emit signals for current state."""
        self.currentPairChanged.emit(self.current_pair)
        self.progressUpdated.emit(self._current_index, len(self._annotation_pairs))

    def get_review_summary(self) -> dict[str, int]:
        """Get summary of review statuses."""
        summary: dict[str, int] = {status.name: 0 for status in ReviewStatus}
        for pair in self._annotation_pairs:
            summary[pair.status.name] += 1
        return summary

    def complete_frame_review(self) -> None:
        """Mark current frame as fully reviewed."""
        if self._persistence and self._frame_filename:
            frame_name = Path(self._frame_filename).name
            self._persistence.mark_frame_completed(frame_name)

    def reset_frame_review(self) -> None:
        """Reset review progress for current frame, starting from first annotation."""
        if not self._active:
            return

        # Reset all annotation statuses to PENDING
        for pair in self._annotation_pairs:
            pair.status = ReviewStatus.PENDING

        # Clear persisted state for this frame
        if self._persistence and self._frame_filename:
            frame_name = Path(self._frame_filename).name
            self._persistence.reset_frame(frame_name)

        # Reset to first annotation
        self._current_index = 0
        self._emit_current_state()
