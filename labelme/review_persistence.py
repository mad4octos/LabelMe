"""Review state persistence for Guided Review Mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path

from loguru import logger


class AnnotationReviewStatus(Enum):
    """Status of an annotation pair during review."""

    PENDING = "pending"
    TO_EDIT = "to_edit"  # Intermediate: user clicked Edit, needs to confirm after
    CONFIRMED = "confirmed"
    EDITED = "edited"  # Final: user edited and confirmed
    DELETED = "deleted"


class FrameStatus(Enum):
    """Status of a frame's review progress."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class AnnotationReviewState:
    """Review state for a single annotation (group_id)."""

    status: AnnotationReviewStatus = AnnotationReviewStatus.PENDING
    reviewed_at: str | None = None


@dataclass
class FrameReviewState:
    """Review state for a single frame."""

    status: FrameStatus = FrameStatus.PENDING
    annotations: dict[str, AnnotationReviewState] = field(default_factory=dict)


class ReviewPersistence:
    """
    Manage loading/saving review state to disk.

    Creates .labelme_review.json in the dataset directory.
    """

    FILENAME = ".labelme_review.json"
    VERSION = "1.0"

    def __init__(self, dataset_dir: str | Path) -> None:
        self._dataset_dir = Path(dataset_dir)
        self._filepath = self._dataset_dir / self.FILENAME
        self._frames: dict[str, FrameReviewState] = {}

    @property
    def filepath(self) -> Path:
        return self._filepath

    def load(self) -> bool:
        """Load review state from disk. Returns True if file existed."""
        if not self._filepath.exists():
            return False

        try:
            with open(self._filepath, encoding="utf-8") as f:
                data = json.load(f)

            for frame_name, frame_data in data.get("frames", {}).items():
                annotations = {}
                for gid, ann_data in frame_data.get("annotations", {}).items():
                    annotations[gid] = AnnotationReviewState(
                        status=AnnotationReviewStatus(ann_data.get("status", "pending")),
                        reviewed_at=ann_data.get("reviewed_at"),
                    )

                self._frames[frame_name] = FrameReviewState(
                    status=FrameStatus(frame_data.get("status", "pending")),
                    annotations=annotations,
                )

            return True

        except Exception as e:
            logger.error(f"Failed to load review state: {e}")
            self._frames = {}
            return False

    def save(self) -> bool:
        """Save current review state to disk."""
        data: dict = {
            "version": self.VERSION,
            "frames": {},
        }

        for frame_name, frame_state in self._frames.items():
            data["frames"][frame_name] = {
                "status": frame_state.status.value,
                "annotations": {
                    gid: {
                        "status": ann.status.value,
                        "reviewed_at": ann.reviewed_at,
                    }
                    for gid, ann in frame_state.annotations.items()
                },
            }

        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved review state to {self._filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save review state: {e}")
            return False

    def get_frame_state(self, frame_name: str) -> FrameReviewState:
        """Get review state for a frame, creating if not exists."""
        if frame_name not in self._frames:
            self._frames[frame_name] = FrameReviewState()
        return self._frames[frame_name]

    def set_annotation_status(
        self, frame_name: str, group_id: int, status: AnnotationReviewStatus
    ) -> None:
        """Update status for a specific annotation and auto-save."""
        frame = self.get_frame_state(frame_name)
        gid_str = str(group_id)

        frame.annotations[gid_str] = AnnotationReviewState(
            status=status,
            reviewed_at=datetime.utcnow().isoformat() + "Z",
        )

        self.save()

    def mark_frame_in_progress(self, frame_name: str) -> None:
        """Mark frame as review started."""
        logger.debug("Marking frame in progress")
        frame = self.get_frame_state(frame_name)
        frame.status = FrameStatus.IN_PROGRESS
        self.save()

    def mark_frame_completed(self, frame_name: str) -> None:
        """Mark frame as fully reviewed."""
        logger.debug("Marking frame completed")
        frame = self.get_frame_state(frame_name)
        frame.status = FrameStatus.COMPLETED
        self.save()

    def reset_frame(self, frame_name: str) -> None:
        """Reset frame review state, clearing all annotation statuses."""
        logger.debug(f"Resetting frame: {frame_name}")
        self._frames[frame_name] = FrameReviewState()
        self.save()

    def get_summary(self, all_frames: list[str]) -> dict:
        """
        Get overall review progress summary across all frames.

        Args:
            all_frames: List of all frame filenames in the dataset.

        Returns:
            Dictionary with keys:
                - total: Total number of frames
                - completed: Number of fully reviewed frames
                - in_progress: Number of frames currently being reviewed
                - pending: Number of frames not yet started
                - confirmed: Total annotations confirmed across all frames
                - edited: Total annotations marked as edited
                - deleted: Total annotations marked as deleted
        """
        summary = {
            "total": len(all_frames),
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "confirmed": 0,
            "edited": 0,
            "deleted": 0,
        }

        for frame_name in all_frames:
            if frame_name in self._frames:
                frame_state = self._frames[frame_name]

                # Count frame status
                if frame_state.status == FrameStatus.COMPLETED:
                    summary["completed"] += 1
                elif frame_state.status == FrameStatus.IN_PROGRESS:
                    summary["in_progress"] += 1
                else:
                    summary["pending"] += 1

                # Count annotation statuses
                for ann in frame_state.annotations.values():
                    if ann.status == AnnotationReviewStatus.CONFIRMED:
                        summary["confirmed"] += 1
                    elif ann.status == AnnotationReviewStatus.EDITED:
                        summary["edited"] += 1
                    elif ann.status == AnnotationReviewStatus.DELETED:
                        summary["deleted"] += 1
            else:
                summary["pending"] += 1

        return summary
