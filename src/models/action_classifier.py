"""
Cross-Species Action Classifier.

Unified action recognition for both humans and animals using normalized trajectories.
Supports rule-based classification and trainable models.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionMetrics:
    """Container for action classification results."""
    trajectory: np.ndarray  # (frames, 2) normalized trajectory
    velocity: np.ndarray    # (frames,) velocity magnitude
    action_labels: np.ndarray  # (frames,) action labels
    action_summary: Dict    # Summary statistics
    species: str           # 'mouse', 'human', etc.


# Unified action types (cross-species)
ACTION_TYPES = {
    "stationary": 0,  # resting / standing still
    "walking": 1,     # slow locomotion
    "running": 2,     # fast locomotion
    "other": 3,       # species-specific actions
}

# Species-specific velocity thresholds (normalized)
# Values are in body-lengths per second
VELOCITY_THRESHOLDS = {
    "mouse": {
        "stationary": 0.5,   # < 0.5 body-length/sec
        "walking": 3.0,      # 0.5 - 3.0 body-length/sec
        # > 3.0 = running
    },
    "human": {
        "stationary": 0.3,   # < 0.3 body-length/sec
        "walking": 1.5,      # 0.3 - 1.5 body-length/sec
        # > 1.5 = running
    },
    "quadruped": {
        "stationary": 0.5,
        "walking": 2.5,
    },
}


class UnifiedActionClassifier:
    """
    Cross-species action classifier using normalized trajectories.

    Key insight: By normalizing trajectories by body size, we can use
    similar velocity thresholds across species for basic actions.
    """

    def __init__(
        self,
        species: str = "mouse",
        fps: float = 30.0,
        smoothing_window: int = 5,
    ):
        """
        Initialize action classifier.

        Args:
            species: Species type ('mouse', 'human', 'quadruped')
            fps: Video frame rate
            smoothing_window: Window size for velocity smoothing
        """
        self.species = species
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.thresholds = VELOCITY_THRESHOLDS.get(species, VELOCITY_THRESHOLDS["mouse"])

    def extract_center_trajectory(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
    ) -> np.ndarray:
        """
        Extract center-of-mass trajectory from keypoints.

        Args:
            keypoints: (frames, num_kp, 3) array
            keypoint_names: List of keypoint names

        Returns:
            (frames, 2) center trajectory
        """
        # Find center keypoint based on species
        center_candidates = {
            "mouse": ["mouse_center", "mid_back", "neck"],
            "human": ["left_hip", "right_hip"],  # Will average
            "quadruped": ["spine_mid", "back", "neck"],
        }

        candidates = center_candidates.get(self.species, center_candidates["mouse"])

        # Try to find center keypoint
        for candidate in candidates:
            if candidate in keypoint_names:
                idx = keypoint_names.index(candidate)
                return keypoints[:, idx, :2]

        # Fallback: average all high-confidence keypoints
        logger.warning(f"No center keypoint found, using average")
        mask = keypoints[:, :, 2] > 0.3  # confidence > 0.3
        trajectory = np.zeros((len(keypoints), 2))

        for i in range(len(keypoints)):
            valid = mask[i]
            if valid.any():
                trajectory[i] = keypoints[i, valid, :2].mean(axis=0)

        return trajectory

    def estimate_body_size(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
    ) -> float:
        """
        Estimate body size for normalization.

        Args:
            keypoints: (frames, num_kp, 3) array
            keypoint_names: List of keypoint names

        Returns:
            Estimated body size in pixels
        """
        # Species-specific body length estimation
        length_pairs = {
            "mouse": [("nose", "tail_base"), ("nose", "mouse_center")],
            "human": [("nose", "left_ankle"), ("left_shoulder", "left_ankle")],
            "quadruped": [("nose", "tail_base"), ("neck", "tail_base")],
        }

        pairs = length_pairs.get(self.species, length_pairs["mouse"])

        for start, end in pairs:
            if start in keypoint_names and end in keypoint_names:
                start_idx = keypoint_names.index(start)
                end_idx = keypoint_names.index(end)

                # Calculate median distance across frames
                start_pts = keypoints[:, start_idx, :2]
                end_pts = keypoints[:, end_idx, :2]

                # Only use high-confidence frames
                conf_mask = (keypoints[:, start_idx, 2] > 0.3) & (keypoints[:, end_idx, 2] > 0.3)
                if conf_mask.any():
                    distances = np.linalg.norm(start_pts[conf_mask] - end_pts[conf_mask], axis=1)
                    return np.median(distances)

        # Fallback: use bounding box diagonal
        logger.warning("Using bounding box for body size estimation")
        valid = keypoints[:, :, 2] > 0.3
        sizes = []
        for i in range(len(keypoints)):
            if valid[i].any():
                pts = keypoints[i, valid[i], :2]
                bbox_size = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
                sizes.append(bbox_size)

        return np.median(sizes) if sizes else 100.0

    def compute_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute smoothed velocity from trajectory.

        Args:
            trajectory: (frames, 2) array

        Returns:
            (frames,) velocity magnitude
        """
        # Compute frame-to-frame displacement
        displacement = np.diff(trajectory, axis=0)
        velocity = np.linalg.norm(displacement, axis=1)

        # Pad to match original length
        velocity = np.concatenate([[0], velocity])

        # Smooth velocity
        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            velocity = np.convolve(velocity, kernel, mode='same')

        return velocity

    def classify_actions(
        self,
        velocity: np.ndarray,
        body_size: float,
    ) -> np.ndarray:
        """
        Classify actions based on normalized velocity.

        Args:
            velocity: (frames,) velocity in pixels/frame
            body_size: Body size in pixels

        Returns:
            (frames,) action labels
        """
        # Convert to body-lengths per second
        normalized_velocity = (velocity / body_size) * self.fps

        labels = np.full(len(velocity), ACTION_TYPES["other"], dtype=np.int32)

        stationary_thresh = self.thresholds["stationary"]
        walking_thresh = self.thresholds["walking"]

        labels[normalized_velocity < stationary_thresh] = ACTION_TYPES["stationary"]
        labels[(normalized_velocity >= stationary_thresh) &
               (normalized_velocity < walking_thresh)] = ACTION_TYPES["walking"]
        labels[normalized_velocity >= walking_thresh] = ACTION_TYPES["running"]

        return labels

    def analyze(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
    ) -> ActionMetrics:
        """
        Full action analysis pipeline.

        Args:
            keypoints: (frames, num_kp, 3) array
            keypoint_names: List of keypoint names

        Returns:
            ActionMetrics with all results
        """
        # Extract trajectory
        trajectory = self.extract_center_trajectory(keypoints, keypoint_names)

        # Estimate body size
        body_size = self.estimate_body_size(keypoints, keypoint_names)
        logger.info(f"Estimated body size: {body_size:.1f} pixels")

        # Compute velocity
        velocity = self.compute_velocity(trajectory)

        # Classify actions
        action_labels = self.classify_actions(velocity, body_size)

        # Compute summary
        action_names = {v: k for k, v in ACTION_TYPES.items()}
        action_counts = {name: 0 for name in ACTION_TYPES.keys()}

        for label in action_labels:
            name = action_names.get(label, "other")
            action_counts[name] += 1

        total = len(action_labels)
        summary = {
            "total_frames": total,
            "duration_sec": total / self.fps,
            "body_size_px": body_size,
            "actions": {
                name: {
                    "frames": count,
                    "percentage": 100 * count / total if total > 0 else 0,
                    "duration_sec": count / self.fps,
                }
                for name, count in action_counts.items()
            }
        }

        return ActionMetrics(
            trajectory=trajectory,
            velocity=velocity,
            action_labels=action_labels,
            action_summary=summary,
            species=self.species,
        )


class CrossSpeciesComparator:
    """Compare actions across different species."""

    def __init__(self, fps: float = 30.0):
        """Initialize comparator."""
        self.fps = fps
        self.results: Dict[str, ActionMetrics] = {}

    def add_result(self, name: str, metrics: ActionMetrics):
        """Add analysis result."""
        self.results[name] = metrics

    def compare_action_distributions(self) -> Dict:
        """Compare action distributions across all results."""
        comparison = {}

        for name, metrics in self.results.items():
            comparison[name] = {
                "species": metrics.species,
                "actions": metrics.action_summary["actions"],
                "duration_sec": metrics.action_summary["duration_sec"],
            }

        return comparison

    def save_comparison_csv(self, output_path: Union[str, Path]):
        """Save comparison results to CSV."""
        import pandas as pd

        rows = []
        for name, metrics in self.results.items():
            for action, stats in metrics.action_summary["actions"].items():
                rows.append({
                    "subject": name,
                    "species": metrics.species,
                    "action": action,
                    "frames": stats["frames"],
                    "percentage": stats["percentage"],
                    "duration_sec": stats["duration_sec"],
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved comparison to: {output_path}")
        return df
