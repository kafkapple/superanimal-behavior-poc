"""
Behavior analysis module for SuperAnimal keypoint predictions.
Analyzes motion patterns, classifies behaviors, and computes statistics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import ndimage
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class BehaviorMetrics:
    """Container for behavior analysis results."""
    velocity: np.ndarray  # (num_frames,) instantaneous velocity
    acceleration: np.ndarray  # (num_frames,) instantaneous acceleration
    distance_traveled: float  # total distance
    trajectory: np.ndarray  # (num_frames, 2) x, y positions
    behavior_labels: np.ndarray  # (num_frames,) behavior classification
    behavior_summary: Dict  # summary statistics


class BehaviorAnalyzer:
    """
    Analyze animal behavior from keypoint predictions.
    """

    BEHAVIOR_TYPES = {
        "resting": 0,
        "walking": 1,
        "running": 2,
        "rearing": 3,
        "grooming": 4,
        "unknown": -1,
    }

    def __init__(
        self,
        model_type: str = "topviewmouse",
        fps: float = 30.0,
        smoothing_window: int = 5,
        velocity_thresholds: Optional[Dict] = None,
    ):
        """
        Initialize behavior analyzer.

        Args:
            model_type: 'topviewmouse' or 'quadruped'
            fps: Video frames per second
            smoothing_window: Window size for smoothing
            velocity_thresholds: Dict of velocity thresholds for classification
        """
        self.model_type = model_type
        self.fps = fps
        self.smoothing_window = smoothing_window

        # Default velocity thresholds (pixels/frame)
        self.velocity_thresholds = velocity_thresholds or {
            "resting_max": 0.5,
            "walking_max": 5.0,
            "running_min": 5.0,
        }

        # Reference keypoint for center tracking
        self.center_keypoint = self._get_center_keypoint()

    def _get_center_keypoint(self) -> str:
        """Get the reference keypoint for center of mass."""
        if self.model_type == "topviewmouse":
            return "mouse_center"
        elif self.model_type == "quadruped":
            return "back_middle"
        return "nose"

    def analyze(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        confidence_threshold: float = 0.5,
    ) -> BehaviorMetrics:
        """
        Perform comprehensive behavior analysis.

        Args:
            keypoints: Array of shape (num_frames, num_keypoints, 3) [x, y, conf]
            keypoint_names: List of keypoint names
            confidence_threshold: Minimum confidence for valid keypoints

        Returns:
            BehaviorMetrics with all analysis results
        """
        logger.info(f"Analyzing behavior for {len(keypoints)} frames")

        # Get center trajectory
        trajectory = self._get_trajectory(keypoints, keypoint_names, confidence_threshold)

        # Compute motion metrics
        velocity = self._compute_velocity(trajectory)
        acceleration = self._compute_acceleration(velocity)
        distance_traveled = self._compute_total_distance(trajectory)

        # Classify behaviors
        behavior_labels = self._classify_behaviors(
            keypoints, keypoint_names, velocity, confidence_threshold
        )

        # Generate summary
        behavior_summary = self._compute_summary(
            velocity, acceleration, distance_traveled, behavior_labels
        )

        return BehaviorMetrics(
            velocity=velocity,
            acceleration=acceleration,
            distance_traveled=distance_traveled,
            trajectory=trajectory,
            behavior_labels=behavior_labels,
            behavior_summary=behavior_summary,
        )

    def _get_trajectory(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        confidence_threshold: float,
    ) -> np.ndarray:
        """Extract center-of-mass trajectory."""
        # Find center keypoint index
        try:
            center_idx = keypoint_names.index(self.center_keypoint)
        except ValueError:
            # Fallback: use mean of all keypoints
            logger.warning(f"Center keypoint '{self.center_keypoint}' not found, using mean")
            mask = keypoints[:, :, 2] > confidence_threshold
            trajectory = np.zeros((len(keypoints), 2))
            for i in range(len(keypoints)):
                if mask[i].any():
                    trajectory[i, 0] = np.mean(keypoints[i, mask[i], 0])
                    trajectory[i, 1] = np.mean(keypoints[i, mask[i], 1])
            return trajectory

        trajectory = keypoints[:, center_idx, :2].copy()

        # Interpolate low-confidence points
        confidence = keypoints[:, center_idx, 2]
        low_conf = confidence < confidence_threshold

        if low_conf.any() and not low_conf.all():
            for dim in range(2):
                trajectory[low_conf, dim] = np.interp(
                    np.where(low_conf)[0],
                    np.where(~low_conf)[0],
                    trajectory[~low_conf, dim],
                )

        return trajectory

    def _compute_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute instantaneous velocity (pixels/frame)."""
        # First-order difference
        velocity = np.sqrt(
            np.diff(trajectory[:, 0], prepend=trajectory[0, 0]) ** 2 +
            np.diff(trajectory[:, 1], prepend=trajectory[0, 1]) ** 2
        )

        # Smooth velocity
        if len(velocity) > self.smoothing_window:
            velocity = savgol_filter(
                velocity,
                min(self.smoothing_window, len(velocity) // 2 * 2 - 1),
                2,
            )

        return velocity

    def _compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Compute instantaneous acceleration."""
        acceleration = np.diff(velocity, prepend=velocity[0])

        if len(acceleration) > self.smoothing_window:
            acceleration = savgol_filter(
                acceleration,
                min(self.smoothing_window, len(acceleration) // 2 * 2 - 1),
                2,
            )

        return acceleration

    def _compute_total_distance(self, trajectory: np.ndarray) -> float:
        """Compute total distance traveled."""
        distances = np.sqrt(
            np.diff(trajectory[:, 0]) ** 2 + np.diff(trajectory[:, 1]) ** 2
        )
        return float(np.sum(distances))

    def _classify_behaviors(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        velocity: np.ndarray,
        confidence_threshold: float,
    ) -> np.ndarray:
        """Classify behavior for each frame."""
        num_frames = len(keypoints)
        labels = np.full(num_frames, self.BEHAVIOR_TYPES["unknown"])

        # Basic velocity-based classification
        for i in range(num_frames):
            v = velocity[i]
            if v < self.velocity_thresholds["resting_max"]:
                labels[i] = self.BEHAVIOR_TYPES["resting"]
            elif v < self.velocity_thresholds["walking_max"]:
                labels[i] = self.BEHAVIOR_TYPES["walking"]
            else:
                labels[i] = self.BEHAVIOR_TYPES["running"]

        # Model-specific behavior detection
        if self.model_type == "topviewmouse":
            labels = self._detect_mouse_behaviors(
                keypoints, keypoint_names, labels, confidence_threshold
            )

        return labels

    def _detect_mouse_behaviors(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        labels: np.ndarray,
        confidence_threshold: float,
    ) -> np.ndarray:
        """Detect mouse-specific behaviors like grooming and rearing."""
        try:
            # Get relevant keypoint indices
            nose_idx = keypoint_names.index("nose")
            left_paw_idx = keypoint_names.index("left_front_paw")
            right_paw_idx = keypoint_names.index("right_front_paw")

            for i in range(len(keypoints)):
                # Check confidence
                if (keypoints[i, nose_idx, 2] < confidence_threshold or
                    keypoints[i, left_paw_idx, 2] < confidence_threshold or
                    keypoints[i, right_paw_idx, 2] < confidence_threshold):
                    continue

                # Grooming: paws close to nose
                nose = keypoints[i, nose_idx, :2]
                left_paw = keypoints[i, left_paw_idx, :2]
                right_paw = keypoints[i, right_paw_idx, :2]

                left_dist = np.linalg.norm(nose - left_paw)
                right_dist = np.linalg.norm(nose - right_paw)

                # If both paws are close to nose and animal is relatively still
                if left_dist < 20 and right_dist < 20 and labels[i] == self.BEHAVIOR_TYPES["resting"]:
                    labels[i] = self.BEHAVIOR_TYPES["grooming"]

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not detect mouse-specific behaviors: {e}")

        return labels

    def _compute_summary(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        distance_traveled: float,
        behavior_labels: np.ndarray,
    ) -> Dict:
        """Compute summary statistics."""
        # Behavior durations (in frames)
        behavior_counts = {}
        for name, code in self.BEHAVIOR_TYPES.items():
            count = np.sum(behavior_labels == code)
            if count > 0:
                behavior_counts[name] = {
                    "frames": int(count),
                    "duration_sec": count / self.fps,
                    "percentage": 100.0 * count / len(behavior_labels),
                }

        return {
            "total_frames": len(velocity),
            "total_duration_sec": len(velocity) / self.fps,
            "distance_traveled_px": distance_traveled,
            "mean_velocity": float(np.mean(velocity)),
            "max_velocity": float(np.max(velocity)),
            "std_velocity": float(np.std(velocity)),
            "mean_acceleration": float(np.mean(np.abs(acceleration))),
            "behaviors": behavior_counts,
        }

    def get_behavior_name(self, code: int) -> str:
        """Get behavior name from code."""
        for name, c in self.BEHAVIOR_TYPES.items():
            if c == code:
                return name
        return "unknown"

    def to_dataframe(self, metrics: BehaviorMetrics) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        data = {
            "frame": np.arange(len(metrics.velocity)),
            "x": metrics.trajectory[:, 0],
            "y": metrics.trajectory[:, 1],
            "velocity": metrics.velocity,
            "acceleration": metrics.acceleration,
            "behavior_code": metrics.behavior_labels,
            "behavior_name": [self.get_behavior_name(c) for c in metrics.behavior_labels],
        }
        return pd.DataFrame(data)


def estimate_body_size(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    model_type: str = "topviewmouse",
    confidence_threshold: float = 0.3,
) -> Dict:
    """
    Estimate body size from keypoints.

    Body size is estimated as the distance between head and tail,
    or the bounding box of all keypoints.

    Args:
        keypoints: Array of shape (num_frames, num_keypoints, 3)
        keypoint_names: List of keypoint names
        model_type: Model type for selecting reference points
        confidence_threshold: Minimum confidence

    Returns:
        Dictionary with body size statistics (mean, std, min, max, per_frame)
    """
    # Define reference keypoints for each model type
    reference_pairs = {
        "topviewmouse": [("nose", "tail_base"), ("nose", "tail_end")],
        "quadruped": [("nose", "tail_base"), ("neck_base", "tail_base")],
        "human": [("nose", "left_hip"), ("nose", "right_hip")],
    }

    pairs = reference_pairs.get(model_type, reference_pairs["topviewmouse"])

    body_sizes = []

    for frame_kp in keypoints:
        frame_sizes = []

        # Try reference pairs
        for kp1_name, kp2_name in pairs:
            if kp1_name in keypoint_names and kp2_name in keypoint_names:
                idx1 = keypoint_names.index(kp1_name)
                idx2 = keypoint_names.index(kp2_name)

                if (frame_kp[idx1, 2] > confidence_threshold and
                    frame_kp[idx2, 2] > confidence_threshold):
                    dist = np.sqrt(
                        (frame_kp[idx1, 0] - frame_kp[idx2, 0]) ** 2 +
                        (frame_kp[idx1, 1] - frame_kp[idx2, 1]) ** 2
                    )
                    frame_sizes.append(dist)

        # Fallback: bounding box diagonal
        if not frame_sizes:
            valid_mask = frame_kp[:, 2] > confidence_threshold
            if valid_mask.sum() >= 2:
                valid_points = frame_kp[valid_mask, :2]
                bbox_diag = np.sqrt(
                    (valid_points[:, 0].max() - valid_points[:, 0].min()) ** 2 +
                    (valid_points[:, 1].max() - valid_points[:, 1].min()) ** 2
                )
                frame_sizes.append(bbox_diag)

        if frame_sizes:
            body_sizes.append(np.mean(frame_sizes))

    if not body_sizes:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "per_frame": []}

    body_sizes = np.array(body_sizes)

    return {
        "mean": float(np.mean(body_sizes)),
        "std": float(np.std(body_sizes)),
        "min": float(np.min(body_sizes)),
        "max": float(np.max(body_sizes)),
        "median": float(np.median(body_sizes)),
        "per_frame": body_sizes.tolist(),
    }


def compute_body_area(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    confidence_threshold: float = 0.3,
) -> Dict:
    """
    Estimate body area from keypoint convex hull.

    Args:
        keypoints: Array of shape (num_frames, num_keypoints, 3)
        keypoint_names: List of keypoint names
        confidence_threshold: Minimum confidence

    Returns:
        Dictionary with body area statistics
    """
    from scipy.spatial import ConvexHull

    areas = []

    for frame_kp in keypoints:
        valid_mask = frame_kp[:, 2] > confidence_threshold
        if valid_mask.sum() >= 3:  # Need at least 3 points for hull
            try:
                points = frame_kp[valid_mask, :2]
                hull = ConvexHull(points)
                areas.append(hull.volume)  # 2D volume = area
            except Exception:
                pass

    if not areas:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}

    areas = np.array(areas)

    return {
        "mean": float(np.mean(areas)),
        "std": float(np.std(areas)),
        "min": float(np.min(areas)),
        "max": float(np.max(areas)),
        "median": float(np.median(areas)),
    }
