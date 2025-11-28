"""
Baseline Models for Action Classification.

Provides simple baseline methods to compare against SuperAnimal-based classification.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Container for baseline classification results."""
    action_labels: np.ndarray  # (frames,) action labels
    action_summary: Dict       # Summary statistics
    method: str               # Baseline method name


class RandomBaseline:
    """
    Random baseline: assigns random actions with fixed distribution.

    This is the simplest baseline - any useful model should beat this.
    """

    def __init__(
        self,
        action_distribution: Dict[str, float] = None,
        seed: int = 42,
    ):
        """
        Initialize random baseline.

        Args:
            action_distribution: Dict of action -> probability
            seed: Random seed for reproducibility
        """
        self.action_distribution = action_distribution or {
            "stationary": 0.4,
            "walking": 0.4,
            "running": 0.2,
        }
        self.seed = seed
        self.action_names = list(self.action_distribution.keys())
        self.probabilities = list(self.action_distribution.values())

    def predict(self, num_frames: int) -> BaselineMetrics:
        """Generate random action predictions."""
        np.random.seed(self.seed)

        action_indices = np.random.choice(
            len(self.action_names),
            size=num_frames,
            p=self.probabilities,
        )

        # Create summary
        summary = self._compute_summary(action_indices, num_frames)

        return BaselineMetrics(
            action_labels=action_indices,
            action_summary=summary,
            method="random",
        )

    def _compute_summary(self, labels: np.ndarray, total: int) -> Dict:
        """Compute action summary statistics."""
        actions = {}
        for i, name in enumerate(self.action_names):
            count = np.sum(labels == i)
            actions[name] = {
                "frames": int(count),
                "percentage": float(count / total * 100),
            }

        return {
            "total_frames": total,
            "actions": actions,
        }


class MajorityBaseline:
    """
    Majority class baseline: predicts the most common action for all frames.

    If ground truth is available, uses majority from ground truth.
    Otherwise, defaults to "stationary".
    """

    def __init__(self, default_action: str = "stationary"):
        """
        Initialize majority baseline.

        Args:
            default_action: Default action when no ground truth available
        """
        self.default_action = default_action
        self.action_names = ["stationary", "walking", "running"]

    def predict(
        self,
        num_frames: int,
        ground_truth: np.ndarray = None,
    ) -> BaselineMetrics:
        """
        Generate majority class predictions.

        Args:
            num_frames: Number of frames to predict
            ground_truth: Optional ground truth labels
        """
        if ground_truth is not None:
            # Find majority class from ground truth
            unique, counts = np.unique(ground_truth, return_counts=True)
            majority_idx = unique[np.argmax(counts)]
        else:
            # Use default action
            majority_idx = self.action_names.index(self.default_action)

        action_labels = np.full(num_frames, majority_idx, dtype=int)
        summary = self._compute_summary(action_labels, num_frames)

        return BaselineMetrics(
            action_labels=action_labels,
            action_summary=summary,
            method="majority",
        )

    def _compute_summary(self, labels: np.ndarray, total: int) -> Dict:
        """Compute action summary statistics."""
        actions = {}
        for i, name in enumerate(self.action_names):
            count = np.sum(labels == i)
            actions[name] = {
                "frames": int(count),
                "percentage": float(count / total * 100),
            }

        return {
            "total_frames": total,
            "actions": actions,
        }


class SimpleThresholdBaseline:
    """
    Simple threshold baseline: uses fixed pixel velocity threshold.

    Unlike UnifiedActionClassifier, this does NOT normalize by body size.
    This tests whether body-size normalization actually helps.
    """

    def __init__(
        self,
        stationary_threshold: float = 2.0,  # pixels/frame
        walking_threshold: float = 10.0,    # pixels/frame
        smoothing_window: int = 5,
    ):
        """
        Initialize simple threshold baseline.

        Args:
            stationary_threshold: Velocity below this = stationary
            walking_threshold: Velocity above this = running
            smoothing_window: Window for velocity smoothing
        """
        self.stationary_threshold = stationary_threshold
        self.walking_threshold = walking_threshold
        self.smoothing_window = smoothing_window
        self.action_names = ["stationary", "walking", "running"]

    def predict(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
    ) -> BaselineMetrics:
        """
        Predict actions using simple velocity thresholds.

        Args:
            keypoints: (frames, keypoints, 3) array
            keypoint_names: List of keypoint names
        """
        # Extract center trajectory (no body-size normalization)
        trajectory = self._extract_center(keypoints, keypoint_names)

        # Compute raw velocity (pixels/frame)
        velocity = self._compute_velocity(trajectory)

        # Classify based on fixed thresholds
        action_labels = self._classify(velocity)

        summary = self._compute_summary(action_labels, len(action_labels))

        return BaselineMetrics(
            action_labels=action_labels,
            action_summary=summary,
            method="simple_threshold",
        )

    def _extract_center(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
    ) -> np.ndarray:
        """Extract center point trajectory."""
        # Try common center keypoints
        center_candidates = ["mouse_center", "back_middle", "mid_back", "neck"]

        for candidate in center_candidates:
            if candidate in keypoint_names:
                idx = keypoint_names.index(candidate)
                return keypoints[:, idx, :2]

        # Fallback: use mean of all keypoints
        valid_mask = keypoints[:, :, 2] > 0.3
        trajectory = np.zeros((len(keypoints), 2))
        for i in range(len(keypoints)):
            if valid_mask[i].any():
                trajectory[i] = keypoints[i, valid_mask[i], :2].mean(axis=0)

        return trajectory

    def _compute_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute velocity magnitude."""
        diff = np.diff(trajectory, axis=0)
        velocity = np.sqrt(np.sum(diff ** 2, axis=1))
        velocity = np.concatenate([[0], velocity])

        # Smooth
        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            velocity = np.convolve(velocity, kernel, mode='same')

        return velocity

    def _classify(self, velocity: np.ndarray) -> np.ndarray:
        """Classify actions based on velocity."""
        labels = np.ones(len(velocity), dtype=int)  # Default: walking
        labels[velocity < self.stationary_threshold] = 0  # stationary
        labels[velocity > self.walking_threshold] = 2     # running
        return labels

    def _compute_summary(self, labels: np.ndarray, total: int) -> Dict:
        """Compute action summary statistics."""
        actions = {}
        for i, name in enumerate(self.action_names):
            count = np.sum(labels == i)
            actions[name] = {
                "frames": int(count),
                "percentage": float(count / total * 100),
            }

        return {
            "total_frames": total,
            "actions": actions,
        }


class CentroidOnlyBaseline:
    """
    Centroid-only baseline: uses single centroid point instead of full skeleton.

    Tests whether full keypoint information adds value over simple centroid tracking.
    """

    def __init__(
        self,
        fps: float = 30.0,
        stationary_threshold: float = 0.5,  # body-lengths/sec
        walking_threshold: float = 3.0,
    ):
        """Initialize centroid-only baseline."""
        self.fps = fps
        self.stationary_threshold = stationary_threshold
        self.walking_threshold = walking_threshold
        self.action_names = ["stationary", "walking", "running"]

    def predict(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
    ) -> BaselineMetrics:
        """
        Predict using only centroid (mean of all keypoints).

        Args:
            keypoints: (frames, keypoints, 3) array
            keypoint_names: List of keypoint names
        """
        # Compute centroid (mean of all valid keypoints per frame)
        centroids = []
        for frame_kp in keypoints:
            valid_mask = frame_kp[:, 2] > 0.3
            if valid_mask.any():
                centroid = frame_kp[valid_mask, :2].mean(axis=0)
            else:
                centroid = np.array([0, 0])
            centroids.append(centroid)

        centroids = np.array(centroids)

        # Estimate body size from keypoint spread
        body_sizes = []
        for frame_kp in keypoints:
            valid_mask = frame_kp[:, 2] > 0.3
            if valid_mask.sum() >= 2:
                valid_points = frame_kp[valid_mask, :2]
                size = np.max(np.linalg.norm(
                    valid_points - valid_points.mean(axis=0), axis=1
                )) * 2
            else:
                size = 50  # default
            body_sizes.append(size)

        body_size = np.median(body_sizes)

        # Compute normalized velocity
        diff = np.diff(centroids, axis=0)
        velocity = np.sqrt(np.sum(diff ** 2, axis=1))
        velocity = np.concatenate([[0], velocity])
        normalized_velocity = velocity / body_size * self.fps

        # Classify
        labels = np.ones(len(velocity), dtype=int)
        labels[normalized_velocity < self.stationary_threshold] = 0
        labels[normalized_velocity > self.walking_threshold] = 2

        summary = self._compute_summary(labels, len(labels))

        return BaselineMetrics(
            action_labels=labels,
            action_summary=summary,
            method="centroid_only",
        )

    def _compute_summary(self, labels: np.ndarray, total: int) -> Dict:
        """Compute action summary statistics."""
        actions = {}
        for i, name in enumerate(self.action_names):
            count = np.sum(labels == i)
            actions[name] = {
                "frames": int(count),
                "percentage": float(count / total * 100),
            }

        return {
            "total_frames": total,
            "actions": actions,
        }


# ============================================================
# Baseline Registry
# ============================================================

BASELINES = {
    "random": RandomBaseline,
    "majority": MajorityBaseline,
    "simple_threshold": SimpleThresholdBaseline,
    "centroid_only": CentroidOnlyBaseline,
}


def get_baseline(name: str, **kwargs):
    """Get baseline model by name."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINES.keys())}")
    return BASELINES[name](**kwargs)
