"""
Model Comparison Module.

Compare different pose estimation and action recognition models:
- SuperAnimal (DeepLabCut 3.0)
- YOLO Pose
- MMPose (optional)
- Custom baselines

Metrics:
- Keypoint Detection: PCK (Percentage of Correct Keypoints), OKS (Object Keypoint Similarity)
- Action Recognition: Accuracy, F1, Consistency
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json

logger = logging.getLogger(__name__)


# ============================================================
# Keypoint Evaluation Metrics
# ============================================================

@dataclass
class KeypointMetrics:
    """Metrics for keypoint detection evaluation."""
    pck: Dict[str, float]       # PCK at different thresholds
    mean_pck: float             # Mean PCK
    oks: float                  # Object Keypoint Similarity
    detection_rate: float       # % of frames with valid detections
    mean_confidence: float      # Mean keypoint confidence
    num_keypoints: int          # Number of keypoints


@dataclass
class ActionMetricsComparison:
    """Metrics for action recognition comparison."""
    accuracy: float
    per_class_accuracy: Dict[str, float]
    f1_scores: Dict[str, float]
    confusion_matrix: np.ndarray
    consistency_score: float    # Temporal consistency
    agreement_rate: float       # Agreement with reference


@dataclass
class ModelComparisonResult:
    """Complete comparison result between models."""
    model_a: str
    model_b: str
    keypoint_comparison: Optional[Dict] = None
    action_comparison: Optional[Dict] = None
    summary: Dict = field(default_factory=dict)
    winner: str = ""


def compute_pck(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold_ratios: List[float] = [0.05, 0.1, 0.2],
    normalize_by: str = "bbox",
    bbox_size: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute Percentage of Correct Keypoints (PCK).

    Args:
        predictions: Predicted keypoints (frames, keypoints, 2 or 3)
        ground_truth: Ground truth keypoints (frames, keypoints, 2 or 3)
        threshold_ratios: List of threshold ratios
        normalize_by: "bbox" or "torso"
        bbox_size: Bounding box size for normalization

    Returns:
        Dictionary of PCK values at each threshold
    """
    if predictions.shape[:2] != ground_truth.shape[:2]:
        logger.warning("Shape mismatch between predictions and ground truth")
        min_frames = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_frames]
        ground_truth = ground_truth[:min_frames]

    # Extract x, y coordinates
    pred_xy = predictions[:, :, :2]
    gt_xy = ground_truth[:, :, :2]

    # Compute distances
    distances = np.linalg.norm(pred_xy - gt_xy, axis=2)  # (frames, keypoints)

    # Determine normalization factor
    if bbox_size is None:
        # Estimate from data
        valid_gt = gt_xy[gt_xy[:, :, 0] > 0]
        if len(valid_gt) > 0:
            bbox_size = np.max(valid_gt) - np.min(valid_gt)
        else:
            bbox_size = 100  # default

    # Compute PCK at each threshold
    pck_results = {}
    for ratio in threshold_ratios:
        threshold = ratio * bbox_size
        correct = distances < threshold

        # Only count keypoints with valid ground truth
        valid_mask = (gt_xy[:, :, 0] > 0) & (gt_xy[:, :, 1] > 0)
        if valid_mask.sum() > 0:
            pck = correct[valid_mask].mean()
        else:
            pck = 0.0

        pck_results[f"PCK@{ratio}"] = float(pck)

    return pck_results


def compute_oks(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    sigmas: Optional[np.ndarray] = None,
    area: Optional[float] = None,
) -> float:
    """
    Compute Object Keypoint Similarity (OKS).

    OKS is the standard metric used in COCO keypoint evaluation.

    Args:
        predictions: Predicted keypoints (frames, keypoints, 2 or 3)
        ground_truth: Ground truth keypoints (frames, keypoints, 2 or 3)
        sigmas: Per-keypoint variance (default: uniform)
        area: Object area for normalization

    Returns:
        Mean OKS score
    """
    num_keypoints = predictions.shape[1]

    if sigmas is None:
        # Default sigmas (uniform)
        sigmas = np.ones(num_keypoints) * 0.05

    # Extract coordinates
    pred_xy = predictions[:, :, :2]
    gt_xy = ground_truth[:, :, :2]

    # Compute squared distances
    d2 = np.sum((pred_xy - gt_xy) ** 2, axis=2)  # (frames, keypoints)

    # Estimate area if not provided
    if area is None:
        valid_gt = gt_xy[gt_xy[:, :, 0] > 0]
        if len(valid_gt) > 0:
            x_range = np.max(valid_gt[:, 0]) - np.min(valid_gt[:, 0])
            y_range = np.max(valid_gt[:, 1]) - np.min(valid_gt[:, 1])
            area = x_range * y_range
        else:
            area = 10000  # default

    # Compute OKS per keypoint
    # OKS = exp(-d^2 / (2 * area * sigma^2))
    vars = (sigmas * 2) ** 2
    oks_per_kp = np.exp(-d2 / (2 * area * vars))

    # Visibility mask
    if predictions.shape[2] > 2:
        vis_mask = predictions[:, :, 2] > 0.3
    else:
        vis_mask = np.ones_like(oks_per_kp, dtype=bool)

    if gt_xy.shape[2] > 2:
        gt_vis = ground_truth[:, :, 2] > 0
        vis_mask = vis_mask & gt_vis

    # Mean OKS
    if vis_mask.sum() > 0:
        mean_oks = oks_per_kp[vis_mask].mean()
    else:
        mean_oks = 0.0

    return float(mean_oks)


def compute_keypoint_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    keypoint_names: List[str] = None,
) -> KeypointMetrics:
    """
    Compute comprehensive keypoint metrics.

    Args:
        predictions: Predicted keypoints (frames, keypoints, 3)
        ground_truth: Ground truth keypoints (frames, keypoints, 3)
        keypoint_names: Names of keypoints

    Returns:
        KeypointMetrics object
    """
    # PCK at multiple thresholds
    pck = compute_pck(predictions, ground_truth)

    # OKS
    oks = compute_oks(predictions, ground_truth)

    # Detection rate
    if predictions.shape[2] > 2:
        valid_preds = predictions[:, :, 2] > 0.3
        detection_rate = valid_preds.any(axis=1).mean()
        mean_confidence = predictions[:, :, 2][valid_preds].mean() if valid_preds.any() else 0
    else:
        detection_rate = 1.0
        mean_confidence = 1.0

    return KeypointMetrics(
        pck=pck,
        mean_pck=np.mean(list(pck.values())),
        oks=oks,
        detection_rate=detection_rate,
        mean_confidence=mean_confidence,
        num_keypoints=predictions.shape[1],
    )


# ============================================================
# Action Recognition Comparison
# ============================================================

def compute_action_agreement(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
) -> float:
    """
    Compute agreement rate between two action predictions.

    Args:
        predictions_a: Action labels from model A
        predictions_b: Action labels from model B

    Returns:
        Agreement rate (0-1)
    """
    min_len = min(len(predictions_a), len(predictions_b))
    return float(np.mean(predictions_a[:min_len] == predictions_b[:min_len]))


def compute_action_comparison(
    predictions: np.ndarray,
    reference: np.ndarray,
    class_names: List[str] = None,
) -> ActionMetricsComparison:
    """
    Compare action predictions against reference.

    Args:
        predictions: Predicted action labels
        reference: Reference action labels
        class_names: Action class names

    Returns:
        ActionMetricsComparison object
    """
    from src.evaluation.metrics import (
        compute_classification_metrics,
        compute_consistency_metrics,
    )

    if class_names is None:
        class_names = ["stationary", "walking", "running"]

    # Ensure same length
    min_len = min(len(predictions), len(reference))
    predictions = predictions[:min_len]
    reference = reference[:min_len]

    # Classification metrics
    clf_metrics = compute_classification_metrics(predictions, reference, class_names)

    # Consistency
    consistency = compute_consistency_metrics(predictions)

    # Agreement
    agreement = compute_action_agreement(predictions, reference)

    return ActionMetricsComparison(
        accuracy=clf_metrics.accuracy,
        per_class_accuracy=clf_metrics.per_class_accuracy,
        f1_scores=clf_metrics.f1_score,
        confusion_matrix=clf_metrics.confusion_matrix,
        consistency_score=consistency.smoothness_score,
        agreement_rate=agreement,
    )


# ============================================================
# Full Model Comparison
# ============================================================

class ModelComparator:
    """Compare multiple pose estimation and action recognition models."""

    def __init__(self, output_dir: Path = None):
        """Initialize comparator."""
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/model_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def compare_keypoints(
        self,
        model_a_name: str,
        model_a_keypoints: np.ndarray,
        model_b_name: str,
        model_b_keypoints: np.ndarray,
        reference_keypoints: np.ndarray = None,
    ) -> Dict:
        """
        Compare keypoint predictions from two models.

        Args:
            model_a_name: Name of model A
            model_a_keypoints: Keypoints from model A
            model_b_name: Name of model B
            model_b_keypoints: Keypoints from model B
            reference_keypoints: Optional ground truth/reference

        Returns:
            Comparison results dictionary
        """
        comparison = {
            "model_a": model_a_name,
            "model_b": model_b_name,
        }

        # If reference available, compute metrics against reference
        if reference_keypoints is not None:
            metrics_a = compute_keypoint_metrics(model_a_keypoints, reference_keypoints)
            metrics_b = compute_keypoint_metrics(model_b_keypoints, reference_keypoints)

            comparison["model_a_metrics"] = {
                "pck": metrics_a.pck,
                "mean_pck": metrics_a.mean_pck,
                "oks": metrics_a.oks,
                "detection_rate": metrics_a.detection_rate,
            }
            comparison["model_b_metrics"] = {
                "pck": metrics_b.pck,
                "mean_pck": metrics_b.mean_pck,
                "oks": metrics_b.oks,
                "detection_rate": metrics_b.detection_rate,
            }

            # Determine winner
            if metrics_a.mean_pck > metrics_b.mean_pck:
                comparison["keypoint_winner"] = model_a_name
            else:
                comparison["keypoint_winner"] = model_b_name
        else:
            # Cross-comparison (use one as reference for other)
            metrics_ab = compute_keypoint_metrics(model_a_keypoints, model_b_keypoints)
            comparison["cross_agreement"] = {
                "pck": metrics_ab.pck,
                "oks": metrics_ab.oks,
            }

        return comparison

    def compare_actions(
        self,
        model_a_name: str,
        model_a_actions: np.ndarray,
        model_b_name: str,
        model_b_actions: np.ndarray,
        reference_actions: np.ndarray = None,
        class_names: List[str] = None,
    ) -> Dict:
        """
        Compare action predictions from two models.

        Args:
            model_a_name: Name of model A
            model_a_actions: Actions from model A
            model_b_name: Name of model B
            model_b_actions: Actions from model B
            reference_actions: Optional ground truth
            class_names: Action class names

        Returns:
            Comparison results dictionary
        """
        if class_names is None:
            class_names = ["stationary", "walking", "running"]

        comparison = {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "class_names": class_names,
        }

        # Agreement between models
        agreement = compute_action_agreement(model_a_actions, model_b_actions)
        comparison["inter_model_agreement"] = agreement

        # If reference available
        if reference_actions is not None:
            metrics_a = compute_action_comparison(model_a_actions, reference_actions, class_names)
            metrics_b = compute_action_comparison(model_b_actions, reference_actions, class_names)

            comparison["model_a_metrics"] = {
                "accuracy": metrics_a.accuracy,
                "f1_scores": metrics_a.f1_scores,
                "consistency": metrics_a.consistency_score,
            }
            comparison["model_b_metrics"] = {
                "accuracy": metrics_b.accuracy,
                "f1_scores": metrics_b.f1_scores,
                "consistency": metrics_b.consistency_score,
            }

            # Winner
            if metrics_a.accuracy > metrics_b.accuracy:
                comparison["action_winner"] = model_a_name
            else:
                comparison["action_winner"] = model_b_name

        return comparison

    def run_full_comparison(
        self,
        video_path: Path,
        models: List[str] = None,
        max_frames: int = 200,
        reference_model: str = "superanimal",
    ) -> Dict:
        """
        Run full comparison across multiple models.

        Args:
            video_path: Path to video
            models: List of models to compare
            max_frames: Max frames to process
            reference_model: Model to use as reference

        Returns:
            Full comparison results
        """
        from src.models.yolo_pose import check_available_models, YOLOPosePredictor
        from src.models.predictor import SuperAnimalPredictor
        from src.models.action_classifier import UnifiedActionClassifier

        if models is None:
            available = check_available_models()
            models = [m for m, avail in available.items() if avail]

        logger.info(f"Comparing models: {models}")

        results = {
            "video": str(video_path),
            "models": models,
            "reference": reference_model,
            "keypoint_results": {},
            "action_results": {},
            "summary": {},
        }

        # Run each model
        model_outputs = {}

        for model_name in models:
            logger.info(f"Running {model_name}...")

            try:
                if model_name == "superanimal":
                    predictor = SuperAnimalPredictor(
                        model_type="topviewmouse",
                        device="auto",
                    )
                    output = predictor.predict_video(
                        video_path, max_frames=max_frames
                    )
                    keypoints = output["keypoints"]
                    keypoint_names = predictor.get_keypoint_names()

                elif model_name == "yolo_pose":
                    predictor = YOLOPosePredictor(device="auto")
                    output = predictor.predict_video(
                        video_path, max_frames=max_frames
                    )
                    keypoints = output["keypoints"]
                    keypoint_names = output["keypoint_names"]

                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue

                if keypoints is None:
                    logger.warning(f"No keypoints from {model_name}")
                    continue

                # Run action classification
                classifier = UnifiedActionClassifier(species="mouse", fps=30.0)
                action_metrics = classifier.analyze(keypoints, keypoint_names)

                model_outputs[model_name] = {
                    "keypoints": keypoints,
                    "keypoint_names": keypoint_names,
                    "actions": action_metrics.action_labels,
                    "action_summary": action_metrics.action_summary,
                }

            except Exception as e:
                logger.error(f"Error running {model_name}: {e}")
                continue

        # Compare models
        if len(model_outputs) < 2:
            logger.warning("Need at least 2 models to compare")
            return results

        # Use reference model
        if reference_model in model_outputs:
            ref_kps = model_outputs[reference_model]["keypoints"]
            ref_actions = model_outputs[reference_model]["actions"]
        else:
            # Use first model as reference
            ref_name = list(model_outputs.keys())[0]
            ref_kps = model_outputs[ref_name]["keypoints"]
            ref_actions = model_outputs[ref_name]["actions"]

        # Compute comparisons
        for model_name, output in model_outputs.items():
            if model_name == reference_model:
                continue

            # Keypoint comparison
            kp_comparison = self.compare_keypoints(
                reference_model, ref_kps,
                model_name, output["keypoints"],
            )
            results["keypoint_results"][f"{reference_model}_vs_{model_name}"] = kp_comparison

            # Action comparison
            action_comparison = self.compare_actions(
                reference_model, ref_actions,
                model_name, output["actions"],
            )
            results["action_results"][f"{reference_model}_vs_{model_name}"] = action_comparison

        # Summary
        results["summary"] = {
            "num_models": len(model_outputs),
            "frames_processed": max_frames,
            "models_compared": list(model_outputs.keys()),
        }

        # Save results
        self.save_results(results)

        return results

    def save_results(self, results: Dict, filename: str = "model_comparison.json"):
        """Save comparison results."""
        output_path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert(results), f, indent=2)

        logger.info(f"Saved comparison results: {output_path}")

    def print_summary(self, results: Dict):
        """Print comparison summary."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)

        print(f"\nModels compared: {results['summary'].get('models_compared', [])}")
        print(f"Reference model: {results.get('reference', 'N/A')}")

        # Keypoint results
        if results.get("keypoint_results"):
            print("\n📍 Keypoint Detection:")
            print("-" * 40)
            for name, comp in results["keypoint_results"].items():
                if "cross_agreement" in comp:
                    oks = comp["cross_agreement"].get("oks", 0)
                    print(f"  {name}: OKS={oks:.3f}")

        # Action results
        if results.get("action_results"):
            print("\n🎬 Action Recognition:")
            print("-" * 40)
            for name, comp in results["action_results"].items():
                agreement = comp.get("inter_model_agreement", 0)
                print(f"  {name}: Agreement={agreement*100:.1f}%")

        print("\n" + "=" * 70)
