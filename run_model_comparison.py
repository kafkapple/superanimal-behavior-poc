#!/usr/bin/env python3
"""
Model Comparison Pipeline

Compare different pose estimation and action recognition models:
- SuperAnimal (DeepLabCut 3.0) - Primary model
- YOLO Pose (ultralytics) - Optional comparison
- Baselines (Random, Majority, SimpleThreshold)

Usage:
    python run_model_comparison.py                      # Compare all available
    python run_model_comparison.py --models superanimal,yolo_pose
    python run_model_comparison.py --video /path/to/video.mp4

Metrics computed:
- Keypoint: PCK (Percentage Correct Keypoints), OKS (Object Keypoint Similarity)
- Action: Accuracy, F1, Agreement Rate, Temporal Consistency
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logging, get_device

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check available dependencies and models."""
    available = {"superanimal": True}

    # Check YOLO
    try:
        from ultralytics import YOLO
        available["yolo_pose"] = True
        logger.info("✅ YOLO Pose available")
    except ImportError:
        available["yolo_pose"] = False
        logger.warning("❌ YOLO Pose not available (pip install ultralytics)")

    # Check MMPose
    try:
        import mmpose
        available["mmpose"] = True
        logger.info("✅ MMPose available")
    except ImportError:
        available["mmpose"] = False
        logger.info("ℹ️  MMPose not available (optional)")

    return available


def run_superanimal(video_path: Path, max_frames: int, device: str):
    """Run SuperAnimal model."""
    from src.models.predictor import SuperAnimalPredictor
    from src.models.action_classifier import UnifiedActionClassifier

    logger.info("Running SuperAnimal-TopViewMouse...")

    predictor = SuperAnimalPredictor(
        model_type="topviewmouse",
        model_name="hrnet_w32",
        device=device,
    )

    results = predictor.predict_video(
        video_path=video_path,
        max_frames=max_frames,
    )

    if results["keypoints"] is None:
        logger.error("No keypoints detected")
        return None

    keypoints = results["keypoints"]
    keypoint_names = predictor.get_keypoint_names()
    fps = results["metadata"].get("fps", 30.0)

    # Action classification
    classifier = UnifiedActionClassifier(species="mouse", fps=fps)
    metrics = classifier.analyze(keypoints, keypoint_names)

    return {
        "keypoints": keypoints,
        "keypoint_names": keypoint_names,
        "actions": metrics.action_labels,
        "action_summary": metrics.action_summary,
        "fps": fps,
    }


def run_yolo_pose(video_path: Path, max_frames: int, device: str):
    """Run YOLO Pose model."""
    try:
        from src.models.yolo_pose import YOLOPosePredictor
        from src.models.action_classifier import UnifiedActionClassifier

        logger.info("Running YOLO Pose...")

        predictor = YOLOPosePredictor(
            model_name="yolov8n-pose",
            device=device,
        )

        results = predictor.predict_video(
            video_path=video_path,
            max_frames=max_frames,
        )

        if results["keypoints"] is None:
            logger.warning("YOLO: No keypoints detected")
            return None

        keypoints = results["keypoints"]
        keypoint_names = results["keypoint_names"]
        fps = results["metadata"].get("fps", 30.0)

        # Action classification
        classifier = UnifiedActionClassifier(species="mouse", fps=fps)
        metrics = classifier.analyze(keypoints, keypoint_names)

        return {
            "keypoints": keypoints,
            "keypoint_names": keypoint_names,
            "actions": metrics.action_labels,
            "action_summary": metrics.action_summary,
            "fps": fps,
        }

    except Exception as e:
        logger.error(f"YOLO Pose failed: {e}")
        return None


def run_baselines(num_frames: int, reference_actions: np.ndarray = None):
    """Run baseline models."""
    from src.models.baseline import (
        RandomBaseline,
        MajorityBaseline,
    )

    logger.info("Running baseline models...")

    baselines = {}

    # Random baseline
    random_bl = RandomBaseline()
    random_result = random_bl.predict(num_frames)
    baselines["random"] = {
        "actions": random_result.action_labels,
        "action_summary": random_result.action_summary,
    }

    # Majority baseline
    majority_bl = MajorityBaseline()
    majority_result = majority_bl.predict(num_frames, reference_actions)
    baselines["majority"] = {
        "actions": majority_result.action_labels,
        "action_summary": majority_result.action_summary,
    }

    return baselines


def compute_metrics(model_results: dict, reference_name: str = "superanimal"):
    """Compute comparison metrics."""
    from src.evaluation.metrics import (
        compute_classification_metrics,
        compute_consistency_metrics,
    )
    from src.evaluation.model_comparison import (
        compute_pck,
        compute_oks,
        compute_action_agreement,
    )

    if reference_name not in model_results:
        logger.error(f"Reference model '{reference_name}' not in results")
        return {}

    reference = model_results[reference_name]
    ref_actions = reference["actions"]

    metrics = {"reference": reference_name}

    for model_name, result in model_results.items():
        if model_name == reference_name:
            continue

        model_metrics = {"model": model_name}

        # Action metrics
        if "actions" in result:
            actions = result["actions"]

            # Agreement with reference
            agreement = compute_action_agreement(actions, ref_actions)
            model_metrics["action_agreement"] = agreement

            # Classification metrics (using reference as "ground truth")
            clf = compute_classification_metrics(actions, ref_actions)
            model_metrics["accuracy"] = clf.accuracy
            model_metrics["f1_scores"] = clf.f1_score

            # Consistency
            consistency = compute_consistency_metrics(actions)
            model_metrics["consistency"] = consistency.smoothness_score

        # Keypoint metrics (if both have keypoints)
        if "keypoints" in result and "keypoints" in reference:
            ref_kps = reference["keypoints"]
            model_kps = result["keypoints"]

            # Ensure same length
            min_len = min(len(ref_kps), len(model_kps))
            ref_kps = ref_kps[:min_len]
            model_kps = model_kps[:min_len]

            # Handle different keypoint counts
            min_kps = min(ref_kps.shape[1], model_kps.shape[1])
            ref_kps = ref_kps[:, :min_kps]
            model_kps = model_kps[:, :min_kps]

            # PCK and OKS
            pck = compute_pck(model_kps, ref_kps)
            oks = compute_oks(model_kps, ref_kps)

            model_metrics["pck"] = pck
            model_metrics["oks"] = oks

        metrics[model_name] = model_metrics

    return metrics


def print_results(metrics: dict):
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nReference Model: {metrics.get('reference', 'N/A')}")

    print("\n📊 Action Recognition Comparison:")
    print("-" * 50)
    print(f"{'Model':<25} {'Agreement':>10} {'Accuracy':>10} {'Consistency':>12}")
    print("-" * 50)

    for name, data in metrics.items():
        if name == "reference" or not isinstance(data, dict):
            continue

        agreement = data.get("action_agreement", 0) * 100
        accuracy = data.get("accuracy", 0) * 100
        consistency = data.get("consistency", 0)

        print(f"{name:<25} {agreement:>9.1f}% {accuracy:>9.1f}% {consistency:>11.2f}")

    print("-" * 50)

    # Keypoint metrics if available
    has_keypoint_metrics = any(
        "oks" in data for name, data in metrics.items()
        if isinstance(data, dict) and name != "reference"
    )

    if has_keypoint_metrics:
        print("\n📍 Keypoint Detection Comparison:")
        print("-" * 50)
        print(f"{'Model':<25} {'OKS':>10} {'PCK@0.1':>10} {'PCK@0.2':>10}")
        print("-" * 50)

        for name, data in metrics.items():
            if name == "reference" or not isinstance(data, dict):
                continue
            if "oks" not in data:
                continue

            oks = data.get("oks", 0)
            pck = data.get("pck", {})
            pck_01 = pck.get("PCK@0.1", 0)
            pck_02 = pck.get("PCK@0.2", 0)

            print(f"{name:<25} {oks:>9.3f} {pck_01:>9.3f} {pck_02:>9.3f}")

        print("-" * 50)

    print("\n" + "=" * 70)


def save_results(metrics: dict, output_dir: Path):
    """Save results to JSON."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output_path = output_dir / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(convert(metrics), f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare pose estimation and action recognition models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to video file (default: download sample)",
    )

    parser.add_argument(
        "--models", "-m",
        type=str,
        default=None,
        help="Comma-separated list of models (default: all available)",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum frames to process",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/model_comparison",
        help="Output directory",
    )

    parser.add_argument(
        "--include-baselines",
        action="store_true",
        default=True,
        help="Include baseline models in comparison",
    )

    args = parser.parse_args()

    # Setup
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("Model Comparison Pipeline")
    logger.info("=" * 60)

    # Check dependencies
    available = check_dependencies()

    # Determine models to run
    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",")]
    else:
        models_to_run = [m for m, avail in available.items() if avail]

    logger.info(f"Models to compare: {models_to_run}")

    # Get video
    if args.video:
        video_path = Path(args.video)
    else:
        from src.data.downloader import VideoDownloader
        downloader = VideoDownloader("data")
        video_path = downloader.download_sample("mouse_topview")

    if not video_path or not video_path.exists():
        logger.error("Video not found")
        return

    logger.info(f"Video: {video_path}")
    logger.info(f"Max frames: {args.max_frames}")

    device = get_device(args.device)
    logger.info(f"Device: {device}")

    # Run models
    model_results = {}

    # SuperAnimal (always run as reference)
    if "superanimal" in models_to_run:
        result = run_superanimal(video_path, args.max_frames, device)
        if result:
            model_results["superanimal"] = result

    # YOLO Pose
    if "yolo_pose" in models_to_run and available.get("yolo_pose"):
        result = run_yolo_pose(video_path, args.max_frames, device)
        if result:
            model_results["yolo_pose"] = result

    # Baselines
    if args.include_baselines and model_results:
        ref_actions = model_results.get("superanimal", {}).get("actions")
        num_frames = len(ref_actions) if ref_actions is not None else args.max_frames

        baselines = run_baselines(num_frames, ref_actions)
        model_results.update(baselines)

    # Compute metrics
    if len(model_results) >= 2:
        metrics = compute_metrics(model_results, reference_name="superanimal")

        # Print results
        print_results(metrics)

        # Save results
        output_dir = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(metrics, output_dir)
    else:
        logger.warning("Need at least 2 models to compare")

    logger.info("\nModel comparison complete!")


if __name__ == "__main__":
    main()
