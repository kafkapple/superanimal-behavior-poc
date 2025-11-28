"""
Evaluation Metrics for Action Classification.

Provides quantitative metrics to compare different classification methods.
Supports ground truth label loading and per-keypoint-preset F1/accuracy calculation.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
import csv

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation metrics."""
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    class_names: List[str]
    # Additional metrics
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    # Temporal metrics
    segment_accuracy: float = 0.0
    transition_accuracy: float = 0.0


@dataclass
class ConsistencyMetrics:
    """Metrics for temporal consistency."""
    mean_segment_length: float  # Average frames per action segment
    num_transitions: int        # Number of action changes
    transition_rate: float      # Transitions per second
    smoothness_score: float     # Higher = smoother predictions


@dataclass
class ComparisonResult:
    """Result of comparing two methods."""
    method_a: str
    method_b: str
    metrics_a: ClassificationMetrics
    metrics_b: ClassificationMetrics
    consistency_a: ConsistencyMetrics
    consistency_b: ConsistencyMetrics
    winner: str  # Which method is better overall
    summary: Dict


def compute_classification_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: List[str] = None,
) -> ClassificationMetrics:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted labels (frames,)
        ground_truth: Ground truth labels (frames,)
        class_names: Names of classes

    Returns:
        ClassificationMetrics object
    """
    if class_names is None:
        class_names = ["stationary", "walking", "running"]

    num_classes = len(class_names)

    # Overall accuracy
    accuracy = np.mean(predictions == ground_truth)

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = ground_truth == i
        if mask.sum() > 0:
            per_class_acc[name] = np.mean(predictions[mask] == i)
        else:
            per_class_acc[name] = 0.0

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, gt in zip(predictions, ground_truth):
        if 0 <= pred < num_classes and 0 <= gt < num_classes:
            conf_matrix[gt, pred] += 1

    # Precision, Recall, F1
    precision = {}
    recall = {}
    f1_score = {}

    for i, name in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp

        precision[name] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[name] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[name] + recall[name] > 0:
            f1_score[name] = 2 * precision[name] * recall[name] / (precision[name] + recall[name])
        else:
            f1_score[name] = 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        per_class_accuracy=per_class_acc,
        confusion_matrix=conf_matrix,
        class_names=class_names,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )


def compute_consistency_metrics(
    labels: np.ndarray,
    fps: float = 30.0,
) -> ConsistencyMetrics:
    """
    Compute temporal consistency metrics.

    Args:
        labels: Action labels (frames,)
        fps: Video frame rate

    Returns:
        ConsistencyMetrics object
    """
    if len(labels) == 0:
        return ConsistencyMetrics(
            mean_segment_length=0,
            num_transitions=0,
            transition_rate=0,
            smoothness_score=0,
        )

    # Count transitions
    transitions = np.sum(np.diff(labels) != 0)

    # Compute segment lengths
    segment_lengths = []
    current_length = 1

    for i in range(1, len(labels)):
        if labels[i] == labels[i-1]:
            current_length += 1
        else:
            segment_lengths.append(current_length)
            current_length = 1
    segment_lengths.append(current_length)

    mean_segment = np.mean(segment_lengths) if segment_lengths else 0

    # Transition rate (per second)
    duration = len(labels) / fps
    transition_rate = transitions / duration if duration > 0 else 0

    # Smoothness score (penalize rapid transitions)
    # Higher score = smoother (fewer short segments)
    short_segments = sum(1 for s in segment_lengths if s < fps * 0.5)  # < 0.5 sec
    smoothness = 1 - (short_segments / len(segment_lengths)) if segment_lengths else 0

    return ConsistencyMetrics(
        mean_segment_length=mean_segment,
        num_transitions=int(transitions),
        transition_rate=transition_rate,
        smoothness_score=smoothness,
    )


def compare_methods(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    ground_truth: np.ndarray,
    method_a_name: str,
    method_b_name: str,
    class_names: List[str] = None,
    fps: float = 30.0,
) -> ComparisonResult:
    """
    Compare two classification methods.

    Args:
        predictions_a: Predictions from method A
        predictions_b: Predictions from method B
        ground_truth: Ground truth labels
        method_a_name: Name of method A
        method_b_name: Name of method B
        class_names: Class names
        fps: Video frame rate

    Returns:
        ComparisonResult object
    """
    metrics_a = compute_classification_metrics(predictions_a, ground_truth, class_names)
    metrics_b = compute_classification_metrics(predictions_b, ground_truth, class_names)

    consistency_a = compute_consistency_metrics(predictions_a, fps)
    consistency_b = compute_consistency_metrics(predictions_b, fps)

    # Determine winner based on accuracy and F1
    score_a = metrics_a.accuracy + np.mean(list(metrics_a.f1_score.values()))
    score_b = metrics_b.accuracy + np.mean(list(metrics_b.f1_score.values()))

    winner = method_a_name if score_a > score_b else method_b_name

    # Create summary
    summary = {
        "accuracy_diff": metrics_a.accuracy - metrics_b.accuracy,
        "f1_diff": {
            name: metrics_a.f1_score.get(name, 0) - metrics_b.f1_score.get(name, 0)
            for name in (class_names or ["stationary", "walking", "running"])
        },
        "consistency_diff": consistency_a.smoothness_score - consistency_b.smoothness_score,
        "winner": winner,
    }

    return ComparisonResult(
        method_a=method_a_name,
        method_b=method_b_name,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        consistency_a=consistency_a,
        consistency_b=consistency_b,
        winner=winner,
        summary=summary,
    )


def generate_pseudo_ground_truth(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    fps: float = 30.0,
) -> np.ndarray:
    """
    Generate pseudo ground truth labels for evaluation.

    Uses conservative thresholds to create "soft" ground truth
    when manual annotations are not available.

    Args:
        keypoints: (frames, keypoints, 3) array
        keypoint_names: Keypoint names
        fps: Frame rate

    Returns:
        Pseudo ground truth labels
    """
    from src.models.action_classifier import UnifiedActionClassifier

    # Use very conservative classifier as pseudo ground truth
    classifier = UnifiedActionClassifier(
        species="mouse",
        fps=fps,
        smoothing_window=15,  # Heavy smoothing for pseudo GT
    )

    # Make thresholds more conservative
    classifier.thresholds = {
        "stationary": 0.3,  # Lower threshold = more stationary
        "walking": 4.0,     # Higher threshold = less running
    }

    metrics = classifier.analyze(keypoints, keypoint_names)
    return metrics.action_labels


class ExperimentEvaluator:
    """Evaluate and compare multiple experiments."""

    def __init__(self, output_dir: Path):
        """Initialize evaluator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def add_result(
        self,
        experiment_name: str,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None,
        method_type: str = "model",  # "model" or "baseline"
        metadata: Dict = None,
    ):
        """Add experiment result for evaluation."""
        self.results[experiment_name] = {
            "predictions": predictions,
            "ground_truth": ground_truth,
            "method_type": method_type,
            "metadata": metadata or {},
        }

    def evaluate_all(
        self,
        class_names: List[str] = None,
        fps: float = 30.0,
    ) -> Dict:
        """
        Evaluate all experiments and generate comparison report.

        Returns:
            Dictionary with evaluation results
        """
        if class_names is None:
            class_names = ["stationary", "walking", "running"]

        evaluation = {
            "experiments": {},
            "comparisons": [],
            "ranking": [],
        }

        # Get shared ground truth (from first result or generate pseudo GT)
        ground_truth = None
        for name, data in self.results.items():
            if data["ground_truth"] is not None:
                ground_truth = data["ground_truth"]
                break

        # Evaluate each experiment
        for name, data in self.results.items():
            gt = data["ground_truth"] if data["ground_truth"] is not None else ground_truth

            if gt is None:
                logger.warning(f"No ground truth for {name}, skipping metrics")
                continue

            metrics = compute_classification_metrics(
                data["predictions"], gt, class_names
            )
            consistency = compute_consistency_metrics(data["predictions"], fps)

            evaluation["experiments"][name] = {
                "accuracy": metrics.accuracy,
                "per_class_accuracy": metrics.per_class_accuracy,
                "f1_score": metrics.f1_score,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "confusion_matrix": metrics.confusion_matrix.tolist(),
                "mean_segment_length": consistency.mean_segment_length,
                "num_transitions": consistency.num_transitions,
                "smoothness_score": consistency.smoothness_score,
                "method_type": data["method_type"],
            }

        # Compare models vs baselines
        models = [n for n, d in self.results.items() if d["method_type"] == "model"]
        baselines = [n for n, d in self.results.items() if d["method_type"] == "baseline"]

        for model in models:
            for baseline in baselines:
                if model in evaluation["experiments"] and baseline in evaluation["experiments"]:
                    model_acc = evaluation["experiments"][model]["accuracy"]
                    baseline_acc = evaluation["experiments"][baseline]["accuracy"]

                    evaluation["comparisons"].append({
                        "model": model,
                        "baseline": baseline,
                        "model_accuracy": model_acc,
                        "baseline_accuracy": baseline_acc,
                        "improvement": model_acc - baseline_acc,
                        "relative_improvement": (model_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0,
                    })

        # Rank by accuracy
        ranking = sorted(
            evaluation["experiments"].items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        evaluation["ranking"] = [
            {"rank": i + 1, "name": name, "accuracy": data["accuracy"]}
            for i, (name, data) in enumerate(ranking)
        ]

        return evaluation

    def save_report(self, evaluation: Dict, filename: str = "evaluation_report.json"):
        """Save evaluation report to JSON."""
        report_path = self.output_dir / filename
        with open(report_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        logger.info(f"Saved evaluation report: {report_path}")
        return report_path

    def print_summary(self, evaluation: Dict):
        """Print evaluation summary to console."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Ranking
        print("\n📊 Accuracy Ranking:")
        print("-" * 40)
        for item in evaluation["ranking"]:
            marker = "🥇" if item["rank"] == 1 else "🥈" if item["rank"] == 2 else "🥉" if item["rank"] == 3 else "  "
            print(f"  {marker} #{item['rank']}: {item['name']:30} {item['accuracy']*100:5.1f}%")

        # Model vs Baseline comparisons
        if evaluation["comparisons"]:
            print("\n📈 Model vs Baseline Improvements:")
            print("-" * 40)
            for comp in evaluation["comparisons"]:
                sign = "+" if comp["improvement"] > 0 else ""
                print(f"  {comp['model']} vs {comp['baseline']}: "
                      f"{sign}{comp['improvement']*100:.1f}% ({sign}{comp['relative_improvement']:.1f}% relative)")

        # Detailed metrics
        print("\n📋 Detailed Metrics:")
        print("-" * 40)
        for name, data in evaluation["experiments"].items():
            print(f"\n  {name} ({'Model' if data['method_type'] == 'model' else 'Baseline'}):")
            print(f"    Accuracy: {data['accuracy']*100:.1f}%")
            print(f"    F1 Scores: {', '.join(f'{k}: {v:.2f}' for k, v in data['f1_score'].items())}")
            print(f"    Smoothness: {data['smoothness_score']:.2f}")

        print("\n" + "=" * 70)


# ============================================================
# Ground Truth Label Loading
# ============================================================

ACTION_NAME_TO_ID = {
    "stationary": 0,
    "resting": 0,
    "still": 0,
    "walking": 1,
    "walk": 1,
    "locomotion": 1,
    "running": 2,
    "run": 2,
    "fast": 2,
}

ACTION_ID_TO_NAME = {
    0: "stationary",
    1: "walking",
    2: "running",
}


def load_ground_truth_labels(
    label_path: Union[str, Path],
    num_frames: Optional[int] = None,
    fps: float = 30.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Load ground truth behavior labels from various file formats.

    Supported formats:
    - CSV: frame,label or frame,start,end,label
    - JSON: {"labels": [...], "annotations": [...]}
    - TXT: space/comma separated per-frame labels

    Args:
        label_path: Path to label file
        num_frames: Expected number of frames (for validation/padding)
        fps: Video frame rate (for time-based annotations)

    Returns:
        Tuple of (labels array, metadata dict)
    """
    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    suffix = label_path.suffix.lower()
    metadata = {"source": str(label_path), "format": suffix}

    if suffix == ".csv":
        labels, meta = _load_csv_labels(label_path, num_frames, fps)
    elif suffix == ".json":
        labels, meta = _load_json_labels(label_path, num_frames, fps)
    elif suffix == ".txt":
        labels, meta = _load_txt_labels(label_path, num_frames)
    elif suffix == ".npy":
        labels = np.load(label_path)
        meta = {"loaded_from": "numpy"}
    else:
        raise ValueError(f"Unsupported label file format: {suffix}")

    metadata.update(meta)

    # Validate and adjust length if needed
    if num_frames is not None and len(labels) != num_frames:
        if len(labels) > num_frames:
            labels = labels[:num_frames]
            logger.warning(f"Truncated labels from {len(labels)} to {num_frames} frames")
        else:
            # Pad with last label
            pad_length = num_frames - len(labels)
            last_label = labels[-1] if len(labels) > 0 else 0
            labels = np.concatenate([labels, np.full(pad_length, last_label)])
            logger.warning(f"Padded labels from {len(labels) - pad_length} to {num_frames} frames")

    metadata["num_frames"] = len(labels)
    metadata["class_distribution"] = {
        ACTION_ID_TO_NAME.get(i, f"class_{i}"): int((labels == i).sum())
        for i in np.unique(labels)
    }

    return labels.astype(int), metadata


def _load_csv_labels(
    path: Path,
    num_frames: Optional[int],
    fps: float,
) -> Tuple[np.ndarray, Dict]:
    """Load labels from CSV file."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV file: {path}")

    # Detect format based on header or first row
    num_cols = len(rows[0])

    if num_cols == 1:
        # Per-frame labels: label
        labels = []
        for row in rows:
            label_str = row[0].strip().lower()
            label_id = ACTION_NAME_TO_ID.get(label_str, int(label_str) if label_str.isdigit() else 0)
            labels.append(label_id)
        return np.array(labels), {"csv_format": "per_frame_single"}

    elif num_cols == 2:
        # Format: frame,label
        frame_labels = {}
        for row in rows:
            frame_idx = int(row[0])
            label_str = row[1].strip().lower()
            label_id = ACTION_NAME_TO_ID.get(label_str, int(label_str) if label_str.isdigit() else 0)
            frame_labels[frame_idx] = label_id

        max_frame = max(frame_labels.keys())
        if num_frames:
            max_frame = max(max_frame, num_frames - 1)

        labels = np.zeros(max_frame + 1, dtype=int)
        for frame_idx, label_id in frame_labels.items():
            labels[frame_idx] = label_id

        return labels, {"csv_format": "frame_label"}

    elif num_cols >= 3:
        # Format: start,end,label or start_time,end_time,label
        segments = []
        is_time_based = False

        for row in rows:
            start = float(row[0])
            end = float(row[1])
            label_str = row[2].strip().lower()
            label_id = ACTION_NAME_TO_ID.get(label_str, int(label_str) if label_str.isdigit() else 0)

            # Check if time-based (values less than typical frame counts)
            if start < 100 and end < 1000 and "." in row[0]:
                is_time_based = True

            segments.append((start, end, label_id))

        # Convert to frame labels
        if is_time_based:
            max_time = max(s[1] for s in segments)
            total_frames = int(max_time * fps) + 1
            if num_frames:
                total_frames = max(total_frames, num_frames)

            labels = np.zeros(total_frames, dtype=int)
            for start_time, end_time, label_id in segments:
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                labels[start_frame:end_frame + 1] = label_id
        else:
            max_frame = max(int(s[1]) for s in segments)
            if num_frames:
                max_frame = max(max_frame, num_frames - 1)

            labels = np.zeros(max_frame + 1, dtype=int)
            for start_frame, end_frame, label_id in segments:
                labels[int(start_frame):int(end_frame) + 1] = label_id

        return labels, {"csv_format": "segment", "time_based": is_time_based}

    raise ValueError(f"Unexpected CSV format with {num_cols} columns")


def _load_json_labels(
    path: Path,
    num_frames: Optional[int],
    fps: float,
) -> Tuple[np.ndarray, Dict]:
    """Load labels from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Format 1: Direct array of labels
    if isinstance(data, list):
        labels = []
        for item in data:
            if isinstance(item, int):
                labels.append(item)
            elif isinstance(item, str):
                labels.append(ACTION_NAME_TO_ID.get(item.lower(), 0))
            elif isinstance(item, dict):
                label = item.get("label", item.get("action", 0))
                if isinstance(label, str):
                    label = ACTION_NAME_TO_ID.get(label.lower(), 0)
                labels.append(label)
        return np.array(labels), {"json_format": "array"}

    # Format 2: Dict with labels key
    if "labels" in data:
        labels_data = data["labels"]
        if isinstance(labels_data, list):
            labels = []
            for item in labels_data:
                if isinstance(item, int):
                    labels.append(item)
                elif isinstance(item, str):
                    labels.append(ACTION_NAME_TO_ID.get(item.lower(), 0))
            return np.array(labels), {"json_format": "labels_key"}

    # Format 3: Annotations with segments
    if "annotations" in data:
        annotations = data["annotations"]
        total_frames = num_frames or data.get("num_frames", 1000)
        labels = np.zeros(total_frames, dtype=int)

        for ann in annotations:
            start = ann.get("start", ann.get("start_frame", 0))
            end = ann.get("end", ann.get("end_frame", start + 1))
            label = ann.get("label", ann.get("action", 0))

            if isinstance(label, str):
                label = ACTION_NAME_TO_ID.get(label.lower(), 0)

            # Check if time-based
            if "start_time" in ann or start < 100:
                start = int(start * fps)
                end = int(end * fps)

            labels[int(start):int(end) + 1] = label

        return labels, {"json_format": "annotations"}

    raise ValueError(f"Unrecognized JSON format in: {path}")


def _load_txt_labels(
    path: Path,
    num_frames: Optional[int],
) -> Tuple[np.ndarray, Dict]:
    """Load labels from text file (space/comma/newline separated)."""
    with open(path, "r") as f:
        content = f.read()

    # Try different separators
    for sep in ["\n", ",", " ", "\t"]:
        parts = [p.strip() for p in content.split(sep) if p.strip()]
        if len(parts) > 1:
            break

    labels = []
    for part in parts:
        part_lower = part.lower()
        if part_lower in ACTION_NAME_TO_ID:
            labels.append(ACTION_NAME_TO_ID[part_lower])
        elif part.isdigit():
            labels.append(int(part))

    return np.array(labels), {"txt_format": "separated"}


# ============================================================
# Per-Keypoint-Preset Metrics
# ============================================================

@dataclass
class KeypointPresetMetrics:
    """Metrics for a single keypoint preset evaluation."""
    preset_name: str
    num_keypoints: int
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    class_names: List[str]
    # Additional metadata
    num_frames: int = 0
    class_support: Dict[str, int] = field(default_factory=dict)  # samples per class


@dataclass
class KeypointComparisonReport:
    """Complete report comparing metrics across keypoint presets."""
    presets: List[KeypointPresetMetrics]
    ground_truth_source: str
    num_total_frames: int
    class_names: List[str]
    # Summary statistics
    best_preset_by_accuracy: str = ""
    best_preset_by_f1: str = ""
    accuracy_by_keypoint_count: Dict[int, float] = field(default_factory=dict)
    f1_by_keypoint_count: Dict[int, Dict[str, float]] = field(default_factory=dict)


def compute_preset_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    preset_name: str,
    num_keypoints: int,
    class_names: List[str] = None,
) -> KeypointPresetMetrics:
    """
    Compute comprehensive metrics for a single keypoint preset.

    Args:
        predictions: Predicted action labels
        ground_truth: Ground truth action labels
        preset_name: Name of the keypoint preset
        num_keypoints: Number of keypoints in this preset
        class_names: Names of action classes

    Returns:
        KeypointPresetMetrics object
    """
    if class_names is None:
        class_names = ["stationary", "walking", "running"]

    num_classes = len(class_names)

    # Ensure same length
    min_len = min(len(predictions), len(ground_truth))
    predictions = predictions[:min_len]
    ground_truth = ground_truth[:min_len]

    # Overall accuracy
    accuracy = np.mean(predictions == ground_truth)

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, gt in zip(predictions, ground_truth):
        if 0 <= pred < num_classes and 0 <= gt < num_classes:
            conf_matrix[int(gt), int(pred)] += 1

    # Per-class metrics
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_accuracy = {}
    class_support = {}

    for i, name in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        tn = conf_matrix.sum() - tp - fp - fn

        # Support (number of true samples)
        support = conf_matrix[i, :].sum()
        class_support[name] = int(support)

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class_precision[name] = precision

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_recall[name] = recall

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[name] = f1

        # Per-class accuracy
        class_acc = tp / support if support > 0 else 0.0
        per_class_accuracy[name] = class_acc

    # Macro F1
    macro_f1 = np.mean(list(per_class_f1.values()))

    return KeypointPresetMetrics(
        preset_name=preset_name,
        num_keypoints=num_keypoints,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=conf_matrix,
        class_names=class_names,
        num_frames=min_len,
        class_support=class_support,
    )


def compute_all_preset_metrics(
    preset_predictions: Dict[str, Tuple[np.ndarray, int]],
    ground_truth: np.ndarray,
    class_names: List[str] = None,
    ground_truth_source: str = "unknown",
) -> KeypointComparisonReport:
    """
    Compute metrics for all keypoint presets and generate comparison report.

    Args:
        preset_predictions: Dict mapping preset_name -> (predictions, num_keypoints)
        ground_truth: Ground truth action labels
        class_names: Names of action classes
        ground_truth_source: Source of ground truth labels

    Returns:
        KeypointComparisonReport object
    """
    if class_names is None:
        class_names = ["stationary", "walking", "running"]

    presets = []
    accuracy_by_kp = {}
    f1_by_kp = {}

    for preset_name, (predictions, num_keypoints) in preset_predictions.items():
        metrics = compute_preset_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            preset_name=preset_name,
            num_keypoints=num_keypoints,
            class_names=class_names,
        )
        presets.append(metrics)

        accuracy_by_kp[num_keypoints] = metrics.accuracy
        f1_by_kp[num_keypoints] = metrics.per_class_f1

    # Find best presets
    best_by_accuracy = max(presets, key=lambda x: x.accuracy).preset_name
    best_by_f1 = max(presets, key=lambda x: x.macro_f1).preset_name

    return KeypointComparisonReport(
        presets=presets,
        ground_truth_source=ground_truth_source,
        num_total_frames=len(ground_truth),
        class_names=class_names,
        best_preset_by_accuracy=best_by_accuracy,
        best_preset_by_f1=best_by_f1,
        accuracy_by_keypoint_count=accuracy_by_kp,
        f1_by_keypoint_count=f1_by_kp,
    )


def save_metrics_report(
    report: KeypointComparisonReport,
    output_path: Union[str, Path],
    include_confusion_matrices: bool = True,
) -> Path:
    """
    Save metrics report to JSON file.

    Args:
        report: KeypointComparisonReport object
        output_path: Path to save JSON report
        include_confusion_matrices: Whether to include full confusion matrices

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_dict = {
        "summary": {
            "ground_truth_source": report.ground_truth_source,
            "num_total_frames": report.num_total_frames,
            "class_names": report.class_names,
            "best_preset_by_accuracy": report.best_preset_by_accuracy,
            "best_preset_by_f1": report.best_preset_by_f1,
        },
        "accuracy_by_keypoint_count": {
            str(k): v for k, v in report.accuracy_by_keypoint_count.items()
        },
        "f1_by_keypoint_count": {
            str(k): v for k, v in report.f1_by_keypoint_count.items()
        },
        "presets": [],
    }

    for preset in report.presets:
        preset_dict = {
            "preset_name": preset.preset_name,
            "num_keypoints": preset.num_keypoints,
            "accuracy": preset.accuracy,
            "macro_f1": preset.macro_f1,
            "per_class_f1": preset.per_class_f1,
            "per_class_precision": preset.per_class_precision,
            "per_class_recall": preset.per_class_recall,
            "per_class_accuracy": preset.per_class_accuracy,
            "num_frames": preset.num_frames,
            "class_support": preset.class_support,
        }

        if include_confusion_matrices:
            preset_dict["confusion_matrix"] = preset.confusion_matrix.tolist()

        report_dict["presets"].append(preset_dict)

    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Saved metrics report: {output_path}")
    return output_path


def print_metrics_report(report: KeypointComparisonReport):
    """Print metrics report to console."""
    print("\n" + "=" * 80)
    print("KEYPOINT PRESET EVALUATION REPORT")
    print("=" * 80)
    print(f"Ground Truth Source: {report.ground_truth_source}")
    print(f"Total Frames: {report.num_total_frames}")
    print(f"Classes: {', '.join(report.class_names)}")
    print()

    # Header
    print(f"{'Preset':<12} {'KP#':>4} {'Acc%':>7} {'F1':>7} | ", end="")
    for cls in report.class_names:
        print(f"F1({cls[:4]})", end=" ")
    print()
    print("-" * 80)

    # Sort by keypoint count (descending)
    sorted_presets = sorted(report.presets, key=lambda x: x.num_keypoints, reverse=True)

    for preset in sorted_presets:
        print(f"{preset.preset_name:<12} {preset.num_keypoints:>4} "
              f"{preset.accuracy*100:>6.1f}% {preset.macro_f1:>6.3f} | ", end="")
        for cls in report.class_names:
            f1 = preset.per_class_f1.get(cls, 0)
            print(f"{f1:>6.3f} ", end="")

        # Mark best
        markers = []
        if preset.preset_name == report.best_preset_by_accuracy:
            markers.append("*Acc")
        if preset.preset_name == report.best_preset_by_f1:
            markers.append("*F1")
        if markers:
            print(f" [{', '.join(markers)}]", end="")
        print()

    print("-" * 80)
    print(f"Best by Accuracy: {report.best_preset_by_accuracy}")
    print(f"Best by F1: {report.best_preset_by_f1}")
    print("=" * 80)
