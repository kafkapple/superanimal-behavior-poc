"""
Comprehensive Evaluation Pipeline.

Evaluates action recognition models with:
1. Proper train/val/test splits
2. Multiple action classifiers (Rule-based, MLP, LSTM, Transformer)
3. Multiple keypoint presets/models
4. Real ground truth labels from MARS/CalMS21 datasets
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data.datasets import (
    MARSDatasetLoader,
    CalMS21DatasetLoader,
    CustomDatasetLoader,
    DatasetSplit,
    BehaviorSequence,
    download_mars_sample,
    download_locomotion_sample,
    get_dataset_loader,
    get_dataset_info,
    DATASET_TYPES,
)
from src.models.action_models import (
    BaseActionClassifier,
    RuleBasedClassifier,
    MLPClassifier,
    LSTMClassifier,
    TransformerClassifier,
    ModelMetrics,
    get_action_classifier,
    compare_models,
    TORCH_AVAILABLE,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation."""
    dataset_name: str = "locomotion_sample"  # locomotion_sample, mars_sample, mars, calms21
    dataset_path: Optional[Path] = None
    action_models: List[str] = field(default_factory=lambda: ["rule_based", "lstm", "transformer"])
    keypoint_presets: List[str] = field(default_factory=lambda: ["full", "minimal"])
    num_classes: int = None  # Auto-detect from dataset
    class_names: List[str] = None  # Auto-detect from dataset
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    output_dir: Path = Path("outputs/evaluation")

    def __post_init__(self):
        """Auto-detect class info from dataset type."""
        if self.num_classes is None or self.class_names is None:
            dataset_info = get_dataset_info(self.dataset_name)
            if self.num_classes is None:
                self.num_classes = dataset_info["num_classes"]
            if self.class_names is None:
                self.class_names = dataset_info["class_names"]


@dataclass
class EvaluationResult:
    """Results from comprehensive evaluation."""
    config: EvaluationConfig
    dataset_info: Dict
    model_results: Dict[str, ModelMetrics]
    best_model: str
    best_accuracy: float
    best_f1: float
    comparison_table: List[Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": {
                "dataset_name": self.config.dataset_name,
                "action_models": self.config.action_models,
                "keypoint_presets": self.config.keypoint_presets,
                "num_classes": self.config.num_classes,
                "class_names": self.config.class_names,
            },
            "dataset_info": self.dataset_info,
            "model_results": {
                name: {
                    "accuracy": m.accuracy,
                    "f1_macro": m.f1_macro,
                    "f1_per_class": m.f1_per_class,
                    "confusion_matrix": m.confusion_matrix.tolist(),
                }
                for name, m in self.model_results.items()
            },
            "best_model": self.best_model,
            "best_accuracy": self.best_accuracy,
            "best_f1": self.best_f1,
            "comparison_table": self.comparison_table,
        }


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for action recognition.

    Runs multiple models on train/val/test splits and compares results.
    """

    def __init__(self, config: EvaluationConfig = None):
        """Initialize evaluator with config."""
        self.config = config or EvaluationConfig()
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_split: Optional[DatasetSplit] = None
        self.results: Dict[str, ModelMetrics] = {}

    def load_dataset(self) -> DatasetSplit:
        """Load dataset with train/val/test splits."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        logger.info(f"  Label type: {get_dataset_info(self.config.dataset_name)['label_type']}")
        logger.info(f"  Classes: {self.config.class_names}")

        if self.config.dataset_name == "locomotion_sample":
            # Create locomotion-based sample (stationary/walking/running)
            sample_dir = self.config.output_dir / "datasets" / "locomotion_sample"
            if not (sample_dir / "train").exists():
                download_locomotion_sample(sample_dir)

            loader = MARSDatasetLoader(sample_dir)  # Same loader, different labels
            self.dataset_split = loader.load()

        elif self.config.dataset_name == "mars_sample":
            # Create synthetic MARS sample (social behaviors)
            sample_dir = self.config.output_dir / "datasets" / "mars_sample"
            if not (sample_dir / "train").exists():
                download_mars_sample(sample_dir)

            loader = MARSDatasetLoader(sample_dir)
            self.dataset_split = loader.load()

        elif self.config.dataset_name == "mars":
            if self.config.dataset_path is None:
                raise ValueError("dataset_path required for MARS dataset")
            loader = MARSDatasetLoader(self.config.dataset_path)
            self.dataset_split = loader.load()

        elif self.config.dataset_name == "calms21":
            if self.config.dataset_path is None:
                raise ValueError("dataset_path required for CalMS21 dataset")
            loader = CalMS21DatasetLoader(self.config.dataset_path)
            self.dataset_split = loader.load()

        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

        # Log dataset info
        logger.info(f"Dataset loaded:")
        logger.info(f"  Train sequences: {len(self.dataset_split.train)}")
        logger.info(f"  Val sequences: {len(self.dataset_split.val)}")
        logger.info(f"  Test sequences: {len(self.dataset_split.test)}")

        train_dist = self.dataset_split.get_class_distribution("train")
        logger.info(f"  Train class distribution: {train_dist}")

        return self.dataset_split

    def prepare_features(
        self,
        sequences: List[BehaviorSequence],
        keypoint_preset: str = "full"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from sequences.

        Args:
            sequences: List of BehaviorSequence objects
            keypoint_preset: Which keypoint subset to use

        Returns:
            (features, labels) arrays
        """
        all_features = []
        all_labels = []

        for seq in sequences:
            # Extract keypoints based on preset
            keypoints = seq.keypoints  # (frames, num_kp, 3)

            # Select keypoint subset
            if keypoint_preset == "minimal":
                # Use only 3 keypoints: nose, center, tail
                if keypoints.shape[1] >= 7:
                    indices = [0, 3, 6]  # nose, neck, tail_base
                    keypoints = keypoints[:, indices, :]
            elif keypoint_preset == "locomotion":
                # Use body and limb keypoints
                if keypoints.shape[1] >= 7:
                    indices = [3, 4, 5, 6]  # neck, left_hip, right_hip, tail_base
                    keypoints = keypoints[:, indices, :]

            # Flatten keypoints to features: (frames, num_kp * 2)
            # Only use x, y coordinates for features
            features = keypoints[:, :, :2].reshape(len(keypoints), -1)

            all_features.append(features)
            all_labels.append(seq.labels)

        # Concatenate all sequences
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)

        return X, y

    def run_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelMetrics:
        """
        Train and evaluate a single model.

        Args:
            model_name: Name of the model type
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data

        Returns:
            ModelMetrics with evaluation results
        """
        input_dim = X_train.shape[1]

        # Create model with appropriate parameters
        model_kwargs = {
            "num_classes": self.config.num_classes,
            "class_names": self.config.class_names,
        }

        if model_name in ["mlp", "lstm", "transformer"]:
            model_kwargs.update({
                "input_dim": input_dim,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            })

        try:
            model = get_action_classifier(model_name, **model_kwargs)
        except ImportError as e:
            logger.warning(f"Cannot create {model_name}: {e}")
            # Return empty metrics
            return ModelMetrics(
                accuracy=0.0,
                f1_macro=0.0,
                f1_per_class={c: 0.0 for c in self.config.class_names},
                confusion_matrix=np.zeros((self.config.num_classes, self.config.num_classes)),
                model_name=model_name,
            )

        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")

        # Reshape for sequence models
        if model_name in ["lstm", "transformer"]:
            # Reshape to (num_samples, seq_len, features)
            seq_len = 64
            X_train_seq = self._reshape_to_sequences(X_train, y_train, seq_len)
            y_train_seq = self._reshape_labels(y_train, seq_len)
            X_val_seq = self._reshape_to_sequences(X_val, y_val, seq_len)
            y_val_seq = self._reshape_labels(y_val, seq_len)
            X_test_seq = self._reshape_to_sequences(X_test, y_test, seq_len)
            y_test_seq = self._reshape_labels(y_test, seq_len)

            # Train
            model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)

            # Evaluate
            metrics = model.evaluate(X_test_seq, y_test_seq)
        else:
            # MLP and rule-based use windowed or frame-by-frame
            X_train_2d = X_train.reshape(-1, 1, input_dim)
            y_train_2d = y_train.reshape(-1, 1)
            X_val_2d = X_val.reshape(-1, 1, input_dim) if X_val is not None else None
            y_val_2d = y_val.reshape(-1, 1) if y_val is not None else None
            X_test_2d = X_test.reshape(-1, 1, input_dim)
            y_test_2d = y_test.reshape(-1, 1)

            # Train
            model.fit(X_train_2d, y_train_2d, X_val_2d, y_val_2d)

            # Evaluate
            metrics = model.evaluate(X_test_2d, y_test_2d)

        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"  F1 (macro): {metrics.f1_macro:.4f}")
        logger.info(f"  F1 per class: {metrics.f1_per_class}")

        return metrics

    def _reshape_to_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int
    ) -> np.ndarray:
        """Reshape flat data to sequences."""
        num_frames = len(X)
        num_sequences = num_frames // seq_len

        if num_sequences == 0:
            # Pad to at least one sequence
            pad_len = seq_len - num_frames
            X = np.pad(X, ((0, pad_len), (0, 0)), mode='edge')
            return X.reshape(1, seq_len, -1)

        # Trim to fit
        X = X[:num_sequences * seq_len]
        return X.reshape(num_sequences, seq_len, -1)

    def _reshape_labels(self, y: np.ndarray, seq_len: int) -> np.ndarray:
        """Reshape labels to match sequences."""
        num_frames = len(y)
        num_sequences = num_frames // seq_len

        if num_sequences == 0:
            pad_len = seq_len - num_frames
            y = np.pad(y, (0, pad_len), mode='edge')
            return y.reshape(1, seq_len)

        y = y[:num_sequences * seq_len]
        return y.reshape(num_sequences, seq_len)

    def run_evaluation(self) -> EvaluationResult:
        """
        Run comprehensive evaluation across all configurations.

        Returns:
            EvaluationResult with all comparison data
        """
        # Load dataset
        if self.dataset_split is None:
            self.load_dataset()

        all_results = {}
        comparison_table = []

        # Iterate over keypoint presets
        for preset in self.config.keypoint_presets:
            logger.info(f"\n{'#'*60}")
            logger.info(f"Evaluating keypoint preset: {preset}")
            logger.info(f"{'#'*60}")

            # Prepare features for this preset
            X_train, y_train = self.prepare_features(self.dataset_split.train, preset)
            X_val, y_val = self.prepare_features(self.dataset_split.val, preset)
            X_test, y_test = self.prepare_features(self.dataset_split.test, preset)

            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

            # Iterate over action models
            for model_name in self.config.action_models:
                if model_name in ["mlp", "lstm", "transformer"] and not TORCH_AVAILABLE:
                    logger.warning(f"Skipping {model_name} (PyTorch not available)")
                    continue

                full_name = f"{preset}_{model_name}"

                try:
                    metrics = self.run_model(
                        model_name,
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test,
                    )
                    all_results[full_name] = metrics

                    comparison_table.append({
                        "keypoint_preset": preset,
                        "model": model_name,
                        "accuracy": metrics.accuracy,
                        "f1_macro": metrics.f1_macro,
                        "f1_per_class": metrics.f1_per_class,
                    })

                except Exception as e:
                    logger.error(f"Error running {full_name}: {e}")
                    import traceback
                    traceback.print_exc()

        # Find best model
        if all_results:
            best_name = max(all_results.keys(), key=lambda k: all_results[k].f1_macro)
            best_metrics = all_results[best_name]
        else:
            best_name = "none"
            best_metrics = ModelMetrics(
                accuracy=0, f1_macro=0,
                f1_per_class={}, confusion_matrix=np.zeros((1, 1)),
                model_name="none"
            )

        # Dataset info
        dataset_info = {
            "name": self.config.dataset_name,
            "num_classes": self.config.num_classes,
            "class_names": self.config.class_names,
            "train_samples": sum(len(s.labels) for s in self.dataset_split.train),
            "val_samples": sum(len(s.labels) for s in self.dataset_split.val),
            "test_samples": sum(len(s.labels) for s in self.dataset_split.test),
            "train_distribution": self.dataset_split.get_class_distribution("train"),
            "test_distribution": self.dataset_split.get_class_distribution("test"),
        }

        result = EvaluationResult(
            config=self.config,
            dataset_info=dataset_info,
            model_results=all_results,
            best_model=best_name,
            best_accuracy=best_metrics.accuracy,
            best_f1=best_metrics.f1_macro,
            comparison_table=comparison_table,
        )

        # Save results
        self.save_results(result)

        return result

    def save_results(self, result: EvaluationResult):
        """Save evaluation results to JSON."""
        output_path = self.config.output_dir / "evaluation_results.json"

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved evaluation results: {output_path}")

        # Also print comparison table
        self.print_comparison_table(result)

    def print_comparison_table(self, result: EvaluationResult):
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nDataset: {result.dataset_info['name']}")
        print(f"Classes: {result.config.class_names}")
        print(f"Train samples: {result.dataset_info['train_samples']}")
        print(f"Test samples: {result.dataset_info['test_samples']}")

        print("\n" + "-" * 80)
        print(f"{'Keypoint Preset':<20} {'Model':<15} {'Accuracy':<12} {'F1 (Macro)':<12}")
        print("-" * 80)

        for row in result.comparison_table:
            print(f"{row['keypoint_preset']:<20} {row['model']:<15} "
                  f"{row['accuracy']:.4f}       {row['f1_macro']:.4f}")

        print("-" * 80)
        print(f"\nBest Model: {result.best_model}")
        print(f"Best Accuracy: {result.best_accuracy:.4f}")
        print(f"Best F1 (Macro): {result.best_f1:.4f}")

        # Per-class F1 for best model
        if result.best_model in result.model_results:
            best = result.model_results[result.best_model]
            print(f"\nPer-class F1 scores:")
            for cls, f1 in best.f1_per_class.items():
                print(f"  {cls}: {f1:.4f}")

        print("=" * 80)


def run_quick_evaluation(
    output_dir: Path = Path("outputs/evaluation"),
    epochs: int = 10,
) -> EvaluationResult:
    """
    Run a quick evaluation with synthetic data for demonstration.

    Args:
        output_dir: Output directory
        epochs: Number of training epochs

    Returns:
        EvaluationResult
    """
    config = EvaluationConfig(
        dataset_name="mars_sample",
        action_models=["rule_based", "mlp"],
        keypoint_presets=["full", "minimal"],
        epochs=epochs,
        output_dir=output_dir,
    )

    evaluator = ComprehensiveEvaluator(config)
    return evaluator.run_evaluation()


def run_full_evaluation(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path = Path("outputs/evaluation"),
    epochs: int = 50,
) -> EvaluationResult:
    """
    Run full evaluation on a real dataset.

    Args:
        dataset_name: "mars" or "calms21"
        dataset_path: Path to dataset
        output_dir: Output directory
        epochs: Number of training epochs

    Returns:
        EvaluationResult
    """
    config = EvaluationConfig(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        action_models=["rule_based", "mlp", "lstm", "transformer"],
        keypoint_presets=["full", "minimal", "locomotion"],
        epochs=epochs,
        output_dir=output_dir,
    )

    evaluator = ComprehensiveEvaluator(config)
    return evaluator.run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run quick evaluation
    result = run_quick_evaluation(epochs=5)
    print(f"\nEvaluation complete. Best model: {result.best_model}")
