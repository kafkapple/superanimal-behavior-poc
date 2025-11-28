#!/usr/bin/env python3
"""
Comprehensive Behavior Action Recognition Evaluation.

Compares multiple models:
- Keypoint presets: full, minimal, locomotion
- Action classifiers: Rule-based, MLP, LSTM, Transformer

With proper train/validation/test splits using MARS-style annotations.
"""
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.comprehensive_evaluation import (
    EvaluationConfig,
    ComprehensiveEvaluator,
    run_quick_evaluation,
    run_full_evaluation,
)
from src.data.datasets import download_mars_sample


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive action recognition evaluation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full", "demo"],
        help="Evaluation mode: quick (synthetic), full (real data), demo (minimal)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="locomotion_sample",
        help="Dataset name: locomotion_sample (stationary/walking/running), mars_sample (social behaviors), mars, calms21"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to dataset directory (for real datasets)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["rule_based", "mlp", "lstm", "transformer"],
        help="Action models to evaluate"
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=["full", "minimal"],
        help="Keypoint presets to evaluate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for neural models"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Comprehensive Behavior Recognition Evaluation")
    logger.info("=" * 60)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run based on mode
    if args.mode == "demo":
        # Minimal demo with very few epochs
        logger.info("\nRunning DEMO mode (minimal epochs)...")

        config = EvaluationConfig(
            dataset_name="mars_sample",
            action_models=["rule_based"],  # Only rule-based (no training)
            keypoint_presets=["full"],
            epochs=1,
            output_dir=args.output_dir,
        )

        evaluator = ComprehensiveEvaluator(config)
        result = evaluator.run_evaluation()

    elif args.mode == "quick":
        # Quick evaluation with synthetic or specified dataset
        logger.info(f"\nRunning QUICK mode (dataset: {args.dataset})...")

        config = EvaluationConfig(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            action_models=args.models,
            keypoint_presets=args.presets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        evaluator = ComprehensiveEvaluator(config)
        result = evaluator.run_evaluation()

    elif args.mode == "full":
        # Full evaluation with real data
        logger.info("\nRunning FULL mode (real dataset)...")

        if args.dataset_path is None and args.dataset not in ["mars_sample"]:
            logger.error("--dataset-path required for full evaluation mode")
            sys.exit(1)

        config = EvaluationConfig(
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            action_models=args.models,
            keypoint_presets=args.presets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        evaluator = ComprehensiveEvaluator(config)
        result = evaluator.run_evaluation()

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nBest Model: {result.best_model}")
    print(f"Best Accuracy: {result.best_accuracy:.4f}")
    print(f"Best F1 Score: {result.best_f1:.4f}")

    # Print comparison summary
    print("\n--- Model Comparison ---")
    for row in result.comparison_table:
        print(f"  {row['keypoint_preset']}/{row['model']}: "
              f"Acc={row['accuracy']:.3f}, F1={row['f1_macro']:.3f}")

    return result


if __name__ == "__main__":
    main()
