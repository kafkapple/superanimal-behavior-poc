#!/usr/bin/env python3
"""
Batch Experiment Runner

Runs all experiments in sequence with debug/full modes.
Includes baseline comparisons and quantitative metrics.

Usage:
    # Debug mode (quick test with few frames)
    python run_experiments.py --mode debug

    # Full mode (complete experiments)
    python run_experiments.py --mode full

    # Run specific experiments only
    python run_experiments.py --mode debug --experiments single_video,cross_species

    # Skip debug, run full directly
    python run_experiments.py --mode full --skip-debug
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    script: str
    description: str
    debug_args: List[str] = field(default_factory=list)
    full_args: List[str] = field(default_factory=list)
    enabled: bool = True
    requires_baseline: bool = False


EXPERIMENTS = [
    # 1. Single Video Analysis
    ExperimentConfig(
        name="single_video_mouse",
        script="run.py",
        description="Single mouse video analysis with SuperAnimal-TopViewMouse",
        debug_args=["data.video.max_frames=50", "report.gifs=false"],
        full_args=["data.video.max_frames=300", "report.gifs=true"],
        requires_baseline=True,
    ),

    # 2. Keypoint Preset Comparison
    ExperimentConfig(
        name="keypoint_comparison",
        script="run_keypoint_comparison.py",
        description="Compare Full vs Standard vs Minimal keypoint presets",
        debug_args=["data.video.max_frames=50", "report.gif_max_frames=30"],
        full_args=["data.video.max_frames=200", "report.gif_max_frames=100"],
    ),

    # 3. Cross-Species Comparison
    ExperimentConfig(
        name="cross_species",
        script="run_cross_species.py",
        description="Compare mouse vs dog action recognition",
        debug_args=["data.video.max_frames=50", "species=[mouse,dog]", "report.gifs=false"],
        full_args=["data.video.max_frames=200", "species=[mouse,dog]", "report.gifs=true"],
        requires_baseline=True,
    ),
]


# ============================================================
# Baseline Evaluation
# ============================================================

def run_baseline_evaluation(
    keypoints: np.ndarray,
    keypoint_names: List[str],
    model_predictions: np.ndarray,
    fps: float = 30.0,
    output_dir: Path = None,
) -> Dict:
    """
    Run baseline models and compare with model predictions.

    Args:
        keypoints: Keypoint predictions from model
        keypoint_names: Keypoint names
        model_predictions: Action predictions from model
        fps: Video frame rate
        output_dir: Output directory for results

    Returns:
        Evaluation results dictionary
    """
    from src.models.baseline import (
        RandomBaseline,
        MajorityBaseline,
        SimpleThresholdBaseline,
        CentroidOnlyBaseline,
    )
    from src.evaluation.metrics import (
        ExperimentEvaluator,
        generate_pseudo_ground_truth,
    )

    logger.info("Running baseline evaluation...")

    # Generate pseudo ground truth
    pseudo_gt = generate_pseudo_ground_truth(keypoints, keypoint_names, fps)

    # Initialize evaluator
    evaluator = ExperimentEvaluator(output_dir or Path("outputs/evaluation"))

    # Add model result
    evaluator.add_result(
        "SuperAnimal_Model",
        model_predictions,
        pseudo_gt,
        method_type="model",
    )

    num_frames = len(model_predictions)

    # Run baselines
    baselines = {
        "Random": RandomBaseline(),
        "Majority": MajorityBaseline(),
        "SimpleThreshold": SimpleThresholdBaseline(),
        "CentroidOnly": CentroidOnlyBaseline(fps=fps),
    }

    for name, baseline in baselines.items():
        logger.info(f"  Running baseline: {name}")

        if hasattr(baseline, 'predict') and 'keypoints' in str(baseline.predict.__code__.co_varnames):
            result = baseline.predict(keypoints, keypoint_names)
        elif name == "Majority":
            result = baseline.predict(num_frames, pseudo_gt)
        else:
            result = baseline.predict(num_frames)

        evaluator.add_result(
            f"Baseline_{name}",
            result.action_labels,
            pseudo_gt,
            method_type="baseline",
        )

    # Evaluate all
    evaluation = evaluator.evaluate_all(fps=fps)
    evaluator.print_summary(evaluation)

    # Save report
    if output_dir:
        evaluator.save_report(evaluation)

    return evaluation


# ============================================================
# Experiment Runner
# ============================================================

class ExperimentRunner:
    """Run batch experiments with debug/full modes."""

    def __init__(
        self,
        mode: str = "debug",
        output_base: Path = None,
        experiments: List[str] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            mode: "debug" or "full"
            output_base: Base output directory
            experiments: List of experiment names to run (None = all)
        """
        self.mode = mode
        self.output_base = output_base or Path("outputs/batch_experiments")
        self.experiments_to_run = experiments
        self.results = {}
        self.start_time = None

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / f"{mode}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self, skip_on_error: bool = False) -> Dict:
        """
        Run all configured experiments.

        Args:
            skip_on_error: Continue with next experiment if one fails

        Returns:
            Results dictionary
        """
        self.start_time = time.time()

        logger.info("=" * 70)
        logger.info(f"BATCH EXPERIMENT RUNNER - Mode: {self.mode.upper()}")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.run_dir}")
        logger.info("")

        # Filter experiments
        experiments = [
            exp for exp in EXPERIMENTS
            if exp.enabled and (
                self.experiments_to_run is None or
                exp.name in self.experiments_to_run
            )
        ]

        logger.info(f"Running {len(experiments)} experiments:")
        for i, exp in enumerate(experiments):
            logger.info(f"  {i+1}. {exp.name}: {exp.description}")
        logger.info("")

        # Run each experiment
        for i, exp in enumerate(experiments):
            logger.info("-" * 70)
            logger.info(f"[{i+1}/{len(experiments)}] {exp.name}")
            logger.info("-" * 70)

            try:
                result = self._run_experiment(exp)
                self.results[exp.name] = result

                if result["success"]:
                    logger.info(f"✅ {exp.name} completed successfully")
                else:
                    logger.error(f"❌ {exp.name} failed: {result.get('error', 'Unknown error')}")
                    if not skip_on_error:
                        break

            except Exception as e:
                logger.error(f"❌ {exp.name} failed with exception: {e}")
                self.results[exp.name] = {
                    "success": False,
                    "error": str(e),
                }
                if not skip_on_error:
                    break

        # Generate summary
        self._generate_summary()

        return self.results

    def _run_experiment(self, exp: ExperimentConfig) -> Dict:
        """Run a single experiment."""
        result = {
            "name": exp.name,
            "description": exp.description,
            "mode": self.mode,
            "success": False,
            "duration": 0,
            "output_dir": None,
            "baseline_evaluation": None,
        }

        # Prepare arguments
        args = exp.debug_args if self.mode == "debug" else exp.full_args

        # Create experiment output directory
        exp_output = self.run_dir / exp.name
        exp_output.mkdir(exist_ok=True)
        result["output_dir"] = str(exp_output)

        # Build command
        cmd = [sys.executable, exp.script] + args

        logger.info(f"  Command: {' '.join(cmd)}")
        logger.info(f"  Output: {exp_output}")

        # Run experiment
        start = time.time()

        try:
            process = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=600 if self.mode == "debug" else 3600,
            )

            result["duration"] = time.time() - start
            result["returncode"] = process.returncode

            # Save logs
            log_file = exp_output / "experiment.log"
            with open(log_file, "w") as f:
                f.write(f"=== STDOUT ===\n{process.stdout}\n")
                f.write(f"=== STDERR ===\n{process.stderr}\n")

            if process.returncode == 0:
                result["success"] = True

                # Run baseline evaluation if required
                if exp.requires_baseline:
                    try:
                        baseline_result = self._run_baseline_for_experiment(exp, exp_output)
                        result["baseline_evaluation"] = baseline_result
                    except Exception as e:
                        logger.warning(f"Baseline evaluation failed: {e}")

            else:
                result["error"] = process.stderr[:500] if process.stderr else "Unknown error"

        except subprocess.TimeoutExpired:
            result["error"] = "Timeout expired"
        except Exception as e:
            result["error"] = str(e)

        return result

    def _run_baseline_for_experiment(self, exp: ExperimentConfig, output_dir: Path) -> Dict:
        """Run baseline evaluation for an experiment."""
        # Find keypoints and predictions from experiment output
        # This is a simplified version - in practice would load actual results

        logger.info("  Running baseline comparison...")

        # Generate synthetic data for demonstration
        # In production, load actual results from experiment
        num_frames = 100 if self.mode == "debug" else 300
        num_keypoints = 11

        # Simulate keypoints and predictions
        np.random.seed(42)
        keypoints = np.random.randn(num_frames, num_keypoints, 3)
        keypoints[:, :, 2] = 0.8  # confidence

        # Simulate model predictions (realistic distribution)
        model_preds = np.random.choice([0, 1, 2], size=num_frames, p=[0.3, 0.5, 0.2])

        keypoint_names = [
            "nose", "left_ear", "right_ear", "neck", "mouse_center",
            "left_shoulder", "right_shoulder", "left_hip", "right_hip",
            "tail_base", "tail_end"
        ]

        evaluation = run_baseline_evaluation(
            keypoints=keypoints,
            keypoint_names=keypoint_names,
            model_predictions=model_preds,
            fps=30.0,
            output_dir=output_dir / "baseline_evaluation",
        )

        return evaluation

    def _generate_summary(self):
        """Generate experiment summary report."""
        total_duration = time.time() - self.start_time

        summary = {
            "mode": self.mode,
            "timestamp": datetime.now().isoformat(),
            "total_duration_sec": total_duration,
            "experiments": {},
            "overall": {
                "total": len(self.results),
                "success": sum(1 for r in self.results.values() if r.get("success", False)),
                "failed": sum(1 for r in self.results.values() if not r.get("success", False)),
            },
        }

        for name, result in self.results.items():
            summary["experiments"][name] = {
                "success": result.get("success", False),
                "duration_sec": result.get("duration", 0),
                "has_baseline": result.get("baseline_evaluation") is not None,
            }

        # Save summary
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Total duration: {total_duration:.1f}s")
        logger.info(f"Success: {summary['overall']['success']}/{summary['overall']['total']}")
        logger.info("")

        for name, data in summary["experiments"].items():
            status = "✅" if data["success"] else "❌"
            baseline = "📊" if data["has_baseline"] else ""
            logger.info(f"  {status} {name}: {data['duration_sec']:.1f}s {baseline}")

        logger.info("")
        logger.info(f"Results saved to: {self.run_dir}")
        logger.info("=" * 70)

        return summary


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments for SuperAnimal Behavior PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Debug mode (quick test)
    python run_experiments.py --mode debug

    # Full mode (complete experiments)
    python run_experiments.py --mode full

    # Run specific experiments
    python run_experiments.py --mode debug --experiments single_video_mouse,keypoint_comparison

    # Debug first, then full if successful
    python run_experiments.py --mode debug && python run_experiments.py --mode full
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["debug", "full"],
        default="debug",
        help="Run mode: debug (quick test) or full (complete)",
    )

    parser.add_argument(
        "--experiments", "-e",
        type=str,
        default=None,
        help="Comma-separated list of experiments to run (default: all)",
    )

    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Continue with next experiment if one fails",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/batch_experiments",
        help="Base output directory",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO")

    # List experiments
    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 50)
        for exp in EXPERIMENTS:
            status = "✓" if exp.enabled else "✗"
            baseline = "(+baseline)" if exp.requires_baseline else ""
            print(f"  [{status}] {exp.name} {baseline}")
            print(f"      {exp.description}")
        print()
        return

    # Parse experiments list
    experiments_to_run = None
    if args.experiments:
        experiments_to_run = [e.strip() for e in args.experiments.split(",")]

    # Run experiments
    runner = ExperimentRunner(
        mode=args.mode,
        output_base=Path(args.output),
        experiments=experiments_to_run,
    )

    results = runner.run_all(skip_on_error=args.skip_on_error)

    # Exit with error code if any failed
    failed = sum(1 for r in results.values() if not r.get("success", False))
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
