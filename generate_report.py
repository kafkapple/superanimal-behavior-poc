#!/usr/bin/env python3
"""
Comprehensive Report Generator for SuperAnimal Behavior Analysis.

This script aggregates results from all experiments and generates:
- Interactive HTML dashboard
- Keypoint preset comparison visualizations
- Cross-species comparison visualizations
- Action recognition performance reports
- Animated GIF galleries

Usage:
    python generate_report.py --input outputs/full_pipeline/<timestamp>
    python generate_report.py --input outputs/full_pipeline/<timestamp> --output-dir custom_report
    python generate_report.py --input outputs/full_pipeline/<timestamp> --include-gifs
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logging
from src.analysis.dashboard import (
    DashboardGenerator,
    ExperimentSummary,
    KeypointPresetResult,
    SpeciesResult,
    ActionRecognitionResult,
)
from src.analysis.keypoint_visualizer import (
    KeypointVisualizer,
    KeypointAnalysisResult,
    KEYPOINT_PRESETS,
    PRESET_DESCRIPTIONS,
)
from src.analysis.species_visualizer import (
    SpeciesVisualizer,
    SpeciesAnalysisResult,
    SpeciesComparisonSummary,
)

logger = logging.getLogger(__name__)


# ============================================================
# Result Collection
# ============================================================

def collect_keypoint_results(experiment_dir: Path) -> List[KeypointPresetResult]:
    """
    Collect keypoint comparison results from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of KeypointPresetResult
    """
    results = []

    # First, check for performance_metrics.json with full accuracy/F1 data
    perf_metrics_files = list(experiment_dir.glob("**/performance_metrics.json"))
    if perf_metrics_files:
        for perf_file in perf_metrics_files:
            try:
                with open(perf_file) as f:
                    perf_data = json.load(f)

                # Extract results from performance_metrics.json
                for r in perf_data.get("results", []):
                    preset_kps = KEYPOINT_PRESETS.get(r["preset_name"])
                    num_kps = r.get("num_keypoints", len(preset_kps) if preset_kps else 27)

                    result = KeypointPresetResult(
                        preset_name=r["preset_name"],
                        num_keypoints=num_kps,
                        keypoint_names=preset_kps or [],
                        action_accuracy=r.get("accuracy", 0),
                        action_distribution=r.get("action_distribution", {}),
                        mean_confidence=r.get("mean_confidence", 0.8),
                        trajectory_distance=0,
                        velocity_stats={},
                        f1_scores=r.get("f1_scores", {}),
                        precision=r.get("precision", {}),
                        recall=r.get("recall", {}),
                        agreement_with_full=r.get("agreement_with_full", 0),
                    )
                    results.append(result)

                if results:
                    results.sort(key=lambda x: x.num_keypoints, reverse=True)
                    logger.info(f"Loaded {len(results)} preset results from {perf_file}")
                    return results

            except Exception as e:
                logger.warning(f"Failed to parse {perf_file}: {e}")

    # Fall back to old method: look for keypoint comparison outputs
    kp_dirs = list(experiment_dir.glob("**/keypoint_comparison*")) + \
              list(experiment_dir.glob("**/preset_*"))

    # Also check comprehensive directory
    comp_dir = experiment_dir / "comprehensive"
    if comp_dir.exists():
        kp_dirs.extend(list(comp_dir.glob("preset_*")))

    preset_data = {}

    for kp_dir in kp_dirs:
        # Try to load metrics
        metrics_files = list(kp_dir.glob("**/behavior_metrics.csv")) + \
                       list(kp_dir.glob("**/metrics.json"))

        for mf in metrics_files:
            try:
                if mf.suffix == ".json":
                    with open(mf) as f:
                        data = json.load(f)
                elif mf.suffix == ".csv":
                    import pandas as pd
                    df = pd.read_csv(mf)
                    data = df.to_dict('records')[0] if len(df) > 0 else {}
                else:
                    continue

                # Extract preset name from directory
                preset_name = None
                for preset in KEYPOINT_PRESETS.keys():
                    if preset in str(kp_dir).lower():
                        preset_name = preset
                        break

                if preset_name and preset_name not in preset_data:
                    preset_data[preset_name] = data

            except Exception as e:
                logger.warning(f"Failed to parse {mf}: {e}")

    # Convert to KeypointPresetResult
    for preset_name, data in preset_data.items():
        preset_kps = KEYPOINT_PRESETS.get(preset_name)
        num_kps = len(preset_kps) if preset_kps else 27  # Assume full is 27

        # Extract action distribution
        action_dist = {}
        for action in ["stationary", "walking", "running"]:
            if f"{action}_pct" in data:
                action_dist[action] = data[f"{action}_pct"]
            elif f"{action}_percentage" in data:
                action_dist[action] = data[f"{action}_percentage"]
            elif action in data:
                action_dist[action] = data[action]

        # Extract F1 scores
        f1_scores = {}
        precision = {}
        recall = {}
        for action in ["stationary", "walking", "running"]:
            if f"f1_{action}" in data:
                f1_scores[action] = data[f"f1_{action}"]
            if f"precision_{action}" in data:
                precision[action] = data[f"precision_{action}"]
            if f"recall_{action}" in data:
                recall[action] = data[f"recall_{action}"]

        result = KeypointPresetResult(
            preset_name=preset_name,
            num_keypoints=num_kps,
            keypoint_names=preset_kps or [],
            action_accuracy=data.get("accuracy", data.get("action_accuracy", 0)),
            action_distribution=action_dist,
            mean_confidence=data.get("mean_confidence", 0.8),
            trajectory_distance=data.get("total_distance", 0),
            velocity_stats={
                "mean": data.get("mean_velocity", 0),
                "std": data.get("std_velocity", 0),
                "max": data.get("max_velocity", 0),
            },
            f1_scores=f1_scores,
            precision=precision,
            recall=recall,
            agreement_with_full=data.get("agreement_with_full", data.get("agreement", 0)),
        )
        results.append(result)

    # Sort by keypoint count (descending)
    results.sort(key=lambda x: x.num_keypoints, reverse=True)

    return results


def collect_species_results(experiment_dir: Path) -> List[SpeciesResult]:
    """
    Collect cross-species comparison results.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of SpeciesResult
    """
    results = []

    # Look for cross-species outputs
    species_dirs = list(experiment_dir.glob("**/cross_species*")) + \
                   list(experiment_dir.glob("**/all_species*"))

    species_data = {}

    for sp_dir in species_dirs:
        # Check for comparison CSV
        csv_files = list(sp_dir.glob("**/*comparison*.csv")) + \
                   list(sp_dir.glob("**/*action*.csv"))

        for csv_file in csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)

                for _, row in df.iterrows():
                    species_name = row.get("species", row.get("name", "unknown"))
                    if species_name not in species_data:
                        species_data[species_name] = dict(row)

            except Exception as e:
                logger.warning(f"Failed to parse {csv_file}: {e}")

        # Check for JSON files
        json_files = list(sp_dir.glob("**/*.json"))
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, dict) and "body_size" in val:
                            if key not in species_data:
                                species_data[key] = val

            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")

    # Convert to SpeciesResult
    for species_name, data in species_data.items():
        body_stats = data.get("body_size", {})
        if isinstance(body_stats, (int, float)):
            body_stats = {"mean": body_stats, "std": 0, "min": body_stats, "max": body_stats}

        action_dist = {}
        for action in ["stationary", "walking", "running"]:
            if action in data:
                val = data[action]
                if isinstance(val, dict):
                    action_dist[action] = val.get("percentage", 0)
                else:
                    action_dist[action] = float(val)

        result = SpeciesResult(
            species_name=species_name,
            model_type=data.get("model_type", data.get("model", "unknown")),
            num_frames=int(data.get("total_frames", data.get("frames", 0))),
            body_size_stats=body_stats,
            action_distribution=action_dist,
            velocity_stats={
                "mean": data.get("mean_velocity", 0),
                "std": data.get("std_velocity", 0),
            },
        )
        results.append(result)

    return results


def collect_action_results(experiment_dir: Path) -> List[ActionRecognitionResult]:
    """
    Collect action recognition results.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of ActionRecognitionResult
    """
    results = []

    # Look for model comparison and evaluation outputs
    eval_dirs = list(experiment_dir.glob("**/evaluation*")) + \
                list(experiment_dir.glob("**/model_comparison*"))

    model_data = {}

    for eval_dir in eval_dirs:
        json_files = list(eval_dir.glob("**/*.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle evaluation report format
                if "experiments" in data:
                    for model_name, metrics in data["experiments"].items():
                        if model_name not in model_data:
                            model_data[model_name] = metrics

                # Handle model comparison format
                if "reference" in data:
                    for key, val in data.items():
                        if isinstance(val, dict) and "accuracy" in val:
                            model_name = val.get("model", key)
                            if model_name not in model_data:
                                model_data[model_name] = val

            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")

    # Convert to ActionRecognitionResult
    for model_name, data in model_data.items():
        f1_scores = data.get("f1_score", data.get("f1_scores", {}))
        if isinstance(f1_scores, (int, float)):
            f1_scores = {"overall": f1_scores}

        result = ActionRecognitionResult(
            model_name=model_name,
            accuracy=data.get("accuracy", 0),
            f1_scores=f1_scores,
            confusion_matrix=np.array(data["confusion_matrix"]) if "confusion_matrix" in data else None,
            consistency_score=data.get("smoothness_score", data.get("consistency", 0)),
            per_class_accuracy=data.get("per_class_accuracy", {}),
        )
        results.append(result)

    # Sort by accuracy
    results.sort(key=lambda x: x.accuracy, reverse=True)

    return results


def collect_gif_paths(experiment_dir: Path) -> Dict[str, List[Path]]:
    """
    Collect GIF paths organized by action/category.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dict mapping category to list of GIF paths
    """
    gif_paths = {}

    all_gifs = list(experiment_dir.rglob("*.gif"))

    for gif_path in all_gifs:
        # Categorize by filename
        stem = gif_path.stem.lower()

        # Check for preset×action GIFs (e.g., preset_full_walking_1_video.gif)
        if stem.startswith("preset_") and any(action in stem for action in ["stationary", "walking", "running"]):
            # Parse preset and action from filename
            parts = stem.split("_")
            if len(parts) >= 3:
                preset_name = parts[1]  # e.g., "full", "standard"
                action_name = parts[2]  # e.g., "stationary", "walking", "running"
                category = f"preset_{preset_name}_{action_name}"
        elif "stationary" in stem:
            category = "stationary"
        elif "walking" in stem:
            category = "walking"
        elif "running" in stem:
            category = "running"
        elif "comparison" in stem:
            category = "comparison"
        elif "keypoint" in stem or "preset" in stem:
            category = "keypoint_comparison"
        elif "species" in stem:
            category = "species_comparison"
        else:
            category = "other"

        if category not in gif_paths:
            gif_paths[category] = []
        gif_paths[category].append(gif_path)

    return gif_paths


def collect_plot_paths(experiment_dir: Path) -> Dict[str, Path]:
    """
    Collect plot image paths.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dict mapping plot name to path
    """
    plot_paths = {}

    # Look for PNG files
    png_files = list(experiment_dir.rglob("*.png"))

    for png_path in png_files:
        stem = png_path.stem.lower()

        # Skip small/thumbnail images
        if "thumb" in stem or "_small" in stem:
            continue

        # Key important plots
        if "trajectory" in stem:
            plot_paths["trajectory"] = png_path
        elif "velocity" in stem:
            plot_paths["velocity_profile"] = png_path
        elif "behavior" in stem or "timeline" in stem:
            plot_paths["behavior_timeline"] = png_path
        elif "comparison" in stem and "species" in stem:
            plot_paths["cross_species_comparison"] = png_path
        elif "comparison" in stem and ("keypoint" in stem or "preset" in stem):
            plot_paths["keypoint_comparison"] = png_path
        elif "body" in stem and "size" in stem:
            plot_paths["body_size_comparison"] = png_path
        elif "action" in stem and "distribution" in stem:
            plot_paths["action_distribution"] = png_path
        elif "comprehensive" in stem:
            plot_paths["comprehensive_analysis"] = png_path
        elif "report" in stem:
            plot_paths["analysis_report"] = png_path
        # New hierarchical action comparison plots
        elif "hierarchical" in stem and "action" in stem:
            plot_paths["hierarchical_action_comparison"] = png_path
        elif "confusion" in stem and "matrix" in stem:
            plot_paths["confusion_matrix_grid"] = png_path
        elif "performance" in stem and "keypoint" in stem:
            plot_paths["performance_by_keypoint"] = png_path

    return plot_paths


# ============================================================
# Report Generation
# ============================================================

def generate_comprehensive_report(
    experiment_dir: Path,
    output_dir: Path,
    experiment_name: str = "SuperAnimal Behavior Analysis",
    include_gifs: bool = True,
) -> Path:
    """
    Generate comprehensive HTML report from experiment results.

    Args:
        experiment_dir: Path to experiment results directory
        output_dir: Path to output directory for report
        experiment_name: Name for the experiment
        include_gifs: Whether to include GIFs in report

    Returns:
        Path to generated report
    """
    logger.info(f"Generating report from: {experiment_dir}")
    logger.info(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    logger.info("Collecting keypoint results...")
    keypoint_results = collect_keypoint_results(experiment_dir)
    logger.info(f"  Found {len(keypoint_results)} keypoint preset results")

    logger.info("Collecting species results...")
    species_results = collect_species_results(experiment_dir)
    logger.info(f"  Found {len(species_results)} species results")

    logger.info("Collecting action recognition results...")
    action_results = collect_action_results(experiment_dir)
    logger.info(f"  Found {len(action_results)} action recognition results")

    logger.info("Collecting visualization files...")
    gif_paths = collect_gif_paths(experiment_dir) if include_gifs else {}
    plot_paths = collect_plot_paths(experiment_dir)
    logger.info(f"  Found {sum(len(v) for v in gif_paths.values())} GIFs, {len(plot_paths)} plots")

    # Calculate total frames
    total_frames = sum(r.num_frames for r in species_results) if species_results else 0

    # Get species and presets list
    species_list = [r.species_name for r in species_results] if species_results else []
    presets_list = [r.preset_name for r in keypoint_results] if keypoint_results else []

    # Create experiment summary
    summary = ExperimentSummary(
        experiment_name=experiment_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_frames=total_frames,
        species=species_list,
        presets_tested=presets_list,
        keypoint_results=keypoint_results,
        species_results=species_results,
        action_results=action_results,
    )

    # Generate dashboard
    logger.info("Generating HTML dashboard...")
    dashboard_gen = DashboardGenerator(output_dir)
    dashboard_path = dashboard_gen.generate_full_dashboard(
        summary=summary,
        gif_paths=gif_paths,
        plot_paths=plot_paths,
    )

    logger.info(f"Dashboard generated: {dashboard_path}")

    return dashboard_path


def generate_standalone_visualizations(
    experiment_dir: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate standalone visualization files.

    Args:
        experiment_dir: Path to experiment directory
        output_dir: Path to output directory

    Returns:
        Dict of visualization paths
    """
    output_paths = {}

    # Try to load raw data for additional visualizations
    # This requires the actual numpy arrays which may be saved in experiment outputs

    # Look for .npy or .npz files with keypoint data
    npy_files = list(experiment_dir.rglob("*.npy")) + list(experiment_dir.rglob("*.npz"))

    for npy_file in npy_files:
        try:
            if npy_file.suffix == ".npz":
                data = np.load(npy_file, allow_pickle=True)
                if "keypoints" in data:
                    logger.info(f"Found keypoint data in {npy_file}")
            else:
                data = np.load(npy_file, allow_pickle=True)
        except Exception as e:
            logger.warning(f"Failed to load {npy_file}: {e}")

    return output_paths


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive report from SuperAnimal experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_report.py --input outputs/full_pipeline/20241127_123456
    python generate_report.py --input outputs/full_pipeline/20241127_123456 --name "Mouse Behavior Study"
    python generate_report.py --input outputs/full_pipeline/20241127_123456 --output-dir reports/final
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to experiment output directory",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for report (default: <input>/report)",
    )

    parser.add_argument(
        "--name", "-n",
        type=str,
        default="SuperAnimal Behavior Analysis",
        help="Experiment name for report title",
    )

    parser.add_argument(
        "--include-gifs",
        action="store_true",
        default=True,
        help="Include GIFs in report (default: True)",
    )

    parser.add_argument(
        "--no-gifs",
        action="store_true",
        help="Exclude GIFs from report",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")

    logger.info("=" * 60)
    logger.info("SuperAnimal Behavior Analysis - Report Generator")
    logger.info("=" * 60)

    # Validate input directory
    experiment_dir = Path(args.input)
    if not experiment_dir.exists():
        logger.error(f"Input directory not found: {experiment_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir / "report"

    # Generate report
    include_gifs = args.include_gifs and not args.no_gifs

    try:
        report_path = generate_comprehensive_report(
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            experiment_name=args.name,
            include_gifs=include_gifs,
        )

        logger.info("=" * 60)
        logger.info("Report Generation Complete!")
        logger.info("=" * 60)
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Open in browser: file://{report_path.absolute()}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
