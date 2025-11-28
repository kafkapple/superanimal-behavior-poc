"""
Comprehensive Report Generator - Aggregates experiment results and generates reports.

This module collects results from experiments and generates HTML dashboards.
Moved from generate_report.py for better modularity.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from .dashboard import (
    DashboardGenerator,
    ExperimentSummary,
    KeypointPresetResult,
    SpeciesResult,
    ActionRecognitionResult,
)
from .keypoint_visualizer import KEYPOINT_PRESETS

logger = logging.getLogger(__name__)


def collect_keypoint_results(experiment_dir: Path) -> List[KeypointPresetResult]:
    """
    Collect keypoint comparison results from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of KeypointPresetResult
    """
    results = []

    # Check for performance_metrics.json
    perf_metrics_files = list(experiment_dir.glob("**/performance_metrics.json"))
    if perf_metrics_files:
        for perf_file in perf_metrics_files:
            try:
                with open(perf_file) as f:
                    perf_data = json.load(f)

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

    # Fall back to directory scanning
    kp_dirs = list(experiment_dir.glob("**/keypoint_comparison*")) + \
              list(experiment_dir.glob("**/preset_*"))

    comp_dir = experiment_dir / "comprehensive"
    if comp_dir.exists():
        kp_dirs.extend(list(comp_dir.glob("preset_*")))

    preset_data = {}

    for kp_dir in kp_dirs:
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

                preset_name = None
                for preset in KEYPOINT_PRESETS.keys():
                    if preset in str(kp_dir).lower():
                        preset_name = preset
                        break

                if preset_name and preset_name not in preset_data:
                    preset_data[preset_name] = data

            except Exception as e:
                logger.warning(f"Failed to parse {mf}: {e}")

    for preset_name, data in preset_data.items():
        preset_kps = KEYPOINT_PRESETS.get(preset_name)
        num_kps = len(preset_kps) if preset_kps else 27

        action_dist = {}
        for action in ["stationary", "walking", "running"]:
            if f"{action}_pct" in data:
                action_dist[action] = data[f"{action}_pct"]
            elif action in data:
                action_dist[action] = data[action]

        result = KeypointPresetResult(
            preset_name=preset_name,
            num_keypoints=num_kps,
            keypoint_names=preset_kps or [],
            action_accuracy=data.get("accuracy", 0),
            action_distribution=action_dist,
            mean_confidence=data.get("mean_confidence", 0.8),
            trajectory_distance=data.get("total_distance", 0),
            velocity_stats={
                "mean": data.get("mean_velocity", 0),
                "std": data.get("std_velocity", 0),
            },
        )
        results.append(result)

    results.sort(key=lambda x: x.num_keypoints, reverse=True)
    return results


def collect_species_results(experiment_dir: Path) -> List[SpeciesResult]:
    """Collect cross-species comparison results."""
    results = []
    species_dirs = list(experiment_dir.glob("**/cross_species*")) + \
                   list(experiment_dir.glob("**/all_species*"))

    species_data = {}

    for sp_dir in species_dirs:
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

        json_files = list(sp_dir.glob("**/*.json"))
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, dict) and "body_size" in val:
                            if key not in species_data:
                                species_data[key] = val
            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")

    for species_name, data in species_data.items():
        body_stats = data.get("body_size", {})
        if isinstance(body_stats, (int, float)):
            body_stats = {"mean": body_stats, "std": 0}

        action_dist = {}
        for action in ["stationary", "walking", "running"]:
            if action in data:
                val = data[action]
                action_dist[action] = val.get("percentage", 0) if isinstance(val, dict) else float(val)

        result = SpeciesResult(
            species_name=species_name,
            model_type=data.get("model_type", "unknown"),
            num_frames=int(data.get("total_frames", 0)),
            body_size_stats=body_stats,
            action_distribution=action_dist,
            velocity_stats={"mean": data.get("mean_velocity", 0)},
        )
        results.append(result)

    return results


def collect_action_results(experiment_dir: Path) -> List[ActionRecognitionResult]:
    """Collect action recognition results."""
    results = []
    eval_dirs = list(experiment_dir.glob("**/evaluation*")) + \
                list(experiment_dir.glob("**/model_comparison*"))

    model_data = {}

    for eval_dir in eval_dirs:
        json_files = list(eval_dir.glob("**/*.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if "experiments" in data:
                    for model_name, metrics in data["experiments"].items():
                        if model_name not in model_data:
                            model_data[model_name] = metrics

                if "reference" in data:
                    for key, val in data.items():
                        if isinstance(val, dict) and "accuracy" in val:
                            model_name = val.get("model", key)
                            if model_name not in model_data:
                                model_data[model_name] = val

            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")

    for model_name, data in model_data.items():
        f1_scores = data.get("f1_score", data.get("f1_scores", {}))
        if isinstance(f1_scores, (int, float)):
            f1_scores = {"overall": f1_scores}

        result = ActionRecognitionResult(
            model_name=model_name,
            accuracy=data.get("accuracy", 0),
            f1_scores=f1_scores,
            confusion_matrix=np.array(data["confusion_matrix"]) if "confusion_matrix" in data else None,
            consistency_score=data.get("smoothness_score", 0),
            per_class_accuracy=data.get("per_class_accuracy", {}),
        )
        results.append(result)

    results.sort(key=lambda x: x.accuracy, reverse=True)
    return results


def collect_gif_paths(experiment_dir: Path) -> Dict[str, List[Path]]:
    """Collect GIF paths organized by category."""
    gif_paths = {}
    all_gifs = list(experiment_dir.rglob("*.gif"))

    for gif_path in all_gifs:
        stem = gif_path.stem.lower()

        if "stationary" in stem:
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
    """Collect plot image paths."""
    plot_paths = {}
    png_files = list(experiment_dir.rglob("*.png"))

    for png_path in png_files:
        stem = png_path.stem.lower()
        if "thumb" in stem or "_small" in stem:
            continue

        if "trajectory" in stem:
            plot_paths["trajectory"] = png_path
        elif "velocity" in stem:
            plot_paths["velocity_profile"] = png_path
        elif "behavior" in stem or "timeline" in stem:
            plot_paths["behavior_timeline"] = png_path
        elif "comparison" in stem and "species" in stem:
            plot_paths["cross_species_comparison"] = png_path
        elif "body" in stem and "size" in stem:
            plot_paths["body_size_comparison"] = png_path

    return plot_paths


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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    keypoint_results = collect_keypoint_results(experiment_dir)
    species_results = collect_species_results(experiment_dir)
    action_results = collect_action_results(experiment_dir)
    gif_paths = collect_gif_paths(experiment_dir) if include_gifs else {}
    plot_paths = collect_plot_paths(experiment_dir)

    logger.info(f"Found: {len(keypoint_results)} presets, {len(species_results)} species, {len(action_results)} models")

    total_frames = sum(r.num_frames for r in species_results) if species_results else 0
    species_list = [r.species_name for r in species_results]
    presets_list = [r.preset_name for r in keypoint_results]

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

    dashboard_gen = DashboardGenerator(output_dir)
    dashboard_path = dashboard_gen.generate_full_dashboard(
        summary=summary,
        gif_paths=gif_paths,
        plot_paths=plot_paths,
    )

    logger.info(f"Dashboard generated: {dashboard_path}")
    return dashboard_path
