#!/usr/bin/env python3
"""
Comprehensive Analysis Pipeline - One-Click Execution

This script runs ALL experiments with ALL combinations and generates
a complete visualization dashboard.

Configuration is centrally managed in configs/config.yaml

Usage:
    # Quick test (debug mode, ~2 min)
    python run_comprehensive.py --debug

    # All combinations with minimal frames (~5 min)
    python run_comprehensive.py --debug-full

    # Standard analysis (~10 min)
    python run_comprehensive.py

    # Complete analysis with all species/presets (~30 min)
    python run_comprehensive.py --all

    # Custom configuration
    python run_comprehensive.py --species mouse,dog --presets full,standard,minimal
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import subprocess
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import setup_logging, get_device

logger = logging.getLogger(__name__)


# ============================================================
# Configuration - Loaded from configs/config.yaml
# ============================================================

def load_config_from_yaml() -> Dict[str, Any]:
    """Load mode configurations from configs/config.yaml."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    # Default fallback configuration
    default_config = {
        "debug": {
            "max_frames": 50,
            "gif_max_frames": 30,
            "species": ["mouse"],
            "presets": ["full", "minimal"],
            "gifs": False,
            "models": ["superanimal"],
        },
        "debug-full": {
            "max_frames": 20,
            "gif_max_frames": 15,
            "species": ["mouse", "dog", "horse"],
            "presets": ["full", "standard", "mars", "locomotion", "minimal"],
            "gifs": True,
            "models": ["superanimal", "yolo_pose"],
        },
        "standard": {
            "max_frames": 200,
            "gif_max_frames": 80,
            "species": ["mouse", "dog"],
            "presets": ["full", "standard", "minimal"],
            "gifs": True,
            "models": ["superanimal"],
        },
        "full": {
            "max_frames": 300,
            "gif_max_frames": 100,
            "species": ["mouse", "dog", "horse"],
            "presets": ["full", "standard", "mars", "locomotion", "minimal"],
            "gifs": True,
            "models": ["superanimal", "yolo_pose"],
        },
    }

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_config

    try:
        import yaml
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)

        # Extract modes from yaml config
        if "modes" in yaml_config:
            modes = yaml_config["modes"]
            config = {}
            for mode_name, mode_cfg in modes.items():
                # Normalize mode name (debug_full -> debug-full)
                normalized_name = mode_name.replace("_", "-")
                config[normalized_name] = {
                    "max_frames": mode_cfg.get("max_frames", 200),
                    "gif_max_frames": mode_cfg.get("gif_max_frames", 80),
                    "species": mode_cfg.get("species", ["mouse"]),
                    "presets": mode_cfg.get("presets", ["full", "minimal"]),
                    "gifs": mode_cfg.get("gifs", True),
                    "models": mode_cfg.get("models", ["superanimal"]),
                }
            return config

    except Exception as e:
        logger.warning(f"Failed to load config.yaml: {e}, using defaults")

    return default_config


# Load configuration at module level
DEFAULT_CONFIG = load_config_from_yaml()

# Keypoint presets (also defined in configs/model/topviewmouse.yaml)
KEYPOINT_PRESETS = {
    "full": None,  # All 27 keypoints
    "standard": ["nose", "left_ear", "right_ear", "neck", "mouse_center",
                 "left_shoulder", "right_shoulder", "left_hip", "right_hip",
                 "tail_base", "tail_end"],
    "mars": ["nose", "left_ear", "right_ear", "neck",
             "left_hip", "right_hip", "tail_base"],
    "locomotion": ["nose", "neck", "mouse_center", "tail_base", "tail_end"],
    "minimal": ["nose", "mouse_center", "tail_base"],
}


# ============================================================
# Pipeline Steps
# ============================================================

def run_single_video_analysis(
    output_dir: Path,
    max_frames: int,
    generate_gifs: bool = True,
    species: str = "mouse",
) -> Dict:
    """Run single video analysis."""
    logger.info(f"Running single video analysis for {species}...")

    cmd = [
        "python", "run.py",
        f"data.video.max_frames={max_frames}",
        f"report.gifs={'true' if generate_gifs else 'false'}",
        f"report.gif_max_frames={min(max_frames, 100)}",
        f"paths.experiments={output_dir / 'single_video' / species}",
    ]

    # Timeout scales with max_frames (base 10 min + 2 sec per frame)
    timeout_sec = 600 + (max_frames * 2)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        success = result.returncode == 0
        if not success:
            logger.warning(f"Single video analysis returned code {result.returncode}")
            logger.debug(result.stderr)
    except subprocess.TimeoutExpired:
        logger.error(f"Single video analysis timed out (timeout={timeout_sec}s)")
        success = False
    except Exception as e:
        logger.error(f"Single video analysis failed: {e}")
        success = False

    return {"step": "single_video", "species": species, "success": success}


def run_keypoint_comparison(
    output_dir: Path,
    max_frames: int,
    presets: List[str],
    generate_gifs: bool = True,
) -> Dict:
    """Run keypoint preset comparison."""
    logger.info(f"Running keypoint comparison with presets: {presets}...")

    cmd = [
        "python", "run_keypoint_comparison.py",
        f"data.video.max_frames={max_frames}",
        f"report.gif_max_frames={min(max_frames, 80)}",
    ]

    # Keypoint comparison runs multiple presets, needs more time
    # Base 15 min + 3 sec per frame per preset (5 presets default)
    timeout_sec = 900 + (max_frames * 3 * len(presets))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        success = result.returncode == 0

        # Copy results to output dir
        kp_output = Path("outputs/keypoint_comparison")
        if kp_output.exists():
            dest = output_dir / "keypoint_comparison"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(kp_output, dest)

    except subprocess.TimeoutExpired:
        logger.error(f"Keypoint comparison timed out (timeout={timeout_sec}s)")
        success = False
    except Exception as e:
        logger.error(f"Keypoint comparison failed: {e}")
        success = False

    return {"step": "keypoint_comparison", "presets": presets, "success": success}


def run_cross_species(
    output_dir: Path,
    max_frames: int,
    species: List[str],
    generate_gifs: bool = True,
) -> Dict:
    """Run cross-species comparison."""
    logger.info(f"Running cross-species comparison: {species}...")

    species_str = "[" + ",".join(species) + "]"

    cmd = [
        "python", "run_cross_species.py",
        f"data.video.max_frames={max_frames}",
        f"species={species_str}",
        f"report.gifs={'true' if generate_gifs else 'false'}",
    ]

    # Cross-species runs multiple species, needs more time
    # Base 15 min + 3 sec per frame per species
    timeout_sec = 900 + (max_frames * 3 * len(species))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        success = result.returncode == 0

        # Copy latest results (filter out .DS_Store and other non-directories)
        cross_dir = Path("outputs/cross_species")
        if cross_dir.exists():
            # Only get actual directories, ignore files like .DS_Store
            subdirs = [d for d in cross_dir.iterdir() if d.is_dir()]
            if subdirs:
                latest = sorted(subdirs, key=os.path.getmtime, reverse=True)
                if latest:
                    dest = output_dir / "cross_species"
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(latest[0], dest)

    except subprocess.TimeoutExpired:
        logger.error(f"Cross-species comparison timed out (timeout={timeout_sec}s)")
        success = False
    except Exception as e:
        logger.error(f"Cross-species comparison failed: {e}")
        success = False

    return {"step": "cross_species", "species": species, "success": success}


def run_preset_analysis(
    output_dir: Path,
    max_frames: int,
    preset: str,
) -> Dict:
    """Run analysis with specific keypoint preset."""
    logger.info(f"Running analysis with preset: {preset}...")

    preset_dir = output_dir / "presets" / f"preset_{preset}"
    preset_dir.mkdir(parents=True, exist_ok=True)

    # Use preset-specific keypoints (if applicable)
    cmd = [
        "python", "run.py",
        f"data.video.max_frames={max_frames}",
        "report.gifs=false",
        f"paths.experiments={preset_dir}",
    ]

    # Add keypoint filter if not full
    if preset != "full" and preset in KEYPOINT_PRESETS:
        keypoints = KEYPOINT_PRESETS[preset]
        if keypoints:
            cmd.append(f"model.use_keypoints={keypoints}")

    # Timeout scales with max_frames
    timeout_sec = 300 + (max_frames * 2)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"Preset analysis timed out for {preset} (timeout={timeout_sec}s)")
        success = False
    except Exception as e:
        logger.error(f"Preset analysis failed for {preset}: {e}")
        success = False

    return {"step": "preset_analysis", "preset": preset, "success": success}


def run_model_comparison(
    output_dir: Path,
    max_frames: int,
    models: List[str],
) -> Dict:
    """Run model comparison."""
    logger.info(f"Running model comparison: {models}...")

    model_dir = output_dir / "model_comparison"

    cmd = [
        "python", "run_model_comparison.py",
        f"--max-frames={max_frames}",
        f"--output={model_dir}",
        "--include-baselines",
    ]

    # Model comparison runs multiple models
    timeout_sec = 900 + (max_frames * 3 * len(models))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"Model comparison timed out (timeout={timeout_sec}s)")
        success = False
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        success = False

    return {"step": "model_comparison", "models": models, "success": success}


def generate_visualizations(
    output_dir: Path,
    ground_truth_path: Optional[Path] = None,
) -> Dict:
    """Generate all visualizations from collected results.

    Args:
        output_dir: Output directory containing analysis results
        ground_truth_path: Optional path to ground truth labels file

    Returns:
        Dict with step info and success status
    """
    logger.info("Generating visualizations...")

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Generate keypoint comparison visualizations with accuracy/F1 metrics
    try:
        from src.analysis.keypoint_visualizer import (
            KeypointVisualizer,
            compare_presets_with_metrics,
            KeypointAnalysisResult,
        )
        import numpy as np

        kp_viz = KeypointVisualizer(output_dir=viz_dir / "keypoint_comparison")

        # Generate keypoint count analysis
        all_kp_names = ['nose', 'left_ear', 'right_ear', 'neck', 'mouse_center',
                        'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                        'tail_base', 'tail_end', 'mid_back', 'head_midpoint',
                        'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end']

        kp_viz.create_keypoint_count_analysis(all_kp_names)

        # Try to load keypoint data and compute metrics
        kp_data_dir = output_dir / "keypoint_comparison"
        keypoints = None
        keypoint_names = all_kp_names
        video_path = None

        if kp_data_dir.exists():
            # First try NPZ files
            npz_files = list(kp_data_dir.glob("**/*.npz"))

            # If no NPZ, try loading from JSON prediction files
            if not npz_files:
                json_files = list(kp_data_dir.glob("**/predictions/*.json"))
                if json_files:
                    # Load first JSON file to get keypoints
                    try:
                        with open(json_files[0]) as f:
                            pred_data = json.load(f)

                        # Extract keypoints from JSON format
                        # Format: list of frames, each with 'bodyparts' containing detections
                        frames_kp = []
                        for frame_data in pred_data:
                            if isinstance(frame_data, dict) and 'bodyparts' in frame_data:
                                # Get first detection (index 0)
                                detections = frame_data['bodyparts']
                                if len(detections) > 0 and detections[0][0][0] != -1:
                                    frames_kp.append(detections[0])
                                else:
                                    # Use zeros if no detection
                                    frames_kp.append([[0, 0, 0]] * 27)
                            elif isinstance(frame_data, list):
                                frames_kp.append(frame_data[0] if len(frame_data) > 0 else [[0, 0, 0]] * 27)

                        if frames_kp:
                            keypoints = np.array(frames_kp)
                            logger.info(f"  Loaded keypoints from JSON: {keypoints.shape}")
                    except Exception as json_e:
                        logger.warning(f"Failed to load keypoints from JSON: {json_e}")

            # Look for keypoint data files (NPZ)
            for npz_file in kp_data_dir.glob("**/*.npz"):
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    keypoints = data.get('keypoints', None)
                    keypoint_names = data.get('keypoint_names', all_kp_names)

                    if keypoints is not None and len(keypoints) > 0:
                        # Compare presets with full metrics
                        presets = ["full", "standard", "mars", "locomotion", "minimal"]
                        results, metrics = compare_presets_with_metrics(
                            keypoints,
                            list(keypoint_names) if hasattr(keypoint_names, '__iter__') else all_kp_names,
                            presets=presets,
                            fps=30.0,
                        )

                        # Generate performance visualization
                        perf_plot_path = kp_viz.create_performance_by_keypoint_count(
                            results,
                            video_name=npz_file.stem,
                        )
                        logger.info(f"  Generated performance plot: {perf_plot_path}")

                        # Generate hierarchical action comparison
                        hier_plot_path = kp_viz.create_hierarchical_action_comparison(
                            results,
                            video_name=npz_file.stem,
                        )
                        if hier_plot_path:
                            logger.info(f"  Generated hierarchical action comparison: {hier_plot_path}")

                        # Generate confusion matrix grid
                        cm_plot_path = kp_viz.create_confusion_matrix_grid(
                            results,
                            video_name=npz_file.stem,
                        )
                        if cm_plot_path:
                            logger.info(f"  Generated confusion matrix grid: {cm_plot_path}")

                        # Generate preset × action GIFs
                        # Find source video file
                        video_path = None
                        for video_ext in ['*.mp4', '*.avi', '*.mov']:
                            video_files = list(Path("data/raw").glob(video_ext))
                            if video_files:
                                video_path = video_files[0]
                                break

                        # Also check if video path is stored in npz
                        if video_path is None and 'video_path' in data:
                            stored_path = Path(str(data['video_path']))
                            if stored_path.exists():
                                video_path = stored_path

                        if video_path and video_path.exists():
                            try:
                                # Read config for GIF settings
                                gif_frames = 80  # default
                                config_path = PROJECT_ROOT / "configs" / "config.yaml"
                                if config_path.exists():
                                    import yaml
                                    with open(config_path) as cf:
                                        cfg = yaml.safe_load(cf)
                                        gif_frames = cfg.get("report", {}).get("preset_gif_frames", 80)

                                preset_action_gifs = kp_viz.create_preset_action_gifs(
                                    video_path=video_path,
                                    results=results,
                                    all_keypoint_names=list(keypoint_names) if hasattr(keypoint_names, '__iter__') else all_kp_names,
                                    keypoints=keypoints,
                                    max_frames_per_gif=gif_frames,
                                    fps=10.0,
                                    max_gifs_per_combo=2,
                                )

                                total_preset_gifs = sum(
                                    len(paths)
                                    for preset_dict in preset_action_gifs.values()
                                    for paths in preset_dict.values()
                                )
                                logger.info(f"  Generated {total_preset_gifs} preset×action GIFs")

                                # Save GIF paths for dashboard
                                # Key format: "preset_{preset_name}_{action_name}" for dashboard parsing
                                gif_paths_dict = {}
                                for preset_name, action_dict in preset_action_gifs.items():
                                    for action_name, paths in action_dict.items():
                                        key = f"preset_{preset_name}_{action_name}"
                                        gif_paths_dict[key] = [str(p) for p in paths]

                                gif_index_path = viz_dir / "keypoint_comparison" / "preset_action_gifs.json"
                                with open(gif_index_path, 'w') as f:
                                    json.dump(gif_paths_dict, f, indent=2)
                                logger.info(f"  Saved preset action GIF index: {gif_index_path}")

                            except Exception as gif_e:
                                logger.warning(f"Failed to generate preset×action GIFs: {gif_e}")
                        else:
                            logger.debug("Video file not found for preset×action GIFs")

                        # Save metrics as JSON for dashboard
                        metrics_path = viz_dir / "keypoint_comparison" / "performance_metrics.json"
                        metrics_dict = {
                            "presets_compared": metrics.presets_compared,
                            "reference_preset": metrics.reference_preset,
                            "accuracy_by_preset": metrics.accuracy_by_preset,
                            "f1_by_preset": metrics.f1_by_preset,
                            "accuracy_drop_from_full": metrics.accuracy_drop_from_full,
                            "best_preset_by_accuracy": metrics.best_preset_by_accuracy,
                            "results": [
                                {
                                    "preset_name": r.preset_name,
                                    "num_keypoints": r.num_keypoints,
                                    "accuracy": r.accuracy,
                                    "f1_scores": r.f1_scores,
                                    "precision": r.precision,
                                    "recall": r.recall,
                                    "agreement_with_full": r.agreement_with_full,
                                    "action_distribution": r.action_distribution,
                                    "mean_confidence": r.mean_confidence,
                                }
                                for r in results
                            ]
                        }
                        with open(metrics_path, 'w') as f:
                            json.dump(metrics_dict, f, indent=2)
                        logger.info(f"  Saved performance metrics: {metrics_path}")

                        break  # Process first valid file
                except Exception as e:
                    logger.warning(f"Failed to process {npz_file}: {e}")

            # If no NPZ processed but we loaded from JSON, analyze those
            if keypoints is not None and 'results' not in dir():
                try:
                    # Compare presets with full metrics
                    presets = ["full", "standard", "mars", "locomotion", "minimal"]
                    results, metrics = compare_presets_with_metrics(
                        keypoints,
                        keypoint_names,
                        presets=presets,
                        fps=30.0,
                    )

                    # Find video file for GIF generation
                    for video_ext in ['*.mp4', '*.avi', '*.mov']:
                        video_files = list(Path("data/raw").glob(video_ext))
                        if video_files:
                            video_path = video_files[0]
                            break

                    # Generate performance visualization
                    perf_plot_path = kp_viz.create_performance_by_keypoint_count(
                        results,
                        video_name="mouse_topview_sample",
                    )
                    logger.info(f"  Generated performance plot: {perf_plot_path}")

                    # Generate hierarchical action comparison
                    hier_plot_path = kp_viz.create_hierarchical_action_comparison(
                        results,
                        video_name="mouse_topview_sample",
                    )
                    if hier_plot_path:
                        logger.info(f"  Generated hierarchical action comparison: {hier_plot_path}")

                    # Generate confusion matrix grid
                    cm_plot_path = kp_viz.create_confusion_matrix_grid(
                        results,
                        video_name="mouse_topview_sample",
                    )
                    if cm_plot_path:
                        logger.info(f"  Generated confusion matrix grid: {cm_plot_path}")

                    # Save metrics as JSON for dashboard
                    metrics_path = viz_dir / "keypoint_comparison" / "performance_metrics.json"
                    metrics_dict = {
                        "presets_compared": metrics.presets_compared,
                        "reference_preset": metrics.reference_preset,
                        "accuracy_by_preset": metrics.accuracy_by_preset,
                        "f1_by_preset": metrics.f1_by_preset,
                        "accuracy_drop_from_full": metrics.accuracy_drop_from_full,
                        "best_preset_by_accuracy": metrics.best_preset_by_accuracy,
                        "results": [
                            {
                                "preset_name": r.preset_name,
                                "num_keypoints": r.num_keypoints,
                                "accuracy": r.accuracy,
                                "f1_scores": r.f1_scores,
                                "precision": r.precision,
                                "recall": r.recall,
                                "agreement_with_full": r.agreement_with_full,
                                "action_distribution": r.action_distribution,
                                "mean_confidence": r.mean_confidence,
                            }
                            for r in results
                        ]
                    }
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics_dict, f, indent=2)
                    logger.info(f"  Saved performance metrics: {metrics_path}")

                except Exception as e:
                    logger.warning(f"Failed to analyze JSON keypoints: {e}")
                    import traceback
                    traceback.print_exc()

        logger.info("  Generated keypoint comparison visualizations")

        # === Ground Truth Evaluation (if labels provided) ===
        if ground_truth_path and Path(ground_truth_path).exists():
            try:
                from src.evaluation.metrics import (
                    load_ground_truth_labels,
                    compute_all_preset_metrics,
                    save_metrics_report,
                    print_metrics_report,
                )
                from src.analysis.keypoint_visualizer import GroundTruthComparisonGifGenerator

                logger.info(f"  Loading ground truth labels from: {ground_truth_path}")

                # Get number of frames from results
                num_frames = results[0].action_labels.shape[0] if results else None

                gt_labels, gt_metadata = load_ground_truth_labels(
                    ground_truth_path,
                    num_frames=num_frames,
                )

                logger.info(f"  Ground truth loaded: {len(gt_labels)} frames")
                logger.info(f"  Class distribution: {gt_metadata.get('class_distribution', {})}")

                # Compute metrics for all presets vs ground truth
                preset_predictions = {}
                for r in results:
                    preset_predictions[r.preset_name] = (r.action_labels, r.num_keypoints)

                gt_report = compute_all_preset_metrics(
                    preset_predictions=preset_predictions,
                    ground_truth=gt_labels,
                    ground_truth_source=str(ground_truth_path),
                )

                # Print report to console
                print_metrics_report(gt_report)

                # Save metrics report
                gt_metrics_dir = viz_dir / "ground_truth_evaluation"
                gt_metrics_dir.mkdir(parents=True, exist_ok=True)

                metrics_json_path = gt_metrics_dir / "gt_metrics_report.json"
                save_metrics_report(gt_report, metrics_json_path)
                logger.info(f"  Saved GT metrics report: {metrics_json_path}")

                # Generate GT comparison GIFs
                if video_path.exists():
                    gt_gif_gen = GroundTruthComparisonGifGenerator(
                        output_dir=gt_metrics_dir / "gifs"
                    )

                    gt_gifs = gt_gif_gen.create_all_preset_gt_gifs(
                        video_path=video_path,
                        keypoints=keypoints,
                        all_keypoint_names=keypoint_names,
                        results=results,
                        ground_truth=gt_labels,
                        max_frames=80,
                        fps=8.0,
                    )

                    if gt_gifs:
                        logger.info(f"  Generated {len(gt_gifs)} GT comparison GIFs")

                        # Save GIF paths index
                        gif_index_path = gt_metrics_dir / "gt_gifs_index.json"
                        with open(gif_index_path, 'w') as f:
                            json.dump({k: str(v) for k, v in gt_gifs.items()}, f, indent=2)

            except FileNotFoundError as fnf_e:
                logger.warning(f"Ground truth file not found: {fnf_e}")
            except Exception as gt_e:
                logger.warning(f"Ground truth evaluation failed: {gt_e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        logger.warning(f"Keypoint visualization failed: {e}")
        success = False

    # Generate species comparison visualizations
    try:
        from src.analysis.species_visualizer import SpeciesVisualizer, SpeciesAnalysisResult

        sp_viz = SpeciesVisualizer(output_dir=viz_dir / "species_comparison")

        # Load results from cross-species directory
        cross_dir = output_dir / "cross_species"
        results = []

        if cross_dir.exists():
            for json_file in cross_dir.glob("**/*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    if isinstance(data, dict):
                        for key, val in data.items():
                            if isinstance(val, dict) and ("body_size" in val or "total_frames" in val):
                                body_stats = val.get("body_size", {})
                                if isinstance(body_stats, (int, float)):
                                    body_stats = {"mean": body_stats, "std": 0}

                                action_dist = {}
                                for action in ["stationary", "walking", "running"]:
                                    if action in val:
                                        if isinstance(val[action], dict):
                                            action_dist[action] = val[action].get("percentage", 0)
                                        else:
                                            action_dist[action] = float(val[action])

                                result = SpeciesAnalysisResult(
                                    species_name=key,
                                    model_type=val.get("model_type", "unknown"),
                                    num_frames=val.get("total_frames", 0),
                                    body_size_mean=body_stats.get("mean", 0),
                                    body_size_std=body_stats.get("std", 0),
                                    action_distribution=action_dist,
                                )
                                results.append(result)
                except Exception:
                    pass

        if results:
            sp_viz.create_comprehensive_comparison(results)
            logger.info("  Generated species comparison visualizations")

    except Exception as e:
        logger.warning(f"Species visualization failed: {e}")

    return {"step": "visualizations", "success": success}


def generate_dashboard(output_dir: Path, config: Dict) -> Dict:
    """Generate comprehensive HTML dashboard."""
    logger.info("Generating comprehensive dashboard...")

    try:
        cmd = [
            "python", "generate_report.py",
            f"--input={output_dir}",
            f"--output-dir={output_dir / 'report'}",
            f"--name=SuperAnimal Comprehensive Analysis",
            "--include-gifs",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        success = result.returncode == 0

        # Copy dashboard to root
        dashboard_src = output_dir / "report" / "dashboard.html"
        dashboard_dst = output_dir / "final_dashboard.html"
        if dashboard_src.exists():
            shutil.copy(dashboard_src, dashboard_dst)
            logger.info(f"  Dashboard saved: {dashboard_dst}")

    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        success = False

    return {"step": "dashboard", "success": success}


# ============================================================
# Main Pipeline
# ============================================================

def run_comprehensive_pipeline(
    mode: str = "standard",
    output_dir: Optional[Path] = None,
    species: Optional[List[str]] = None,
    presets: Optional[List[str]] = None,
    max_frames: Optional[int] = None,
    skip_experiments: bool = False,
    ground_truth_path: Optional[Path] = None,
) -> Path:
    """
    Run comprehensive analysis pipeline.

    Args:
        mode: "debug", "standard", or "full"
        output_dir: Output directory (default: auto-generated)
        species: Species to analyze
        presets: Keypoint presets to compare
        max_frames: Max frames per video
        skip_experiments: Skip experiments, only generate visualizations
        ground_truth_path: Path to ground truth labels file (CSV, JSON, TXT, NPY)

    Returns:
        Path to output directory
    """
    # Get configuration
    config = DEFAULT_CONFIG.get(mode, DEFAULT_CONFIG["standard"]).copy()

    # Override with custom settings
    if species:
        config["species"] = species
    if presets:
        config["presets"] = presets
    if max_frames:
        config["max_frames"] = max_frames

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/comprehensive/{timestamp}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SuperAnimal Comprehensive Analysis Pipeline")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Species: {config['species']}")
    logger.info(f"Presets: {config['presets']}")
    logger.info(f"Max Frames: {config['max_frames']}")
    logger.info("=" * 60)

    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "mode": mode,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    results = []

    if not skip_experiments:
        # Step 1: Single Video Analysis
        logger.info("\n[1/6] Single Video Analysis")
        result = run_single_video_analysis(
            output_dir=output_dir,
            max_frames=config["max_frames"],
            generate_gifs=config["gifs"],
        )
        results.append(result)

        # Step 2: Keypoint Comparison
        logger.info("\n[2/6] Keypoint Preset Comparison")
        result = run_keypoint_comparison(
            output_dir=output_dir,
            max_frames=config["max_frames"],
            presets=config["presets"],
            generate_gifs=config["gifs"],
        )
        results.append(result)

        # Step 3: Cross-Species Comparison
        logger.info("\n[3/6] Cross-Species Comparison")
        result = run_cross_species(
            output_dir=output_dir,
            max_frames=config["max_frames"],
            species=config["species"],
            generate_gifs=config["gifs"],
        )
        results.append(result)

        # Step 4: Per-Preset Analysis (full mode only)
        if mode == "full":
            logger.info("\n[4/6] Per-Preset Analysis")
            for preset in config["presets"]:
                result = run_preset_analysis(
                    output_dir=output_dir,
                    max_frames=config["max_frames"] // 2,  # Faster
                    preset=preset,
                )
                results.append(result)
        else:
            logger.info("\n[4/6] Per-Preset Analysis (skipped in {mode} mode)")

        # Step 5: Model Comparison
        logger.info("\n[5/6] Model Comparison")
        result = run_model_comparison(
            output_dir=output_dir,
            max_frames=config["max_frames"],
            models=config["models"],
        )
        results.append(result)

    # Step 6: Generate Visualizations
    logger.info("\n[6/6] Generating Visualizations and Dashboard")
    result = generate_visualizations(output_dir, ground_truth_path=ground_truth_path)
    results.append(result)

    result = generate_dashboard(output_dir, config)
    results.append(result)

    # Save results summary
    summary = {
        "mode": mode,
        "config": config,
        "results": results,
        "success_count": sum(1 for r in results if r.get("success", False)),
        "total_count": len(results),
        "output_dir": str(output_dir),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Success: {summary['success_count']}/{summary['total_count']} steps")
    logger.info(f"Output: {output_dir}")

    # List key output files
    dashboard = output_dir / "final_dashboard.html"
    if dashboard.exists():
        logger.info(f"\n🎯 Dashboard: {dashboard}")
        logger.info(f"   Open: file://{dashboard.absolute()}")

    return output_dir


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive SuperAnimal behavior analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (debug mode, ~2 min)
    python run_comprehensive.py --debug

    # Debug with ALL combinations but minimal frames (~5 min)
    python run_comprehensive.py --debug-full

    # Standard analysis (~10 min)
    python run_comprehensive.py

    # Full analysis with all species/presets (~25 min)
    python run_comprehensive.py --all

    # Custom configuration
    python run_comprehensive.py --species mouse,dog --presets full,standard,minimal

    # Only generate visualizations from existing results
    python run_comprehensive.py --visualize-only --input outputs/comprehensive/20241127_123456
        """
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Debug mode: quick test with mouse only, 50 frames (~2 min)",
    )

    parser.add_argument(
        "--debug-full", "-df",
        action="store_true",
        help="Debug-full mode: ALL combinations (species/presets/models) with minimal 20 frames (~5 min)",
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Full mode (all species, all presets, 300 frames, ~25 min)",
    )

    parser.add_argument(
        "--species", "-s",
        type=str,
        default=None,
        help="Comma-separated species list (e.g., mouse,dog,horse)",
    )

    parser.add_argument(
        "--presets", "-p",
        type=str,
        default=None,
        help="Comma-separated presets (e.g., full,standard,minimal)",
    )

    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum frames per video",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip experiments, only generate visualizations",
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory for visualize-only mode",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=None,
        help="Path to ground truth labels file (CSV, JSON, TXT, or NPY format)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")

    # Determine mode
    if args.debug_full:
        mode = "debug-full"
    elif args.debug:
        mode = "debug"
    elif args.all:
        mode = "full"
    else:
        mode = "standard"

    # Parse species and presets
    species = args.species.split(",") if args.species else None
    presets = args.presets.split(",") if args.presets else None

    # Handle visualize-only mode
    if args.visualize_only:
        if not args.input:
            logger.error("--input required for --visualize-only mode")
            sys.exit(1)
        output_dir = Path(args.input)
    else:
        output_dir = Path(args.output) if args.output else None

    # Run pipeline
    try:
        ground_truth_path = Path(args.labels) if args.labels else None

        result_dir = run_comprehensive_pipeline(
            mode=mode,
            output_dir=output_dir,
            species=species,
            presets=presets,
            max_frames=args.max_frames,
            skip_experiments=args.visualize_only,
            ground_truth_path=ground_truth_path,
        )

        # Try to open dashboard
        dashboard = result_dir / "final_dashboard.html"
        if dashboard.exists():
            try:
                import webbrowser
                webbrowser.open(f"file://{dashboard.absolute()}")
            except Exception:
                pass

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
