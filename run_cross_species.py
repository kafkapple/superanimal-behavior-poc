#!/usr/bin/env python3
"""
Cross-Species Action Recognition Pipeline

Compare actions (walking, running, stationary) and body size between multiple species.
Supports custom species configurations via configs/species/*.yaml

Usage:
    python run_cross_species.py                    # Default: mouse + dog
    python run_cross_species.py data.video.max_frames=100
    python run_cross_species.py species=[mouse,dog,horse]

Custom species config override:
    python run_cross_species.py species.mouse.velocity_thresholds.walking=2.5
"""
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import VideoDownloader
from src.models.predictor import SuperAnimalPredictor
from src.models.action_classifier import (
    UnifiedActionClassifier,
    CrossSpeciesComparator,
    ACTION_TYPES,
    VELOCITY_THRESHOLDS,
)
from src.analysis.behavior import estimate_body_size
from src.analysis.report_generator import HTMLReportGenerator, ActionGifGenerator
from src.utils.helpers import setup_logging, get_device

logger = logging.getLogger(__name__)


def load_species_config(species_name: str, config_dir: Path) -> dict:
    """
    Load species-specific configuration.

    Args:
        species_name: Name of species (mouse, dog, horse, etc.)
        config_dir: Path to configs directory

    Returns:
        Species configuration dictionary
    """
    # Default configurations (fallback)
    defaults = {
        "mouse": {
            "name": "mouse",
            "description": "Mouse (top view)",
            "model": {"type": "topviewmouse"},
            "sample": {"name": "mouse_topview"},
            "velocity_thresholds": {"stationary": 0.5, "walking": 3.0},
            "center_keypoint": "mouse_center",
            "center_fallback": ["mid_back", "neck"],
        },
        "dog": {
            "name": "dog",
            "description": "Dog (side view)",
            "model": {"type": "quadruped"},
            "sample": {"name": "dog_walking"},
            "velocity_thresholds": {"stationary": 0.5, "walking": 2.5},
            "center_keypoint": "back_middle",
            "center_fallback": ["neck_base", "back_base"],
        },
        "horse": {
            "name": "horse",
            "description": "Horse (side view)",
            "model": {"type": "quadruped"},
            "sample": {"name": "horse_running"},
            "velocity_thresholds": {"stationary": 0.3, "walking": 1.5},
            "center_keypoint": "back_middle",
            "center_fallback": ["back_base", "neck_base"],
        },
    }

    # Try to load from YAML config
    config_path = config_dir / "species" / f"{species_name}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
            logger.info(f"Loaded species config: {config_path}")
            return file_config

    # Use default if available
    if species_name in defaults:
        logger.info(f"Using default config for: {species_name}")
        return defaults[species_name]

    logger.warning(f"No config found for species: {species_name}")
    return None


def create_comparison_visualization(
    comparator: CrossSpeciesComparator,
    body_size_data: dict,
    output_path: Path,
):
    """Create visual comparison of action distributions and body sizes."""
    results = comparator.results

    if len(results) < 1:
        logger.warning("No results to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Action distribution bar chart
    ax1 = axes[0, 0]
    action_names = list(ACTION_TYPES.keys())
    x = np.arange(len(action_names))
    width = 0.8 / len(results)

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for i, (name, metrics) in enumerate(results.items()):
        percentages = [
            metrics.action_summary["actions"].get(action, {}).get("percentage", 0)
            for action in action_names
        ]
        ax1.bar(x + i * width, percentages, width, label=f"{name} ({metrics.species})", color=colors[i])

    ax1.set_xlabel("Action")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Action Distribution Comparison")
    ax1.set_xticks(x + width * (len(results) - 1) / 2)
    ax1.set_xticklabels([a.capitalize() for a in action_names])
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Plot 2: Velocity profiles
    ax2 = axes[0, 1]

    for i, (name, metrics) in enumerate(results.items()):
        frames = len(metrics.velocity)
        time_pct = np.linspace(0, 100, frames)

        # Normalize velocity by body size
        body_size = body_size_data.get(name, {}).get("mean", 100)
        normalized_vel = metrics.velocity / body_size * 30  # 30fps

        ax2.plot(time_pct, normalized_vel, label=f"{name}",
                 alpha=0.7, color=colors[i])

    ax2.set_xlabel("Video Progress (%)")
    ax2.set_ylabel("Normalized Velocity (body-lengths/sec)")
    ax2.set_title("Velocity Profile Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Body size comparison (bar chart)
    ax3 = axes[1, 0]
    species_names = list(body_size_data.keys())
    means = [body_size_data[s]["mean"] for s in species_names]
    stds = [body_size_data[s]["std"] for s in species_names]

    bars = ax3.bar(species_names, means, yerr=stds, capsize=5,
                   color=[colors[i] for i in range(len(species_names))], alpha=0.8)
    ax3.set_ylabel("Body Size (pixels)")
    ax3.set_title("Body Size Comparison")
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Body size box plot
    ax4 = axes[1, 1]
    box_data = []
    for name in species_names:
        per_frame = body_size_data[name].get("per_frame", [body_size_data[name]["mean"]])
        box_data.append(per_frame)

    bp = ax4.boxplot(box_data, labels=species_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_ylabel("Body Size (pixels)")
    ax4.set_title("Body Size Distribution")
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved comparison visualization: {output_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run cross-species action comparison."""
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("Cross-Species Action Recognition")
    logger.info("=" * 60)

    original_cwd = hydra.utils.get_original_cwd()
    config_dir = Path(original_cwd) / "configs"
    output_dir = Path(original_cwd) / "outputs" / "cross_species" / cfg.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")
    max_frames = cfg.data.video.max_frames or 200

    # Get species to compare (default: mouse and dog)
    species_list = cfg.get("species", ["mouse", "dog"])
    if isinstance(species_list, str):
        species_list = [species_list]

    logger.info(f"Comparing species: {species_list}")

    # Load species configs and check for overrides
    species_configs = {}
    for species_name in species_list:
        base_config = load_species_config(species_name, config_dir)
        if base_config is None:
            continue

        # Check for config overrides from command line
        if hasattr(cfg, 'species_override') and species_name in cfg.species_override:
            override = OmegaConf.to_container(cfg.species_override[species_name])
            base_config = _deep_merge(base_config, override)
            logger.info(f"Applied overrides for {species_name}")

        species_configs[species_name] = base_config

    comparator = CrossSpeciesComparator(fps=30.0)
    body_size_data = {}
    species_results = {}

    data_dir = Path(original_cwd) / cfg.data.data_dir
    downloader = VideoDownloader(str(data_dir))

    # ========== PROCESS EACH SPECIES ==========
    for idx, (species_name, species_cfg) in enumerate(species_configs.items()):
        logger.info(f"\n[{idx+1}/{len(species_configs)}] Processing {species_cfg.get('description', species_name)}")
        logger.info("-" * 40)

        # Download sample
        sample_name = species_cfg.get("sample", {}).get("name", species_name)
        video_path = downloader.download_sample(sample_name)
        if not video_path or not video_path.exists():
            logger.warning(f"Video not available for {species_name}")
            continue

        # Get model settings
        model_type = species_cfg.get("model", {}).get("type", "quadruped")
        use_keypoints = species_cfg.get("model", {}).get("use_keypoints", None)

        # Run inference
        predictor = SuperAnimalPredictor(
            model_type=model_type,
            model_name=cfg.model.model_name,
            video_adapt=False,
            device=device,
            use_keypoints=use_keypoints,
        )

        results = predictor.predict_video(
            video_path=video_path,
            output_dir=output_dir / f"{species_name}_predictions",
            max_frames=max_frames,
        )

        if results["keypoints"] is None:
            logger.warning(f"No keypoints for {species_name}")
            continue

        keypoint_names = predictor.get_keypoint_names()

        # Get velocity thresholds from species config
        velocity_thresholds = species_cfg.get("velocity_thresholds", None)

        # Action classification with species-specific thresholds
        classifier = UnifiedActionClassifier(
            species=species_name,
            fps=results["metadata"].get("fps", 30.0),
            smoothing_window=5,
        )

        # Override thresholds if specified in config
        if velocity_thresholds:
            classifier.thresholds = velocity_thresholds
            logger.info(f"  Using custom thresholds: {velocity_thresholds}")

        metrics = classifier.analyze(
            keypoints=results["keypoints"],
            keypoint_names=keypoint_names,
        )

        comparator.add_result(species_name.capitalize(), metrics)

        # Body size estimation
        body_stats = estimate_body_size(
            keypoints=results["keypoints"],
            keypoint_names=keypoint_names,
            model_type=model_type,
        )
        body_size_data[species_name.capitalize()] = body_stats

        species_results[species_name] = {
            "metrics": metrics,
            "body_size": body_stats,
            "video_path": video_path,
            "config": species_cfg,
        }

        logger.info(f"  Frames: {metrics.action_summary['total_frames']}")
        logger.info(f"  Body size: {body_stats['mean']:.1f} ± {body_stats['std']:.1f} px")
        for action, stats in metrics.action_summary["actions"].items():
            logger.info(f"  {action}: {stats['percentage']:.1f}%")

    # Skip if no results
    if len(species_results) < 1:
        logger.error("No species data available for comparison. Exiting.")
        return

    # ========== COMPARISON ==========
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Species Comparison Results")
    logger.info("=" * 60)

    # Save comparison CSV
    comparison_csv = output_dir / "action_comparison.csv"
    comparator.save_comparison_csv(comparison_csv)

    # Create visualization with body size
    viz_path = output_dir / "cross_species_comparison.png"
    create_comparison_visualization(comparator, body_size_data, viz_path)

    # Save species configs used
    configs_used_path = output_dir / "species_configs_used.yaml"
    with open(configs_used_path, "w") as f:
        yaml.dump({k: v.get("velocity_thresholds", {}) for k, v in species_configs.items()}, f)
    logger.info(f"Saved configs: {configs_used_path}")

    # Print summary
    logger.info("\n=== Action Distribution ===")
    comparison = comparator.compare_action_distributions()
    for name, data in comparison.items():
        logger.info(f"\n{name} ({data['species']}):")
        for action, stats in data["actions"].items():
            logger.info(f"  {action:12}: {stats['percentage']:5.1f}% ({stats['duration_sec']:.1f}s)")

    # Print body size comparison
    logger.info("\n=== Body Size Comparison ===")
    for species_name, stats in body_size_data.items():
        logger.info(f"{species_name}: {stats['mean']:.1f} ± {stats['std']:.1f} px (range: {stats['min']:.1f}-{stats['max']:.1f})")

    # Generate GIFs if enabled
    if cfg.get("report", {}).get("gifs", True):
        logger.info("\nGenerating action GIFs...")
        for species_name, data in species_results.items():
            gif_generator = ActionGifGenerator(output_dir=output_dir / f"{species_name}_gifs")
            action_names_map = {v: k for k, v in ACTION_TYPES.items()}

            gif_generator.generate_all_action_gifs(
                video_path=data["video_path"],
                keypoints=data["metrics"].trajectory.reshape(-1, 1, 2),  # Simplified
                keypoint_names=["center"],
                action_labels=data["metrics"].action_labels,
                action_names=action_names_map,
                max_segments_per_action=cfg.report.get("gifs_per_action", 2),
                segment_duration_sec=cfg.report.get("gif_duration_sec", 4.0),
                max_frames_per_gif=cfg.report.get("gif_max_frames", 100),
                fps=cfg.report.get("gif_fps", 8),
            )

    # Generate HTML report
    if cfg.get("report", {}).get("html", True):
        logger.info("\nGenerating HTML report...")
        html_generator = HTMLReportGenerator(output_dir=output_dir)

        # Prepare species data for report
        report_species_data = {}
        for species_name, data in species_results.items():
            actions = {}
            for action, stats in data["metrics"].action_summary["actions"].items():
                actions[action] = stats["percentage"]

            report_species_data[species_name.capitalize()] = {
                "body_size": data["body_size"],
                "total_frames": data["metrics"].action_summary["total_frames"],
                "actions": actions,
            }

        report_path = html_generator.generate_cross_species_report(
            species_data=report_species_data,
            comparison_plots={"cross_species_comparison": viz_path},
            title="Cross-Species Behavior Comparison",
        )
        logger.info(f"Saved HTML report: {report_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
