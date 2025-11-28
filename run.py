#!/usr/bin/env python3
"""
SuperAnimal Behavior Analysis PoC - Main Entry Point

This script runs the complete pipeline:
1. Downloads sample videos (if needed) or processes local video/images
2. Runs SuperAnimal keypoint prediction
3. Performs behavior analysis with body size estimation
4. Generates visualizations, GIFs, and HTML reports

Usage:
    # Video mode (default)
    python run.py                           # Default: topviewmouse
    python run.py model=quadruped           # Use quadruped model
    python run.py data.video.max_frames=200

    # Custom video/image
    python run.py input=/path/to/video.mp4
    python run.py input=/path/to/images/    # Directory of images

    # Generate HTML report with action GIFs
    python run.py report.html=true report.gifs=true
"""
import os
import sys
import re
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import VideoDownloader
from src.models.predictor import SuperAnimalPredictor
from src.analysis.behavior import BehaviorAnalyzer, estimate_body_size
from src.analysis.visualizer import Visualizer
from src.analysis.report_generator import ActionGifGenerator, HTMLReportGenerator
from src.utils.helpers import setup_logging, get_device

logger = logging.getLogger(__name__)


def get_model_type_from_config(cfg: DictConfig) -> str:
    """Extract model type from config."""
    model_name = cfg.model.name
    if "topviewmouse" in model_name:
        return "topviewmouse"
    elif "quadruped" in model_name:
        return "quadruped"
    return "topviewmouse"


def get_images_from_directory(image_dir: Path) -> list:
    """
    Get sorted list of images from directory.
    Sorts by numeric prefix in filename for frame ordering.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = []

    for f in image_dir.iterdir():
        if f.suffix.lower() in image_extensions:
            images.append(f)

    # Sort by numeric value in filename
    def extract_number(path):
        numbers = re.findall(r'\d+', path.stem)
        return int(numbers[-1]) if numbers else 0

    images.sort(key=extract_number)
    return images


def determine_input_type(input_path: Path) -> str:
    """Determine if input is video, image directory, or single image."""
    if input_path.is_dir():
        return "image_directory"
    elif input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    elif input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        return "single_image"
    return "unknown"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main pipeline execution."""
    # Setup logging
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("SuperAnimal Behavior Analysis PoC")
    logger.info("=" * 60)

    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Get working directory (Hydra changes cwd)
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = Path.cwd()  # Hydra output dir (now: outputs/experiments/{experiment_id})

    logger.info(f"Working directory: {original_cwd}")
    logger.info(f"Experiment ID: {cfg.experiment_id}")
    logger.info(f"Output directory: {output_dir}")

    # Determine model type
    model_type = get_model_type_from_config(cfg)
    logger.info(f"Model type: {model_type}")

    # Step 1: Get input (video or images)
    logger.info("\n[Step 1] Preparing input data...")
    data_dir = Path(original_cwd) / cfg.data.data_dir
    downloader = VideoDownloader(str(data_dir))

    input_type = "video"
    image_paths = None

    # Check for custom input
    if cfg.get("input") and cfg.input:
        input_path = Path(original_cwd) / cfg.input
        if not input_path.exists():
            logger.error(f"Input path not found: {input_path}")
            return

        input_type = determine_input_type(input_path)
        logger.info(f"Custom input: {input_path} (type: {input_type})")

        if input_type == "video":
            video_path = input_path
        elif input_type == "image_directory":
            image_paths = get_images_from_directory(input_path)
            if not image_paths:
                logger.error(f"No images found in directory: {input_path}")
                return
            logger.info(f"Found {len(image_paths)} images")
            video_path = None  # Will use image inference
        elif input_type == "single_image":
            image_paths = [input_path]
            video_path = None
        else:
            logger.error(f"Unknown input type: {input_path}")
            return
    else:
        # Download appropriate sample based on model type
        if model_type == "topviewmouse":
            video_path = downloader.download_sample("mouse_topview")
        else:
            video_path = downloader.download_sample("dog_walking")

        if video_path is None:
            logger.error("Failed to download sample video. Exiting.")
            return

    # Get video/input info
    if video_path:
        video_info = downloader.get_video_info(video_path)
        logger.info(f"Video info: {video_info}")
    else:
        # Estimate info from images
        import cv2
        first_img = cv2.imread(str(image_paths[0]))
        video_info = {
            "fps": 30.0,  # Assumed FPS for images
            "frame_count": len(image_paths),
            "width": first_img.shape[1],
            "height": first_img.shape[0],
        }
        logger.info(f"Image sequence info: {len(image_paths)} images, {video_info['width']}x{video_info['height']}")

    # Step 2: Run keypoint prediction
    logger.info("\n[Step 2] Running SuperAnimal keypoint prediction...")
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    predictor = SuperAnimalPredictor(
        model_type=model_type,
        model_name=cfg.model.model_name,
        video_adapt=cfg.model.video_adapt,
        scale_list=cfg.model.scale_list,
        device=device,
    )

    # Run inference (video or images)
    if video_path:
        results = predictor.predict_video(
            video_path=video_path,
            output_dir=output_dir / "predictions",
            max_frames=cfg.data.video.max_frames,
        )
        source_path = video_path
    else:
        results = predictor.predict_images(
            image_paths=image_paths,
            output_dir=output_dir / "predictions",
        )
        source_path = image_paths[0].parent  # Use directory as source

    logger.info(f"Prediction complete. Processed {results['metadata'].get('num_frames', 'N/A')} frames")
    if results["metadata"].get("mock_data"):
        logger.warning("Note: Using mock data for demonstration. Install DeepLabCut for real predictions.")

    # Save keypoint coordinates to CSV (default export)
    keypoint_names = predictor.get_keypoint_names()
    if results["keypoints"] is not None:
        import pandas as pd
        keypoints = results["keypoints"]  # Shape: (frames, keypoints, 3)

        # Build CSV data
        csv_data = {"frame": list(range(len(keypoints)))}
        for i, kp_name in enumerate(keypoint_names):
            csv_data[f"{kp_name}_x"] = keypoints[:, i, 0]
            csv_data[f"{kp_name}_y"] = keypoints[:, i, 1]
            csv_data[f"{kp_name}_conf"] = keypoints[:, i, 2]

        coords_csv_path = output_dir / "predictions" / "keypoints_coordinates.csv"
        pd.DataFrame(csv_data).to_csv(coords_csv_path, index=False)
        logger.info(f"Saved keypoint coordinates CSV: {coords_csv_path}")

    # Step 3: Behavior analysis
    logger.info("\n[Step 3] Analyzing behavior...")

    analyzer = BehaviorAnalyzer(
        model_type=model_type,
        fps=video_info["fps"],
        smoothing_window=cfg.behavior.analysis.motion.velocity_smoothing,
    )

    if results["keypoints"] is not None:
        metrics = analyzer.analyze(
            keypoints=results["keypoints"],
            keypoint_names=keypoint_names,
        )

        # Print summary
        logger.info("\nBehavior Summary:")
        for key, value in metrics.behavior_summary.items():
            if key != "behaviors":
                logger.info(f"  {key}: {value}")

        logger.info("\nBehavior Breakdown:")
        for behavior, stats in metrics.behavior_summary.get("behaviors", {}).items():
            logger.info(f"  {behavior}: {stats['percentage']:.1f}% ({stats['duration_sec']:.1f}s)")

        # Save metrics to CSV
        metrics_df = analyzer.to_dataframe(metrics)
        metrics_csv = output_dir / "behavior_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"Saved metrics to: {metrics_csv}")

        # Step 4: Visualization
        logger.info("\n[Step 4] Generating visualizations...")
        visualizer = Visualizer(output_dir=str(output_dir / "plots"))

        # Generate plots
        visualizer.plot_trajectory(
            trajectory=metrics.trajectory,
            velocity=metrics.velocity,
            title=f"Trajectory - {video_path.stem}",
        )

        visualizer.plot_velocity_profile(
            velocity=metrics.velocity,
            fps=video_info["fps"],
            title=f"Velocity Profile - {video_path.stem}",
        )

        behavior_names = {v: k for k, v in analyzer.BEHAVIOR_TYPES.items()}
        visualizer.plot_behavior_timeline(
            behavior_labels=metrics.behavior_labels,
            behavior_names=behavior_names,
            fps=video_info["fps"],
            title=f"Behavior Timeline - {video_path.stem}",
        )

        visualizer.plot_behavior_summary(
            behavior_summary=metrics.behavior_summary,
            title=f"Behavior Summary - {video_path.stem}",
        )

        # Create comprehensive report
        visualizer.create_analysis_report(
            video_name=video_path.stem,
            metrics=metrics,
            analyzer=analyzer,
        )

        # Step 5: Keypoint overlay visualizations
        logger.info("\n[Step 5] Creating keypoint overlay visualizations...")

        # Save sample frames with keypoint overlays
        keypoint_frames = visualizer.save_keypoint_frames(
            video_path=video_path,
            keypoints=results["keypoints"],
            keypoint_names=keypoint_names,
            output_dir=output_dir / "keypoint_frames",
            num_samples=10,
            confidence_threshold=0.3,
            show_labels=True,
            show_skeleton=True,
        )
        logger.info(f"Saved {len(keypoint_frames)} keypoint frames")

        # Create comparison grid
        comparison_grid = visualizer.create_comparison_grid(
            video_path=video_path,
            keypoints=results["keypoints"],
            keypoint_names=keypoint_names,
            output_path=output_dir / "plots" / "keypoint_comparison_grid.png",
            num_frames=9,
            confidence_threshold=0.3,
        )
        logger.info(f"Saved comparison grid: {comparison_grid}")

        # Create keypoint overlay video (if video is short enough)
        if video_path and results["metadata"].get("num_frames", 0) <= 500:
            overlay_video = visualizer.create_keypoint_overlay_video(
                video_path=video_path,
                keypoints=results["keypoints"],
                keypoint_names=keypoint_names,
                output_path=output_dir / "videos" / f"{video_path.stem}_keypoints.mp4",
                confidence_threshold=0.3,
                show_labels=False,
                show_skeleton=True,
            )
            logger.info(f"Saved keypoint overlay video: {overlay_video}")
        elif video_path:
            logger.info("Skipping overlay video (too many frames). Use max_frames to limit.")

        # Step 6: Body size estimation
        logger.info("\n[Step 6] Estimating body size...")
        body_size_stats = estimate_body_size(
            keypoints=results["keypoints"],
            keypoint_names=keypoint_names,
            model_type=model_type,
        )
        logger.info(f"Body size: {body_size_stats['mean']:.1f} ± {body_size_stats['std']:.1f} pixels")
        logger.info(f"  Range: {body_size_stats['min']:.1f} - {body_size_stats['max']:.1f} pixels")

        # Step 7: Generate action GIFs and HTML report
        if cfg.get("report", {}).get("gifs", False) and video_path:
            logger.info("\n[Step 7] Generating action GIFs...")
            gif_generator = ActionGifGenerator(output_dir=output_dir / "gifs")

            action_names = {v: k for k, v in analyzer.BEHAVIOR_TYPES.items()}
            action_gifs = gif_generator.generate_all_action_gifs(
                video_path=video_path,
                keypoints=results["keypoints"],
                keypoint_names=keypoint_names,
                action_labels=metrics.behavior_labels,
                action_names=action_names,
                max_segments_per_action=cfg.report.get("gifs_per_action", 2),
                segment_duration_sec=cfg.report.get("gif_duration_sec", 4.0),
                max_frames_per_gif=cfg.report.get("gif_max_frames", 100),
                fps=cfg.report.get("gif_fps", 8),
            )

            total_gifs = sum(len(v) for v in action_gifs.values())
            logger.info(f"Generated {total_gifs} action GIFs")
        else:
            action_gifs = {}

        # Generate HTML report
        if cfg.get("report", {}).get("html", False):
            logger.info("\n[Step 8] Generating HTML report...")
            html_generator = HTMLReportGenerator(output_dir=output_dir)

            # Collect plot paths
            plot_paths = {
                "trajectory": output_dir / "plots" / "trajectory.png",
                "velocity_profile": output_dir / "plots" / "velocity_profile.png",
                "behavior_timeline": output_dir / "plots" / "behavior_timeline.png",
                "behavior_summary": output_dir / "plots" / "behavior_summary.png",
                "analysis_report": output_dir / "plots" / "analysis_report.png",
            }

            source_name = video_path.stem if video_path else "image_sequence"
            species = "mouse" if model_type == "topviewmouse" else "quadruped"

            report_path = html_generator.generate_behavior_report(
                video_name=source_name,
                species=species,
                metrics={"behavior_summary": metrics.behavior_summary},
                action_gifs=action_gifs,
                plot_paths=plot_paths,
                body_size_stats=body_size_stats,
            )
            logger.info(f"Saved HTML report: {report_path}")

    else:
        logger.warning("No keypoints available for analysis")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
