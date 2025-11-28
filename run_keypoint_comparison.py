#!/usr/bin/env python3
"""
Keypoint Preset Comparison Pipeline

Compare different keypoint configurations (Full, Standard, Minimal) on the same video.
Shows how reducing keypoints affects pose visualization and tracking.

Usage:
    python run_keypoint_comparison.py                           # Default video
    python run_keypoint_comparison.py data.video_path=/path/to/video.mp4
    python run_keypoint_comparison.py data.video.max_frames=100
"""
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import VideoDownloader
from src.models.predictor import SuperAnimalPredictor
from src.analysis.report_generator import (
    KeypointComparisonReport,
    KEYPOINT_PRESETS,
    create_gif_from_frames,
    generate_keypoint_colors,
)
from src.utils.helpers import setup_logging, get_device

logger = logging.getLogger(__name__)


def create_preset_comparison_gif(
    video_path: Path,
    keypoints: np.ndarray,
    all_keypoint_names: list,
    presets: list,
    output_path: Path,
    max_frames: int = 100,
    fps: float = 8.0,
    confidence_threshold: float = 0.3,
) -> Path:
    """
    Create side-by-side GIF comparing different keypoint presets.

    Args:
        video_path: Path to video
        keypoints: Full keypoints array (frames, keypoints, 3)
        all_keypoint_names: All keypoint names
        presets: List of preset names to compare
        output_path: Output GIF path
        max_frames: Maximum frames in GIF
        fps: GIF frame rate
        confidence_threshold: Confidence threshold for visualization

    Returns:
        Path to saved GIF
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(keypoints))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Subsample frames if needed
    if total_frames > max_frames:
        step = total_frames // max_frames
        frame_indices = list(range(0, total_frames, step))[:max_frames]
    else:
        frame_indices = list(range(total_frames))

    # Prepare preset data
    preset_data = []
    for preset_name in presets:
        preset_kps = KEYPOINT_PRESETS.get(preset_name)
        if preset_kps is None:
            # Full preset - use all keypoints
            indices = list(range(len(all_keypoint_names)))
            names = all_keypoint_names
        else:
            indices = []
            names = []
            for kp in preset_kps:
                if kp in all_keypoint_names:
                    indices.append(all_keypoint_names.index(kp))
                    names.append(kp)

        preset_data.append({
            "name": preset_name,
            "indices": indices,
            "names": names,
            "colors": generate_keypoint_colors(len(names)),
        })

    # Create frames
    combined_frames = []
    panel_width = frame_width // 2  # Smaller panels for side-by-side
    panel_height = frame_height // 2

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Create combined frame (2x2 grid or 1x3 depending on preset count)
        if len(presets) <= 3:
            combined = np.zeros((panel_height, panel_width * len(presets), 3), dtype=np.uint8)
        else:
            rows = (len(presets) + 1) // 2
            combined = np.zeros((panel_height * rows, panel_width * 2, 3), dtype=np.uint8)

        for i, preset in enumerate(preset_data):
            # Resize frame for panel
            panel = cv2.resize(frame.copy(), (panel_width, panel_height))

            # Draw keypoints for this preset
            kp_frame = keypoints[frame_idx]

            # Draw skeleton connections
            connections = _get_skeleton_connections(preset["names"])
            for idx1, idx2 in connections:
                if idx1 < len(preset["indices"]) and idx2 < len(preset["indices"]):
                    real_idx1 = preset["indices"][idx1]
                    real_idx2 = preset["indices"][idx2]
                    if (kp_frame[real_idx1, 2] > confidence_threshold and
                        kp_frame[real_idx2, 2] > confidence_threshold):
                        # Scale coordinates to panel size
                        pt1 = (int(kp_frame[real_idx1, 0] * panel_width / frame_width),
                               int(kp_frame[real_idx1, 1] * panel_height / frame_height))
                        pt2 = (int(kp_frame[real_idx2, 0] * panel_width / frame_width),
                               int(kp_frame[real_idx2, 1] * panel_height / frame_height))
                        cv2.line(panel, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw keypoints
            for j, kp_idx in enumerate(preset["indices"]):
                x, y, conf = kp_frame[kp_idx]
                if conf > confidence_threshold:
                    # Scale to panel size
                    px = int(x * panel_width / frame_width)
                    py = int(y * panel_height / frame_height)
                    color = preset["colors"][j] if j < len(preset["colors"]) else (0, 255, 0)
                    cv2.circle(panel, (px, py), 4, color, -1, cv2.LINE_AA)

            # Add label
            label = f"{preset['name'].upper()} ({len(preset['indices'])} kp)"
            cv2.rectangle(panel, (5, 5), (len(label) * 10 + 15, 30), (0, 0, 0), -1)
            cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Place in combined frame
            if len(presets) <= 3:
                combined[:, i * panel_width:(i + 1) * panel_width] = panel
            else:
                row = i // 2
                col = i % 2
                combined[row * panel_height:(row + 1) * panel_height,
                         col * panel_width:(col + 1) * panel_width] = panel

        combined_frames.append(combined)

    cap.release()

    if combined_frames:
        return create_gif_from_frames(combined_frames, output_path, fps=fps)
    return None


def _get_skeleton_connections(keypoint_names: list) -> list:
    """Get skeleton connections based on available keypoints."""
    connections = []
    potential = [
        ("nose", "neck"), ("nose", "left_ear"), ("nose", "right_ear"),
        ("neck", "mouse_center"), ("mouse_center", "tail_base"),
        ("nose", "mouse_center"), ("tail_base", "tail_end"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ]

    for kp1, kp2 in potential:
        if kp1 in keypoint_names and kp2 in keypoint_names:
            connections.append((keypoint_names.index(kp1), keypoint_names.index(kp2)))

    return connections


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run keypoint preset comparison pipeline."""
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("Keypoint Preset Comparison")
    logger.info("=" * 60)
    logger.info("Comparing: Full vs Standard vs Minimal keypoint configurations")

    original_cwd = hydra.utils.get_original_cwd()
    output_dir = Path(original_cwd) / "outputs" / "keypoint_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video
    if hasattr(cfg.data, 'video_path') and cfg.data.video_path:
        video_path = Path(cfg.data.video_path)
    else:
        data_dir = Path(original_cwd) / cfg.data.data_dir
        downloader = VideoDownloader(str(data_dir))
        video_path = downloader.download_sample("mouse_topview")

    if not video_path or not video_path.exists():
        logger.error("Video not found")
        return

    logger.info(f"Video: {video_path}")

    # Run inference (always use full keypoints for comparison)
    device = get_device(cfg.device)
    predictor = SuperAnimalPredictor(
        model_type="topviewmouse",
        model_name=cfg.model.model_name,
        video_adapt=cfg.model.video_adapt,
        device=device,
        use_keypoints=None,  # Always use full for comparison
    )

    logger.info("Running keypoint prediction (full keypoints)...")
    results = predictor.predict_video(
        video_path=video_path,
        output_dir=output_dir / "predictions",
        max_frames=cfg.data.video.max_frames,
    )

    if results["keypoints"] is None:
        logger.error("No keypoints detected")
        return

    keypoints = results["keypoints"]
    keypoint_names = predictor.get_keypoint_names()
    logger.info(f"Detected {len(keypoint_names)} keypoints, {len(keypoints)} frames")

    presets_to_compare = ["full", "standard", "minimal"]

    # ========== Generate Comparison GIF ==========
    logger.info("\nGenerating comparison GIF...")
    gif_path = output_dir / f"keypoint_comparison_{video_path.stem}.gif"

    comparison_gif = create_preset_comparison_gif(
        video_path=video_path,
        keypoints=keypoints,
        all_keypoint_names=keypoint_names,
        presets=presets_to_compare,
        output_path=gif_path,
        max_frames=cfg.report.get("gif_max_frames", 100),
        fps=cfg.report.get("gif_fps", 8),
    )

    if comparison_gif:
        logger.info(f"Saved comparison GIF: {comparison_gif}")

    # ========== Generate Static Reports ==========
    logger.info("\nGenerating static comparison reports...")
    report_gen = KeypointComparisonReport(output_dir=output_dir, dpi=150)

    # Visual comparison report (static frames)
    comparison_report = report_gen.generate_comparison_report(
        video_path=video_path,
        keypoints=keypoints,
        all_keypoint_names=keypoint_names,
        presets=presets_to_compare,
        num_frames=3,
        video_name=video_path.stem,
    )
    logger.info(f"Saved: {comparison_report}")

    # Trajectory comparison
    trajectory_report = report_gen.generate_trajectory_comparison(
        keypoints=keypoints,
        all_keypoint_names=keypoint_names,
        presets=presets_to_compare,
        video_name=video_path.stem,
    )
    logger.info(f"Saved: {trajectory_report}")

    # ========== Print Summary ==========
    logger.info("\n" + "=" * 60)
    logger.info("Keypoint Preset Summary:")
    logger.info("-" * 60)
    for preset_name in presets_to_compare:
        preset = KEYPOINT_PRESETS.get(preset_name)
        if preset is None:
            count = len(keypoint_names)
            kps = keypoint_names[:5]
        else:
            available = [kp for kp in preset if kp in keypoint_names]
            count = len(available)
            kps = available[:5]
        logger.info(f"  {preset_name.upper():12} : {count:2} keypoints - {', '.join(kps)}...")

    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
