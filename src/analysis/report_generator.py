"""
Keypoint Comparison Report Generator.
Creates visual reports comparing different keypoint configurations on the same data.
Supports GIF animation and HTML report generation.
"""
import os
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import cv2
from io import BytesIO

logger = logging.getLogger(__name__)


def create_gif_from_frames(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float = 10.0,
    loop: int = 0,
) -> Path:
    """
    Create GIF from list of frames using imageio.

    Args:
        frames: List of BGR frames
        output_path: Path to save GIF
        fps: Frames per second
        loop: Number of loops (0 = infinite)

    Returns:
        Path to saved GIF
    """
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not installed. Install with: pip install imageio")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert BGR to RGB
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    # Save GIF
    duration = 1.0 / fps
    imageio.mimsave(str(output_path), rgb_frames, duration=duration, loop=loop)

    logger.info(f"Saved GIF: {output_path} ({len(frames)} frames, {fps} fps)")
    return output_path


# Keypoint preset definitions
KEYPOINT_PRESETS = {
    "full": None,  # All keypoints
    "standard": [
        "nose", "left_ear", "right_ear", "neck", "mouse_center",
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "tail_base", "tail_end"
    ],
    "mars": [
        "nose", "left_ear", "right_ear", "neck",
        "left_hip", "right_hip", "tail_base"
    ],
    "locomotion": [
        "nose", "neck", "mouse_center", "tail_base", "tail_end"
    ],
    "minimal": [
        "nose", "mouse_center", "tail_base"
    ],
}


def generate_keypoint_colors(num_keypoints: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for keypoints."""
    colors = []
    for i in range(num_keypoints):
        hue = int(180 * i / num_keypoints)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors


class KeypointComparisonReport:
    """Generate comparison reports for different keypoint configurations."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        dpi: int = 150,
    ):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def filter_keypoints(
        self,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        preset_name: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Filter keypoints based on preset.

        Args:
            keypoints: Full keypoints array (frames, keypoints, 3)
            all_keypoint_names: All keypoint names from model
            preset_name: Name of preset to use

        Returns:
            Tuple of (filtered_keypoints, filtered_names)
        """
        preset = KEYPOINT_PRESETS.get(preset_name)

        if preset is None:
            return keypoints, all_keypoint_names

        indices = []
        filtered_names = []
        for kp in preset:
            if kp in all_keypoint_names:
                indices.append(all_keypoint_names.index(kp))
                filtered_names.append(kp)

        if not indices:
            return keypoints, all_keypoint_names

        return keypoints[:, indices, :], filtered_names

    def draw_keypoints_on_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        confidence_threshold: float = 0.3,
        show_labels: bool = False,
    ) -> np.ndarray:
        """Draw keypoints on a single frame."""
        frame = frame.copy()
        colors = generate_keypoint_colors(len(keypoint_names))

        # Draw skeleton connections
        connections = self._get_skeleton_connections(keypoint_names)
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints):
                if keypoints[i, 2] > confidence_threshold and keypoints[j, 2] > confidence_threshold:
                    pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                    pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw keypoints
        for idx, (name, color) in enumerate(zip(keypoint_names, colors)):
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf > confidence_threshold:
                    center = (int(x), int(y))
                    cv2.circle(frame, center, 6, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, center, 3, (255, 255, 255), -1, cv2.LINE_AA)

                    if show_labels:
                        cv2.putText(
                            frame, name[:6], (int(x) + 8, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
                        )

        return frame

    def _get_skeleton_connections(self, keypoint_names: List[str]) -> List[Tuple[int, int]]:
        """Get skeleton connections based on available keypoints."""
        connections = []

        # Define potential connections (order matters for fallback)
        potential_connections = [
            # Head connections
            ("nose", "neck"), ("nose", "left_ear"), ("nose", "right_ear"),
            ("nose", "head_midpoint"),
            # Spine connections (with fallbacks for minimal preset)
            ("neck", "mouse_center"), ("mouse_center", "tail_base"),
            ("nose", "mouse_center"),  # Direct connection for minimal (nose-center)
            # Shoulder/hip connections
            ("neck", "left_shoulder"), ("neck", "right_shoulder"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            # Tail connections
            ("tail_base", "tail_end"), ("tail_base", "tail1"),
            ("tail1", "tail2"), ("tail2", "tail3"), ("tail3", "tail4"),
            ("tail4", "tail5"), ("tail5", "tail_end"),
            # Back connections
            ("mouse_center", "mid_backend"), ("mid_backend", "tail_base"),
        ]

        for kp1, kp2 in potential_connections:
            if kp1 in keypoint_names and kp2 in keypoint_names:
                idx1 = keypoint_names.index(kp1)
                idx2 = keypoint_names.index(kp2)
                connections.append((idx1, idx2))

        return connections

    def generate_comparison_report(
        self,
        video_path: Union[str, Path],
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        presets: List[str] = ["full", "standard", "minimal"],
        frame_indices: Optional[List[int]] = None,
        num_frames: int = 3,
        confidence_threshold: float = 0.3,
        video_name: Optional[str] = None,
    ) -> Path:
        """
        Generate a comprehensive comparison report.

        Args:
            video_path: Path to video file
            keypoints: Full keypoints array (frames, keypoints, 3)
            all_keypoint_names: All keypoint names
            presets: List of preset names to compare
            frame_indices: Specific frame indices to use
            num_frames: Number of frames if frame_indices not specified
            confidence_threshold: Confidence threshold for visualization
            video_name: Optional name for the video

        Returns:
            Path to saved report
        """
        video_path = Path(video_path)
        video_name = video_name or video_path.stem

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select frames
        if frame_indices is None:
            frame_indices = np.linspace(0, min(len(keypoints), total_frames) - 1, num_frames, dtype=int).tolist()

        # Create figure
        num_presets = len(presets)
        fig = plt.figure(figsize=(4 * num_presets, 4 * len(frame_indices) + 2))

        # Title
        fig.suptitle(
            f"Keypoint Comparison Report: {video_name}\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=14, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(len(frame_indices) + 1, num_presets, height_ratios=[1] * len(frame_indices) + [0.3])

        # Process each frame and preset
        for row, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for col, preset_name in enumerate(presets):
                ax = fig.add_subplot(gs[row, col])

                # Filter keypoints
                filtered_kp, filtered_names = self.filter_keypoints(
                    keypoints, all_keypoint_names, preset_name
                )

                # Draw keypoints
                frame_with_kp = self.draw_keypoints_on_frame(
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    filtered_kp[frame_idx],
                    filtered_names,
                    confidence_threshold,
                    show_labels=(len(filtered_names) <= 11),
                )
                frame_with_kp = cv2.cvtColor(frame_with_kp, cv2.COLOR_BGR2RGB)

                ax.imshow(frame_with_kp)
                ax.axis("off")

                if row == 0:
                    preset_info = KEYPOINT_PRESETS.get(preset_name)
                    kp_count = len(filtered_names) if preset_info else len(all_keypoint_names)
                    ax.set_title(f"{preset_name.upper()}\n({kp_count} keypoints)", fontsize=11, fontweight="bold")

                if col == 0:
                    ax.set_ylabel(f"Frame {frame_idx}", fontsize=10)

        cap.release()

        # Add summary table
        ax_table = fig.add_subplot(gs[-1, :])
        ax_table.axis("off")

        # Create summary data
        summary_data = []
        for preset_name in presets:
            preset = KEYPOINT_PRESETS.get(preset_name)
            if preset is None:
                kp_list = all_keypoint_names
            else:
                kp_list = [kp for kp in preset if kp in all_keypoint_names]

            summary_data.append([
                preset_name.upper(),
                len(kp_list),
                ", ".join(kp_list[:5]) + ("..." if len(kp_list) > 5 else ""),
                self._get_use_case(preset_name),
            ])

        table = ax_table.table(
            cellText=summary_data,
            colLabels=["Preset", "Keypoints", "Includes", "Use Case"],
            loc="center",
            cellLoc="left",
            colWidths=[0.12, 0.1, 0.4, 0.38],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Style header
        for i in range(len(presets)):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        plt.tight_layout()

        # Save report
        report_path = self.output_dir / f"keypoint_comparison_{video_name}.png"
        fig.savefig(report_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Saved comparison report: {report_path}")
        return report_path

    def _get_use_case(self, preset_name: str) -> str:
        """Get use case description for preset."""
        use_cases = {
            "full": "Detailed pose analysis, grooming, gait",
            "standard": "Open Field Test, general behavior",
            "mars": "Social interaction, multi-animal",
            "locomotion": "Movement/speed analysis",
            "minimal": "Basic tracking, real-time",
        }
        return use_cases.get(preset_name, "General use")

    def generate_trajectory_comparison(
        self,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        presets: List[str] = ["full", "standard", "minimal"],
        reference_keypoint: str = "mouse_center",
        video_name: Optional[str] = None,
    ) -> Path:
        """
        Generate trajectory comparison for different presets.

        Args:
            keypoints: Full keypoints array
            all_keypoint_names: All keypoint names
            presets: Presets to compare
            reference_keypoint: Keypoint to use for trajectory
            video_name: Optional video name

        Returns:
            Path to saved figure
        """
        video_name = video_name or "video"

        fig, axes = plt.subplots(1, len(presets), figsize=(5 * len(presets), 5))
        if len(presets) == 1:
            axes = [axes]

        fig.suptitle(f"Trajectory Comparison: {video_name}", fontsize=14, fontweight="bold")

        for ax, preset_name in zip(axes, presets):
            filtered_kp, filtered_names = self.filter_keypoints(
                keypoints, all_keypoint_names, preset_name
            )

            # Find reference keypoint or use center of all keypoints
            if reference_keypoint in filtered_names:
                ref_idx = filtered_names.index(reference_keypoint)
                trajectory = filtered_kp[:, ref_idx, :2]
            else:
                # Use mean of all keypoints
                valid_mask = filtered_kp[:, :, 2] > 0.3
                trajectory = np.zeros((len(filtered_kp), 2))
                for i in range(len(filtered_kp)):
                    valid = valid_mask[i]
                    if valid.any():
                        trajectory[i] = filtered_kp[i, valid, :2].mean(axis=0)

            # Calculate velocity for coloring
            velocity = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
            velocity = np.concatenate([[0], velocity])

            # Plot
            scatter = ax.scatter(
                trajectory[:, 0], trajectory[:, 1],
                c=velocity, cmap="viridis", s=10, alpha=0.7
            )
            ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start")
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], "rs", markersize=10, label="End")

            preset_kp = KEYPOINT_PRESETS.get(preset_name)
            kp_count = len(filtered_names) if preset_kp else len(all_keypoint_names)
            ax.set_title(f"{preset_name.upper()} ({kp_count} kp)")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.invert_yaxis()
            ax.legend(loc="upper right", fontsize=8)
            ax.set_aspect("equal")

        plt.colorbar(scatter, ax=axes[-1], label="Velocity (px/frame)")
        plt.tight_layout()

        save_path = self.output_dir / f"trajectory_comparison_{video_name}.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved trajectory comparison: {save_path}")
        return save_path

    def generate_full_report(
        self,
        video_path: Union[str, Path],
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        presets: List[str] = ["full", "standard", "minimal"],
        confidence_threshold: float = 0.3,
    ) -> Dict[str, Path]:
        """
        Generate all comparison reports.

        Args:
            video_path: Path to video
            keypoints: Full keypoints array
            all_keypoint_names: All keypoint names
            presets: Presets to compare
            confidence_threshold: Confidence threshold

        Returns:
            Dictionary of report paths
        """
        video_path = Path(video_path)
        video_name = video_path.stem

        reports = {}

        # Keypoint comparison
        reports["keypoint_comparison"] = self.generate_comparison_report(
            video_path=video_path,
            keypoints=keypoints,
            all_keypoint_names=all_keypoint_names,
            presets=presets,
            confidence_threshold=confidence_threshold,
            video_name=video_name,
        )

        # Trajectory comparison
        reports["trajectory_comparison"] = self.generate_trajectory_comparison(
            keypoints=keypoints,
            all_keypoint_names=all_keypoint_names,
            presets=presets,
            video_name=video_name,
        )

        logger.info(f"Generated {len(reports)} reports for {video_name}")
        return reports


class ActionGifGenerator:
    """Generate GIF animations for specific action segments."""

    ACTION_COLORS = {
        "stationary": (52, 152, 219),   # blue
        "walking": (46, 204, 113),      # green
        "running": (231, 76, 60),       # red
        "resting": (52, 152, 219),      # blue
        "grooming": (155, 89, 182),     # purple
        "unknown": (149, 165, 166),     # gray
    }

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_action_segments(
        self,
        action_labels: np.ndarray,
        action_names: Dict[int, str],
        min_duration_frames: int = 10,
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Extract continuous segments for each action.

        Args:
            action_labels: Array of action labels per frame
            action_names: Mapping from label id to action name
            min_duration_frames: Minimum segment duration

        Returns:
            Dictionary mapping action name to list of (start, end) tuples
        """
        segments = {}

        # Find all unique actions
        unique_actions = np.unique(action_labels)

        for action_id in unique_actions:
            action_name = action_names.get(action_id, f"action_{action_id}")
            segments[action_name] = []

            # Find contiguous segments
            mask = action_labels == action_id
            changes = np.diff(mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            # Handle edge cases
            if mask[0]:
                starts = np.concatenate([[0], starts])
            if mask[-1]:
                ends = np.concatenate([ends, [len(mask)]])

            for start, end in zip(starts, ends):
                if end - start >= min_duration_frames:
                    segments[action_name].append((start, end))

        return segments

    def create_action_gif(
        self,
        video_path: Union[str, Path],
        keypoints: np.ndarray,
        keypoint_names: List[str],
        start_frame: int,
        end_frame: int,
        action_name: str,
        output_name: str,
        max_frames: int = 100,
        fps: float = 10.0,
        confidence_threshold: float = 0.3,
        show_skeleton: bool = True,
    ) -> Optional[Path]:
        """
        Create GIF for a specific action segment.

        Args:
            video_path: Path to video
            keypoints: Keypoints array (frames, keypoints, 3)
            keypoint_names: List of keypoint names
            start_frame: Start frame index
            end_frame: End frame index
            action_name: Name of action
            output_name: Output filename (without extension)
            max_frames: Maximum frames in GIF
            fps: GIF frame rate
            confidence_threshold: Confidence threshold
            show_skeleton: Whether to draw skeleton

        Returns:
            Path to saved GIF or None if failed
        """
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adjust frame range
        end_frame = min(end_frame, len(keypoints), total_video_frames)
        segment_length = end_frame - start_frame

        # Subsample if too long
        if segment_length > max_frames:
            step = segment_length // max_frames
            frame_indices = list(range(start_frame, end_frame, step))[:max_frames]
        else:
            frame_indices = list(range(start_frame, end_frame))

        frames = []
        colors = generate_keypoint_colors(len(keypoint_names))
        action_color = self.ACTION_COLORS.get(action_name.lower(), (149, 165, 166))

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Draw keypoints
            if frame_idx < len(keypoints):
                kp = keypoints[frame_idx]

                # Draw skeleton
                if show_skeleton:
                    connections = self._get_skeleton_connections(keypoint_names)
                    for i, j in connections:
                        if i < len(kp) and j < len(kp):
                            if kp[i, 2] > confidence_threshold and kp[j, 2] > confidence_threshold:
                                pt1 = (int(kp[i, 0]), int(kp[i, 1]))
                                pt2 = (int(kp[j, 0]), int(kp[j, 1]))
                                cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw keypoints
                for idx, color in enumerate(colors):
                    if idx < len(kp):
                        x, y, conf = kp[idx]
                        if conf > confidence_threshold:
                            center = (int(x), int(y))
                            cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)

            # Add action label overlay
            cv2.rectangle(frame, (10, 10), (200, 50), action_color, -1)
            cv2.putText(
                frame, action_name.upper(),
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            # Add frame counter
            cv2.putText(
                frame, f"Frame: {frame_idx}",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            frames.append(frame)

        cap.release()

        if not frames:
            logger.warning(f"No frames extracted for {action_name}")
            return None

        # Save GIF
        output_path = self.output_dir / f"{output_name}.gif"
        return create_gif_from_frames(frames, output_path, fps=fps)

    def _get_skeleton_connections(self, keypoint_names: List[str]) -> List[Tuple[int, int]]:
        """Get skeleton connections for visualization."""
        connections = []
        potential = [
            ("nose", "neck"), ("nose", "left_ear"), ("nose", "right_ear"),
            ("neck", "mouse_center"), ("mouse_center", "tail_base"),
            ("nose", "mouse_center"), ("tail_base", "tail_end"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ]

        for kp1, kp2 in potential:
            if kp1 in keypoint_names and kp2 in keypoint_names:
                connections.append((keypoint_names.index(kp1), keypoint_names.index(kp2)))

        return connections

    def generate_all_action_gifs(
        self,
        video_path: Union[str, Path],
        keypoints: np.ndarray,
        keypoint_names: List[str],
        action_labels: np.ndarray,
        action_names: Dict[int, str],
        max_segments_per_action: int = 3,
        segment_duration_sec: float = 4.0,
        max_frames_per_gif: int = 100,
        fps: float = 10.0,
    ) -> Dict[str, List[Path]]:
        """
        Generate GIFs for all detected actions.

        Args:
            video_path: Path to video
            keypoints: Keypoints array
            keypoint_names: Keypoint names
            action_labels: Action labels per frame
            action_names: Mapping of label id to name
            max_segments_per_action: Max GIFs per action type
            segment_duration_sec: Target duration for each GIF segment
            max_frames_per_gif: Maximum frames per GIF (controls length)
            fps: Source video fps for duration calculation

        Returns:
            Dictionary mapping action names to list of GIF paths
        """
        video_path = Path(video_path)
        video_name = video_path.stem

        # Extract segments
        min_frames = int(segment_duration_sec * fps * 0.5)  # At least half duration
        segments = self.extract_action_segments(action_labels, action_names, min_frames)

        gifs = {}

        for action_name, action_segments in segments.items():
            gifs[action_name] = []

            # Select best segments (longest ones)
            sorted_segments = sorted(action_segments, key=lambda x: x[1] - x[0], reverse=True)

            for i, (start, end) in enumerate(sorted_segments[:max_segments_per_action]):
                output_name = f"{video_name}_{action_name}_{i+1}"
                gif_path = self.create_action_gif(
                    video_path=video_path,
                    keypoints=keypoints,
                    keypoint_names=keypoint_names,
                    start_frame=start,
                    end_frame=end,
                    action_name=action_name,
                    output_name=output_name,
                    max_frames=max_frames_per_gif,
                    fps=fps,
                )

                if gif_path:
                    gifs[action_name].append(gif_path)

        total_gifs = sum(len(v) for v in gifs.values())
        logger.info(f"Generated {total_gifs} action GIFs")
        return gifs


class HTMLReportGenerator:
    """Generate comprehensive HTML reports with embedded visualizations."""

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .summary-card .label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .action-section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .action-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .action-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
        }}
        .gif-gallery {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .gif-item {{
            border: 2px solid #eee;
            border-radius: 8px;
            overflow: hidden;
        }}
        .gif-item img {{
            max-width: 300px;
            display: block;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th, .stats-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .stats-table th {{
            background: #3498db;
            color: white;
        }}
        .stats-table tr:hover {{
            background: #f9f9f9;
        }}
        .plot-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 5px;
        }}
        .species-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
        <footer>
            Generated by SuperAnimal Behavior Analysis Pipeline<br>
            {timestamp}
        </footer>
    </div>
</body>
</html>
"""

    ACTION_COLORS_HEX = {
        "stationary": "#3498db",
        "walking": "#2ecc71",
        "running": "#e74c3c",
        "resting": "#3498db",
        "grooming": "#9b59b6",
        "unknown": "#95a5a6",
    }

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 for embedding."""
        if not image_path.exists():
            return ""

        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()

        suffix = image_path.suffix.lower()
        if suffix == ".gif":
            mime = "image/gif"
        elif suffix == ".png":
            mime = "image/png"
        else:
            mime = "image/jpeg"

        return f"data:{mime};base64,{data}"

    def generate_behavior_report(
        self,
        video_name: str,
        species: str,
        metrics: dict,
        action_gifs: Dict[str, List[Path]],
        plot_paths: Dict[str, Path],
        body_size_stats: Optional[Dict] = None,
    ) -> Path:
        """
        Generate HTML report for behavior analysis.

        Args:
            video_name: Name of analyzed video
            species: Species name
            metrics: Behavior metrics dictionary
            action_gifs: Dictionary of action name to GIF paths
            plot_paths: Dictionary of plot name to image paths
            body_size_stats: Optional body size statistics

        Returns:
            Path to saved HTML report
        """
        content_parts = []

        # Title
        content_parts.append(f"<h1>Behavior Analysis Report: {video_name}</h1>")
        content_parts.append(f"<p><strong>Species:</strong> {species}</p>")

        # Summary cards
        summary = metrics.get("behavior_summary", metrics.get("action_summary", {}))
        total_frames = summary.get("total_frames", 0)
        duration = summary.get("total_duration_sec", total_frames / 30.0)

        cards_html = '<div class="summary-grid">'
        cards_html += f'''
            <div class="summary-card">
                <div class="value">{total_frames}</div>
                <div class="label">Total Frames</div>
            </div>
            <div class="summary-card">
                <div class="value">{duration:.1f}s</div>
                <div class="label">Duration</div>
            </div>
        '''

        if body_size_stats:
            cards_html += f'''
                <div class="summary-card">
                    <div class="value">{body_size_stats.get("mean", 0):.1f}px</div>
                    <div class="label">Body Size (mean)</div>
                </div>
                <div class="summary-card">
                    <div class="value">±{body_size_stats.get("std", 0):.1f}px</div>
                    <div class="label">Body Size (std)</div>
                </div>
            '''

        cards_html += '</div>'
        content_parts.append(cards_html)

        # Action breakdown table
        actions = summary.get("actions", summary.get("behaviors", {}))
        if actions:
            content_parts.append("<h2>Action Breakdown</h2>")
            table_html = '<table class="stats-table"><thead><tr>'
            table_html += '<th>Action</th><th>Percentage</th><th>Duration</th><th>Frames</th>'
            table_html += '</tr></thead><tbody>'

            for action_name, stats in actions.items():
                pct = stats.get("percentage", 0)
                dur = stats.get("duration_sec", 0)
                frames = stats.get("frames", int(pct * total_frames / 100))
                color = self.ACTION_COLORS_HEX.get(action_name.lower(), "#95a5a6")

                table_html += f'''
                    <tr>
                        <td><span class="action-badge" style="background:{color}">{action_name}</span></td>
                        <td>{pct:.1f}%</td>
                        <td>{dur:.1f}s</td>
                        <td>{frames}</td>
                    </tr>
                '''

            table_html += '</tbody></table>'
            content_parts.append(table_html)

        # Action GIFs
        if action_gifs:
            content_parts.append("<h2>Action Visualizations</h2>")

            for action_name, gif_paths in action_gifs.items():
                if not gif_paths:
                    continue

                color = self.ACTION_COLORS_HEX.get(action_name.lower(), "#95a5a6")
                section_html = f'''
                    <div class="action-section">
                        <div class="action-header">
                            <span class="action-badge" style="background:{color}">{action_name}</span>
                            <span>({len(gif_paths)} samples)</span>
                        </div>
                        <div class="gif-gallery">
                '''

                for gif_path in gif_paths:
                    b64 = self._image_to_base64(gif_path)
                    if b64:
                        section_html += f'<div class="gif-item"><img src="{b64}" alt="{action_name}"></div>'

                section_html += '</div></div>'
                content_parts.append(section_html)

        # Plots
        if plot_paths:
            content_parts.append("<h2>Analysis Plots</h2>")

            for plot_name, plot_path in plot_paths.items():
                if plot_path and plot_path.exists():
                    b64 = self._image_to_base64(plot_path)
                    if b64:
                        content_parts.append(f'''
                            <div class="plot-container">
                                <h3>{plot_name.replace("_", " ").title()}</h3>
                                <img src="{b64}" alt="{plot_name}">
                            </div>
                        ''')

        # Assemble HTML
        content = "\n".join(content_parts)
        html = self.HTML_TEMPLATE.format(
            title=f"Behavior Report - {video_name}",
            content=content,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        output_path = self.output_dir / f"report_{video_name}.html"
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Saved HTML report: {output_path}")
        return output_path

    def generate_cross_species_report(
        self,
        species_data: Dict[str, Dict],
        comparison_plots: Dict[str, Path],
        title: str = "Cross-Species Comparison",
    ) -> Path:
        """
        Generate HTML report comparing multiple species.

        Args:
            species_data: Dictionary mapping species name to metrics/stats
            comparison_plots: Dictionary of comparison plot paths
            title: Report title

        Returns:
            Path to saved HTML report
        """
        content_parts = []
        content_parts.append(f"<h1>{title}</h1>")

        # Species summary cards
        content_parts.append('<div class="summary-grid">')
        for species_name, data in species_data.items():
            body_size = data.get("body_size", {}).get("mean", 0)
            frames = data.get("total_frames", 0)
            content_parts.append(f'''
                <div class="summary-card">
                    <div class="value">{species_name}</div>
                    <div class="label">Body: {body_size:.1f}px | Frames: {frames}</div>
                </div>
            ''')
        content_parts.append('</div>')

        # Comparison table
        content_parts.append("<h2>Body Size Comparison</h2>")
        table_html = '<table class="stats-table"><thead><tr>'
        table_html += '<th>Species</th><th>Mean Size (px)</th><th>Std Dev</th><th>Min</th><th>Max</th>'
        table_html += '</tr></thead><tbody>'

        for species_name, data in species_data.items():
            bs = data.get("body_size", {})
            table_html += f'''
                <tr>
                    <td><strong>{species_name}</strong></td>
                    <td>{bs.get("mean", 0):.1f}</td>
                    <td>{bs.get("std", 0):.1f}</td>
                    <td>{bs.get("min", 0):.1f}</td>
                    <td>{bs.get("max", 0):.1f}</td>
                </tr>
            '''
        table_html += '</tbody></table>'
        content_parts.append(table_html)

        # Action comparison
        content_parts.append("<h2>Action Distribution Comparison</h2>")
        content_parts.append('<div class="species-comparison">')

        for species_name, data in species_data.items():
            actions = data.get("actions", {})
            section_html = f'<div class="action-section"><h3>{species_name}</h3>'
            section_html += '<table class="stats-table"><thead><tr><th>Action</th><th>%</th></tr></thead><tbody>'

            for action_name, pct in actions.items():
                color = self.ACTION_COLORS_HEX.get(action_name.lower(), "#95a5a6")
                section_html += f'''
                    <tr>
                        <td><span class="action-badge" style="background:{color}">{action_name}</span></td>
                        <td>{pct:.1f}%</td>
                    </tr>
                '''

            section_html += '</tbody></table></div>'
            content_parts.append(section_html)

        content_parts.append('</div>')

        # Comparison plots
        if comparison_plots:
            content_parts.append("<h2>Comparison Visualizations</h2>")
            for plot_name, plot_path in comparison_plots.items():
                if plot_path and plot_path.exists():
                    b64 = self._image_to_base64(plot_path)
                    if b64:
                        content_parts.append(f'''
                            <div class="plot-container">
                                <h3>{plot_name.replace("_", " ").title()}</h3>
                                <img src="{b64}" alt="{plot_name}">
                            </div>
                        ''')

        # Assemble HTML
        content = "\n".join(content_parts)
        html = self.HTML_TEMPLATE.format(
            title=title,
            content=content,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        output_path = self.output_dir / "cross_species_report.html"
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Saved cross-species report: {output_path}")
        return output_path
