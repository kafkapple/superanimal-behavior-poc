"""
Visualization module for keypoint predictions and behavior analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Optional cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available - some visualization features will be disabled")


# Keypoint colors (HSV-based rainbow for distinguishing points)
def generate_keypoint_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for keypoints."""
    colors = []
    for i in range(n):
        if CV2_AVAILABLE:
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
            colors.append(tuple(int(c) for c in color))
        else:
            # Fallback: generate colors using matplotlib colormap
            import matplotlib.cm as cm
            cmap = cm.get_cmap('hsv')
            rgba = cmap(i / n)
            colors.append(tuple(int(c * 255) for c in rgba[:3]))
    return colors


class Visualizer:
    """Visualize keypoint predictions and behavior analysis results."""

    # Color scheme for behaviors
    BEHAVIOR_COLORS = {
        "resting": "#3498db",  # blue
        "walking": "#2ecc71",  # green
        "running": "#e74c3c",  # red
        "grooming": "#9b59b6",  # purple
        "rearing": "#f39c12",  # orange
        "unknown": "#95a5a6",  # gray
    }

    def __init__(self, output_dir: str = "outputs", dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_trajectory(
        self,
        trajectory: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        title: str = "Animal Trajectory",
        save_name: str = "trajectory.png",
    ) -> Path:
        """
        Plot the movement trajectory.

        Args:
            trajectory: (num_frames, 2) array of x, y positions
            velocity: Optional velocity for color coding
            title: Plot title
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        if velocity is not None:
            # Color by velocity
            points = trajectory.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            from matplotlib.collections import LineCollection
            norm = plt.Normalize(velocity.min(), velocity.max())
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(velocity[:-1])
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax, label="Velocity (px/frame)")
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", alpha=0.7, linewidth=1)

        # Mark start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=100, marker="o", label="Start", zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=100, marker="s", label="End", zorder=5)

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title(title)
        ax.legend()
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Image coordinates

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved trajectory plot: {save_path}")
        return save_path

    def plot_velocity_profile(
        self,
        velocity: np.ndarray,
        fps: float = 30.0,
        title: str = "Velocity Profile",
        save_name: str = "velocity_profile.png",
    ) -> Path:
        """Plot velocity over time."""
        fig, ax = plt.subplots(figsize=(12, 4))

        time = np.arange(len(velocity)) / fps
        ax.plot(time, velocity, "b-", linewidth=0.8)
        ax.fill_between(time, 0, velocity, alpha=0.3)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Velocity (px/frame)")
        ax.set_title(title)
        ax.set_xlim(0, time[-1])
        ax.set_ylim(0, None)

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved velocity plot: {save_path}")
        return save_path

    def plot_behavior_timeline(
        self,
        behavior_labels: np.ndarray,
        behavior_names: Dict[int, str],
        fps: float = 30.0,
        title: str = "Behavior Timeline",
        save_name: str = "behavior_timeline.png",
    ) -> Path:
        """Plot behavior classification over time."""
        fig, ax = plt.subplots(figsize=(14, 3))

        time = np.arange(len(behavior_labels)) / fps

        # Create color array
        colors = []
        for label in behavior_labels:
            name = behavior_names.get(label, "unknown")
            colors.append(self.BEHAVIOR_COLORS.get(name, "#95a5a6"))

        # Plot as colored bars
        for i in range(len(behavior_labels) - 1):
            ax.axvspan(time[i], time[i + 1], facecolor=colors[i], alpha=0.8)

        # Legend
        patches = []
        seen = set()
        for label in behavior_labels:
            name = behavior_names.get(label, "unknown")
            if name not in seen:
                color = self.BEHAVIOR_COLORS.get(name, "#95a5a6")
                patches.append(mpatches.Patch(color=color, label=name.capitalize()))
                seen.add(name)

        ax.legend(handles=patches, loc="upper right", ncol=len(patches))

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Behavior")
        ax.set_title(title)
        ax.set_xlim(0, time[-1])
        ax.set_yticks([])

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved behavior timeline: {save_path}")
        return save_path

    def plot_behavior_summary(
        self,
        behavior_summary: Dict,
        title: str = "Behavior Summary",
        save_name: str = "behavior_summary.png",
    ) -> Path:
        """Plot pie chart of behavior distribution."""
        behaviors = behavior_summary.get("behaviors", {})
        if not behaviors:
            logger.warning("No behavior data to plot")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        labels = []
        sizes = []
        colors = []
        for name, data in behaviors.items():
            if name != "unknown" and data["percentage"] > 0:
                labels.append(name.capitalize())
                sizes.append(data["percentage"])
                colors.append(self.BEHAVIOR_COLORS.get(name, "#95a5a6"))

        if sizes:
            axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            axes[0].set_title("Behavior Distribution")

        # Bar chart of durations
        names = []
        durations = []
        bar_colors = []
        for name, data in behaviors.items():
            if name != "unknown":
                names.append(name.capitalize())
                durations.append(data["duration_sec"])
                bar_colors.append(self.BEHAVIOR_COLORS.get(name, "#95a5a6"))

        if names:
            axes[1].barh(names, durations, color=bar_colors)
            axes[1].set_xlabel("Duration (seconds)")
            axes[1].set_title("Behavior Durations")

        fig.suptitle(title)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved behavior summary: {save_path}")
        return save_path

    def plot_keypoints_on_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        confidence_threshold: float = 0.5,
        title: str = "Keypoint Detection",
        save_name: str = "keypoints_frame.png",
    ) -> Path:
        """Plot keypoints on a single frame."""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(frame)

        # Plot keypoints
        for i, name in enumerate(keypoint_names):
            x, y, conf = keypoints[i]
            if conf > confidence_threshold:
                ax.scatter(x, y, s=50, c="red", marker="o", zorder=3)
                ax.annotate(name, (x, y), fontsize=6, xytext=(5, 5),
                           textcoords="offset points", color="yellow")

        # Draw skeleton connections (model-specific)
        connections = self._get_skeleton_connections(keypoint_names)
        for i, j in connections:
            if (keypoints[i, 2] > confidence_threshold and
                keypoints[j, 2] > confidence_threshold):
                ax.plot([keypoints[i, 0], keypoints[j, 0]],
                       [keypoints[i, 1], keypoints[j, 1]],
                       "c-", linewidth=1, alpha=0.7)

        ax.set_title(title)
        ax.axis("off")

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved keypoints visualization: {save_path}")
        return save_path

    def _get_skeleton_connections(self, keypoint_names: List[str]) -> List[Tuple[int, int]]:
        """Get skeleton connection pairs based on keypoint names."""
        # Define all potential connections (works for any subset of keypoints)
        # Order matters: earlier connections take priority
        potential_connections = [
            # Head connections
            ("nose", "neck"), ("nose", "left_ear"), ("nose", "right_ear"),
            ("nose", "head_midpoint"), ("left_ear", "left_ear_tip"), ("right_ear", "right_ear_tip"),
            # Spine connections (including fallback for minimal: nose -> center)
            ("neck", "mid_back"), ("mid_back", "mouse_center"),
            ("neck", "mouse_center"),  # Fallback when mid_back missing
            ("nose", "mouse_center"),  # Direct connection for minimal preset
            ("mouse_center", "mid_backend"), ("mid_backend", "tail_base"),
            ("mouse_center", "tail_base"),  # Fallback when mid_backend missing
            # Shoulder/hip connections
            ("neck", "left_shoulder"), ("neck", "right_shoulder"),
            ("left_shoulder", "left_midside"), ("right_shoulder", "right_midside"),
            ("left_midside", "left_hip"), ("right_midside", "right_hip"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),  # Fallback
            ("left_hip", "right_hip"),
            # Tail connections (DLC 3.0 naming)
            ("tail_base", "tail1"), ("tail1", "tail2"), ("tail2", "tail3"),
            ("tail3", "tail4"), ("tail4", "tail5"), ("tail5", "tail_end"),
            ("tail_base", "tail_end"),  # Fallback for minimal tail
            # Legacy tail naming
            ("tail_base", "tail_mid"), ("tail_mid", "tail_tip"),
            # Quadruped skeleton
            ("nose", "neck_base"), ("neck_base", "back_base"), ("back_base", "back_middle"),
            ("back_middle", "back_end"), ("back_end", "tail_base"), ("tail_base", "tail_middle"),
            ("tail_middle", "tail_end"), ("nose", "left_eye"), ("nose", "right_eye"),
        ]

        # Convert to indices (only for keypoints that exist)
        idx_connections = []
        for a, b in potential_connections:
            if a in keypoint_names and b in keypoint_names:
                idx_a = keypoint_names.index(a)
                idx_b = keypoint_names.index(b)
                idx_connections.append((idx_a, idx_b))

        return idx_connections

    def create_analysis_report(
        self,
        video_name: str,
        metrics,
        analyzer,
        save_name: str = "analysis_report.png",
    ) -> Path:
        """Create a comprehensive analysis report figure."""
        fig = plt.figure(figsize=(16, 12))

        # Layout: 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        if metrics.velocity is not None:
            points = metrics.trajectory.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            from matplotlib.collections import LineCollection
            norm = plt.Normalize(metrics.velocity.min(), metrics.velocity.max())
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(metrics.velocity[:-1])
            lc.set_linewidth(2)
            ax1.add_collection(lc)
            ax1.autoscale()
        ax1.scatter(metrics.trajectory[0, 0], metrics.trajectory[0, 1], c="green", s=100, marker="o", label="Start")
        ax1.scatter(metrics.trajectory[-1, 0], metrics.trajectory[-1, 1], c="red", s=100, marker="s", label="End")
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")
        ax1.set_title("Movement Trajectory")
        ax1.legend()
        ax1.invert_yaxis()

        # 2. Velocity profile
        ax2 = fig.add_subplot(gs[0, 1])
        time = np.arange(len(metrics.velocity)) / analyzer.fps
        ax2.plot(time, metrics.velocity, "b-", linewidth=0.8)
        ax2.fill_between(time, 0, metrics.velocity, alpha=0.3)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Velocity (px/frame)")
        ax2.set_title("Velocity Profile")

        # 3. Behavior timeline
        ax3 = fig.add_subplot(gs[1, 0])
        behavior_names = {v: k for k, v in analyzer.BEHAVIOR_TYPES.items()}
        for i in range(len(metrics.behavior_labels) - 1):
            name = behavior_names.get(metrics.behavior_labels[i], "unknown")
            color = self.BEHAVIOR_COLORS.get(name, "#95a5a6")
            ax3.axvspan(time[i], time[i + 1], facecolor=color, alpha=0.8)
        ax3.set_xlabel("Time (seconds)")
        ax3.set_title("Behavior Timeline")
        ax3.set_yticks([])

        # 4. Summary stats
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        summary = metrics.behavior_summary
        text = f"""Analysis Summary for: {video_name}

Total Duration: {summary['total_duration_sec']:.1f} seconds
Total Frames: {summary['total_frames']}
Distance Traveled: {summary['distance_traveled_px']:.1f} pixels

Velocity Statistics:
  Mean: {summary['mean_velocity']:.2f} px/frame
  Max: {summary['max_velocity']:.2f} px/frame
  Std: {summary['std_velocity']:.2f} px/frame

Behavior Breakdown:
"""
        for name, data in summary.get("behaviors", {}).items():
            if name != "unknown":
                text += f"  {name.capitalize()}: {data['percentage']:.1f}% ({data['duration_sec']:.1f}s)\n"

        ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        fig.suptitle(f"Behavior Analysis Report: {video_name}", fontsize=14, fontweight="bold")

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved analysis report: {save_path}")
        return save_path

    def create_keypoint_overlay_video(
        self,
        video_path: Union[str, Path],
        keypoints: np.ndarray,
        keypoint_names: List[str],
        output_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.3,
        show_labels: bool = True,
        show_skeleton: bool = True,
        fps: Optional[float] = None,
    ) -> Path:
        """
        Create a video with keypoint overlays on original RGB frames.

        Args:
            video_path: Path to input video
            keypoints: Array of shape (num_frames, num_keypoints, 3) [x, y, conf]
            keypoint_names: List of keypoint names
            output_path: Output video path (default: input_keypoints.mp4)
            confidence_threshold: Minimum confidence to draw keypoint
            show_labels: Show keypoint labels
            show_skeleton: Draw skeleton connections
            fps: Output FPS (default: same as input)

        Returns:
            Path to output video
        """
        video_path = Path(video_path)
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_keypoints.mp4"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_fps = fps if fps else input_fps

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

        # Generate colors for keypoints
        num_keypoints = len(keypoint_names)
        colors = generate_keypoint_colors(num_keypoints)

        # Get skeleton connections
        connections = self._get_skeleton_connections(keypoint_names)

        logger.info(f"Creating keypoint overlay video: {output_path}")
        logger.info(f"Input: {total_frames} frames, Output: {min(len(keypoints), total_frames)} frames")

        frame_idx = 0
        while cap.isOpened() and frame_idx < len(keypoints):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw keypoints on frame
            frame_with_keypoints = self._draw_keypoints_on_frame(
                frame=frame,
                keypoints=keypoints[frame_idx],
                keypoint_names=keypoint_names,
                colors=colors,
                connections=connections if show_skeleton else [],
                confidence_threshold=confidence_threshold,
                show_labels=show_labels,
            )

            out.write(frame_with_keypoints)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"  Processed {frame_idx}/{len(keypoints)} frames")

        cap.release()
        out.release()

        logger.info(f"Saved keypoint overlay video: {output_path}")
        return output_path

    def _draw_keypoints_on_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        colors: List[Tuple[int, int, int]],
        connections: List[Tuple[int, int]],
        confidence_threshold: float = 0.3,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draw keypoints and skeleton on a single frame."""
        frame = frame.copy()

        # Draw skeleton connections first (so keypoints are on top)
        for i, j in connections:
            if (keypoints[i, 2] > confidence_threshold and
                keypoints[j, 2] > confidence_threshold):
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw keypoints
        for idx, (name, color) in enumerate(zip(keypoint_names, colors)):
            x, y, conf = keypoints[idx]
            if conf > confidence_threshold:
                center = (int(x), int(y))
                # Outer circle
                cv2.circle(frame, center, 6, color, -1, cv2.LINE_AA)
                # Inner white circle
                cv2.circle(frame, center, 3, (255, 255, 255), -1, cv2.LINE_AA)

                # Label
                if show_labels:
                    label = f"{name[:8]}:{conf:.1f}"
                    cv2.putText(
                        frame, label, (int(x) + 8, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
                    )

        return frame

    def save_keypoint_frames(
        self,
        video_path: Union[str, Path],
        keypoints: np.ndarray,
        keypoint_names: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        frame_indices: Optional[List[int]] = None,
        num_samples: int = 10,
        confidence_threshold: float = 0.3,
        show_labels: bool = True,
        show_skeleton: bool = True,
    ) -> List[Path]:
        """
        Save individual frames with keypoint overlays.

        Args:
            video_path: Path to input video
            keypoints: Array of shape (num_frames, num_keypoints, 3)
            keypoint_names: List of keypoint names
            output_dir: Output directory for frames
            frame_indices: Specific frame indices to save (optional)
            num_samples: Number of evenly-spaced samples if frame_indices not specified
            confidence_threshold: Minimum confidence to draw keypoint
            show_labels: Show keypoint labels
            show_skeleton: Draw skeleton connections

        Returns:
            List of saved frame paths
        """
        video_path = Path(video_path)
        if output_dir is None:
            output_dir = self.output_dir / "keypoint_frames"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine which frames to save
        if frame_indices is None:
            # Evenly sample frames
            frame_indices = np.linspace(0, min(len(keypoints), total_frames) - 1, num_samples, dtype=int).tolist()

        # Generate colors
        num_keypoints = len(keypoint_names)
        colors = generate_keypoint_colors(num_keypoints)
        connections = self._get_skeleton_connections(keypoint_names) if show_skeleton else []

        saved_paths = []
        logger.info(f"Saving {len(frame_indices)} keypoint frames to {output_dir}")

        for idx in frame_indices:
            if idx >= len(keypoints) or idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Draw keypoints
            frame_with_keypoints = self._draw_keypoints_on_frame(
                frame=frame,
                keypoints=keypoints[idx],
                keypoint_names=keypoint_names,
                colors=colors,
                connections=connections,
                confidence_threshold=confidence_threshold,
                show_labels=show_labels,
            )

            # Save frame
            output_path = output_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(output_path), frame_with_keypoints)
            saved_paths.append(output_path)

        cap.release()
        logger.info(f"Saved {len(saved_paths)} keypoint frames")
        return saved_paths

    def create_comparison_grid(
        self,
        video_path: Union[str, Path],
        keypoints: np.ndarray,
        keypoint_names: List[str],
        output_path: Optional[Union[str, Path]] = None,
        num_frames: int = 9,
        confidence_threshold: float = 0.3,
    ) -> Path:
        """
        Create a grid comparison of original vs keypoint overlay frames.

        Args:
            video_path: Path to input video
            keypoints: Keypoint predictions
            keypoint_names: Keypoint names
            output_path: Output image path
            num_frames: Number of frames to include in grid
            confidence_threshold: Confidence threshold

        Returns:
            Path to saved grid image
        """
        video_path = Path(video_path)
        if output_path is None:
            output_path = self.output_dir / f"{video_path.stem}_comparison_grid.png"

        cap = cv2.VideoCapture(str(video_path))
        total_frames = min(len(keypoints), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Generate colors
        colors = generate_keypoint_colors(len(keypoint_names))
        connections = self._get_skeleton_connections(keypoint_names)

        # Collect frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Original frame (RGB for matplotlib)
            original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Frame with keypoints
            overlay = self._draw_keypoints_on_frame(
                frame=frame,
                keypoints=keypoints[idx],
                keypoint_names=keypoint_names,
                colors=colors,
                connections=connections,
                confidence_threshold=confidence_threshold,
                show_labels=False,
            )
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            frames.append((original, overlay_rgb, idx))

        cap.release()

        # Create grid
        n_cols = min(3, len(frames))
        n_rows = (len(frames) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 8, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, (orig, overlay, fidx) in enumerate(frames):
            row = i // n_cols
            col = (i % n_cols) * 2

            axes[row, col].imshow(orig)
            axes[row, col].set_title(f"Frame {fidx} - Original")
            axes[row, col].axis("off")

            axes[row, col + 1].imshow(overlay)
            axes[row, col + 1].set_title(f"Frame {fidx} - Keypoints")
            axes[row, col + 1].axis("off")

        # Hide unused axes
        for i in range(len(frames), n_rows * n_cols):
            row = i // n_cols
            col = (i % n_cols) * 2
            axes[row, col].axis("off")
            axes[row, col + 1].axis("off")

        plt.suptitle(f"Keypoint Detection: {video_path.stem}", fontsize=14)
        plt.tight_layout()

        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved comparison grid: {output_path}")
        return Path(output_path)
