"""
Cross-Species Comparison Visualizer Module.

Creates visualizations comparing different species:
- Body size comparison
- Action distribution comparison
- Velocity profiles (normalized by body size)
- Side-by-side GIF comparisons
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available - GIF generation will be disabled")


# ============================================================
# Data Classes
# ============================================================

@dataclass
class SpeciesAnalysisResult:
    """Analysis results for a single species."""
    species_name: str
    model_type: str
    num_frames: int
    video_path: Optional[Path] = None

    # Body size statistics
    body_size_mean: float = 0.0
    body_size_std: float = 0.0
    body_size_min: float = 0.0
    body_size_max: float = 0.0
    body_size_per_frame: Optional[np.ndarray] = None

    # Action statistics
    action_labels: Optional[np.ndarray] = None
    action_distribution: Dict[str, float] = field(default_factory=dict)

    # Velocity statistics (raw)
    velocity_mean: float = 0.0
    velocity_std: float = 0.0
    velocity_max: float = 0.0
    velocity_per_frame: Optional[np.ndarray] = None

    # Normalized velocity (by body size)
    velocity_normalized_mean: float = 0.0
    velocity_normalized_std: float = 0.0

    # Trajectory
    trajectory: Optional[np.ndarray] = None

    # Keypoint info
    num_keypoints: int = 0
    keypoint_names: List[str] = field(default_factory=list)


@dataclass
class SpeciesComparisonSummary:
    """Summary of cross-species comparison."""
    species_compared: List[str]
    largest_species: str
    smallest_species: str
    body_size_ratio: float  # largest / smallest
    most_active_species: str
    least_active_species: str
    action_agreement: Dict[str, float]  # pairwise agreement


# ============================================================
# Species Visualizer Class
# ============================================================

class SpeciesVisualizer:
    """Visualize and compare different species."""

    ACTION_COLORS = {
        "stationary": "#3498db",  # blue
        "walking": "#2ecc71",     # green
        "running": "#e74c3c",     # red
        "resting": "#3498db",
        "grooming": "#9b59b6",
        "unknown": "#95a5a6",
    }

    SPECIES_COLORS = {
        "mouse": "#3498db",
        "dog": "#e74c3c",
        "horse": "#2ecc71",
        "human": "#f39c12",
        "rat": "#9b59b6",
        "cat": "#1abc9c",
    }

    def __init__(self, output_dir: Union[str, Path], dpi: int = 150):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def _get_species_color(self, species: str) -> str:
        """Get color for species."""
        return self.SPECIES_COLORS.get(species.lower(), "#95a5a6")

    def create_body_size_comparison(
        self,
        results: List[SpeciesAnalysisResult],
        title: str = "Body Size Comparison",
    ) -> Path:
        """
        Create body size comparison visualization.

        Args:
            results: List of SpeciesAnalysisResult
            title: Plot title

        Returns:
            Path to saved figure
        """
        if not results:
            logger.warning("No results to visualize")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        species_names = [r.species_name for r in results]
        body_means = [r.body_size_mean for r in results]
        body_stds = [r.body_size_std for r in results]
        colors = [self._get_species_color(s) for s in species_names]

        # 1. Bar chart with error bars
        ax1 = axes[0]
        x = np.arange(len(species_names))
        bars = ax1.bar(x, body_means, yerr=body_stds, capsize=8,
                      color=colors, alpha=0.8, edgecolor='black')

        ax1.set_ylabel("Body Size (pixels)", fontsize=12)
        ax1.set_title("Mean Body Size with Std Dev", fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(species_names, fontsize=11)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, body_means, body_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. Box plot
        ax2 = axes[1]
        box_data = []
        for r in results:
            if r.body_size_per_frame is not None:
                box_data.append(r.body_size_per_frame)
            else:
                # Generate synthetic data from stats
                box_data.append(np.random.normal(r.body_size_mean, r.body_size_std, 100))

        bp = ax2.boxplot(box_data, labels=species_names, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel("Body Size (pixels)", fontsize=12)
        ax2.set_title("Body Size Distribution", fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Relative size (normalized to largest)
        ax3 = axes[2]
        max_size = max(body_means)
        relative_sizes = [m / max_size * 100 for m in body_means]

        bars = ax3.barh(species_names, relative_sizes, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xlabel("Relative Size (%)", fontsize=12)
        ax3.set_title("Relative Body Size (% of largest)", fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 110)
        ax3.grid(axis='x', alpha=0.3)

        for bar, rel in zip(bars, relative_sizes):
            ax3.text(rel + 2, bar.get_y() + bar.get_height()/2,
                    f'{rel:.0f}%', va='center', fontsize=10, fontweight='bold')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "body_size_comparison.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved body size comparison: {output_path}")
        return output_path

    def create_action_distribution_comparison(
        self,
        results: List[SpeciesAnalysisResult],
        title: str = "Action Distribution Comparison",
    ) -> Path:
        """
        Create action distribution comparison visualization.

        Args:
            results: List of SpeciesAnalysisResult
            title: Plot title

        Returns:
            Path to saved figure
        """
        if not results:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        species_names = [r.species_name for r in results]
        actions = ["stationary", "walking", "running"]

        # 1. Grouped bar chart
        ax1 = axes[0]
        x = np.arange(len(species_names))
        width = 0.25

        for i, action in enumerate(actions):
            values = [r.action_distribution.get(action, 0) for r in results]
            bars = ax1.bar(x + i * width, values, width, label=action.capitalize(),
                          color=self.ACTION_COLORS[action])

        ax1.set_ylabel("Percentage (%)", fontsize=12)
        ax1.set_title("Action Distribution by Species", fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(species_names, fontsize=11)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Stacked bar chart
        ax2 = axes[1]
        bottom = np.zeros(len(species_names))

        for action in actions:
            values = [r.action_distribution.get(action, 0) for r in results]
            ax2.bar(species_names, values, bottom=bottom, label=action.capitalize(),
                   color=self.ACTION_COLORS[action])
            bottom += np.array(values)

        ax2.set_ylabel("Percentage (%)", fontsize=12)
        ax2.set_title("Cumulative Action Distribution", fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "action_distribution_comparison.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved action distribution comparison: {output_path}")
        return output_path

    def create_velocity_comparison(
        self,
        results: List[SpeciesAnalysisResult],
        normalize_by_body: bool = True,
        title: str = "Velocity Comparison",
    ) -> Path:
        """
        Create velocity comparison visualization.

        Args:
            results: List of SpeciesAnalysisResult
            normalize_by_body: Whether to normalize by body size
            title: Plot title

        Returns:
            Path to saved figure
        """
        if not results:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        species_names = [r.species_name for r in results]
        colors = [self._get_species_color(s) for s in species_names]

        # 1. Velocity profile over time (normalized by video progress)
        ax1 = axes[0, 0]
        for i, r in enumerate(results):
            if r.velocity_per_frame is not None:
                # Normalize time to percentage
                time_pct = np.linspace(0, 100, len(r.velocity_per_frame))

                if normalize_by_body and r.body_size_mean > 0:
                    # Normalize velocity by body size (body-lengths per frame)
                    vel = r.velocity_per_frame / r.body_size_mean
                    ylabel = "Velocity (body-lengths/frame)"
                else:
                    vel = r.velocity_per_frame
                    ylabel = "Velocity (px/frame)"

                # Smooth for visualization
                window = min(15, len(vel) // 10)
                if window > 1:
                    vel_smooth = np.convolve(vel, np.ones(window)/window, mode='valid')
                    time_smooth = time_pct[:len(vel_smooth)]
                else:
                    vel_smooth = vel
                    time_smooth = time_pct

                ax1.plot(time_smooth, vel_smooth, label=r.species_name,
                        color=colors[i], linewidth=2, alpha=0.8)

        ax1.set_xlabel("Video Progress (%)", fontsize=11)
        ax1.set_ylabel(ylabel, fontsize=11)
        ax1.set_title("Velocity Profile Over Time" +
                     (" (Body-Size Normalized)" if normalize_by_body else ""),
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Mean velocity bar chart
        ax2 = axes[0, 1]
        if normalize_by_body:
            vel_means = [r.velocity_normalized_mean for r in results]
            vel_stds = [r.velocity_normalized_std for r in results]
            ylabel = "Mean Velocity (body-lengths/frame)"
        else:
            vel_means = [r.velocity_mean for r in results]
            vel_stds = [r.velocity_std for r in results]
            ylabel = "Mean Velocity (px/frame)"

        x = np.arange(len(species_names))
        bars = ax2.bar(x, vel_means, yerr=vel_stds, capsize=5, color=colors, alpha=0.8)

        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title("Mean Velocity Comparison", fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(species_names, fontsize=11)
        ax2.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, vel_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

        # 3. Velocity distribution (histogram)
        ax3 = axes[1, 0]
        for i, r in enumerate(results):
            if r.velocity_per_frame is not None:
                if normalize_by_body and r.body_size_mean > 0:
                    vel = r.velocity_per_frame / r.body_size_mean
                else:
                    vel = r.velocity_per_frame

                ax3.hist(vel, bins=50, alpha=0.5, label=r.species_name,
                        color=colors[i], density=True)

        ax3.set_xlabel("Velocity" + (" (body-lengths/frame)" if normalize_by_body else " (px/frame)"),
                      fontsize=11)
        ax3.set_ylabel("Density", fontsize=11)
        ax3.set_title("Velocity Distribution", fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. Max velocity comparison
        ax4 = axes[1, 1]
        max_vels = [r.velocity_max for r in results]
        if normalize_by_body:
            max_vels_norm = [r.velocity_max / r.body_size_mean if r.body_size_mean > 0 else 0
                           for r in results]
            ax4.bar(species_names, max_vels_norm, color=colors, alpha=0.8)
            ax4.set_ylabel("Max Velocity (body-lengths/frame)", fontsize=11)
        else:
            ax4.bar(species_names, max_vels, color=colors, alpha=0.8)
            ax4.set_ylabel("Max Velocity (px/frame)", fontsize=11)

        ax4.set_title("Maximum Velocity Comparison", fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "velocity_comparison.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved velocity comparison: {output_path}")
        return output_path

    def create_trajectory_comparison(
        self,
        results: List[SpeciesAnalysisResult],
        normalize: bool = True,
        title: str = "Trajectory Comparison",
    ) -> Path:
        """
        Create trajectory comparison visualization.

        Args:
            results: List of SpeciesAnalysisResult with trajectory data
            normalize: Whether to normalize trajectories to unit scale
            title: Plot title

        Returns:
            Path to saved figure
        """
        if not results:
            return None

        n_species = len(results)
        fig, axes = plt.subplots(1, n_species, figsize=(5 * n_species, 5))
        if n_species == 1:
            axes = [axes]

        for ax, r in zip(axes, results):
            if r.trajectory is None:
                ax.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
                ax.set_title(r.species_name)
                continue

            traj = r.trajectory.copy()

            if normalize:
                # Normalize to center and scale
                traj_centered = traj - traj.mean(axis=0)
                scale = np.abs(traj_centered).max()
                if scale > 0:
                    traj_norm = traj_centered / scale
                else:
                    traj_norm = traj_centered
                traj = traj_norm

            # Color by time
            colors = plt.cm.viridis(np.linspace(0, 1, len(traj)))
            ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)),
                      cmap='viridis', s=5, alpha=0.7)

            # Mark start and end
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
            ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=12, label='End')

            ax.set_title(f"{r.species_name}\n({r.num_frames} frames)",
                        fontsize=12, fontweight='bold')
            ax.set_xlabel("X" + (" (normalized)" if normalize else " (pixels)"))
            ax.set_ylabel("Y" + (" (normalized)" if normalize else " (pixels)"))
            ax.legend(loc='upper right', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            if not normalize:
                ax.invert_yaxis()

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "trajectory_comparison.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved trajectory comparison: {output_path}")
        return output_path

    def create_comprehensive_comparison(
        self,
        results: List[SpeciesAnalysisResult],
    ) -> Path:
        """
        Create comprehensive comparison figure with all metrics.

        Args:
            results: List of SpeciesAnalysisResult

        Returns:
            Path to saved figure
        """
        if not results:
            return None

        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        species_names = [r.species_name for r in results]
        colors = [self._get_species_color(s) for s in species_names]

        # 1. Body size bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        body_means = [r.body_size_mean for r in results]
        body_stds = [r.body_size_std for r in results]
        x = np.arange(len(species_names))
        ax1.bar(x, body_means, yerr=body_stds, capsize=5, color=colors, alpha=0.8)
        ax1.set_ylabel("Body Size (px)")
        ax1.set_title("Body Size", fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(species_names)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Action distribution (grouped)
        ax2 = fig.add_subplot(gs[0, 1])
        actions = ["stationary", "walking", "running"]
        width = 0.25
        for i, action in enumerate(actions):
            values = [r.action_distribution.get(action, 0) for r in results]
            ax2.bar(x + i * width, values, width, label=action.capitalize(),
                   color=self.ACTION_COLORS[action])
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("Action Distribution", fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(species_names)
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Normalized velocity
        ax3 = fig.add_subplot(gs[0, 2])
        vel_norm_means = [r.velocity_normalized_mean for r in results]
        ax3.bar(species_names, vel_norm_means, color=colors, alpha=0.8)
        ax3.set_ylabel("Velocity (body-lengths/frame)")
        ax3.set_title("Normalized Velocity", fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # 4-6. Trajectories
        for i, r in enumerate(results[:3]):  # Up to 3 species
            ax = fig.add_subplot(gs[1, i])
            if r.trajectory is not None:
                traj = r.trajectory
                ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)),
                          cmap='viridis', s=3, alpha=0.6)
                ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
                ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=8)
                ax.invert_yaxis()
            ax.set_title(f"{r.species_name} Trajectory", fontweight='bold')
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # 7. Velocity profiles
        ax7 = fig.add_subplot(gs[2, 0:2])
        for i, r in enumerate(results):
            if r.velocity_per_frame is not None:
                time_pct = np.linspace(0, 100, len(r.velocity_per_frame))
                vel_norm = r.velocity_per_frame / r.body_size_mean if r.body_size_mean > 0 else r.velocity_per_frame

                # Smooth
                window = min(15, len(vel_norm) // 10)
                if window > 1:
                    vel_smooth = np.convolve(vel_norm, np.ones(window)/window, mode='valid')
                    time_smooth = time_pct[:len(vel_smooth)]
                else:
                    vel_smooth, time_smooth = vel_norm, time_pct

                ax7.plot(time_smooth, vel_smooth, label=r.species_name,
                        color=colors[i], linewidth=2, alpha=0.8)

        ax7.set_xlabel("Video Progress (%)")
        ax7.set_ylabel("Velocity (body-lengths/frame)")
        ax7.set_title("Velocity Profile Comparison", fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Summary statistics table
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        table_data = []
        for r in results:
            table_data.append([
                r.species_name,
                f"{r.num_frames}",
                f"{r.body_size_mean:.1f}",
                f"{r.velocity_mean:.2f}",
                f"{r.action_distribution.get('stationary', 0):.0f}%",
                f"{r.action_distribution.get('walking', 0):.0f}%",
                f"{r.action_distribution.get('running', 0):.0f}%",
            ])

        table = ax8.table(
            cellText=table_data,
            colLabels=["Species", "Frames", "Body (px)", "Vel (px/f)", "Static", "Walk", "Run"],
            loc="center",
            cellLoc="center",
            colWidths=[0.15, 0.1, 0.12, 0.12, 0.1, 0.1, 0.1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Style header
        for j in range(7):
            table[(0, j)].set_facecolor("#3498db")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        fig.suptitle("Cross-Species Behavior Analysis", fontsize=16, fontweight='bold', y=0.98)

        output_path = self.output_dir / "comprehensive_species_comparison.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor='white')
        plt.close(fig)

        logger.info(f"Saved comprehensive comparison: {output_path}")
        return output_path

    def create_comparison_gif(
        self,
        results: List[SpeciesAnalysisResult],
        max_frames: int = 100,
        fps: float = 8.0,
        confidence_threshold: float = 0.3,
    ) -> Optional[Path]:
        """
        Create side-by-side GIF comparing species keypoints.

        Args:
            results: List of SpeciesAnalysisResult with video paths
            max_frames: Maximum frames
            fps: GIF frame rate
            confidence_threshold: Confidence threshold

        Returns:
            Path to saved GIF
        """
        try:
            import imageio
        except ImportError:
            logger.warning("imageio not installed - skipping GIF")
            return None

        # Filter results that have video paths
        valid_results = [r for r in results if r.video_path and Path(r.video_path).exists()]
        if not valid_results:
            logger.warning("No valid video paths for GIF creation")
            return None

        # Open all videos
        caps = []
        for r in valid_results:
            cap = cv2.VideoCapture(str(r.video_path))
            caps.append(cap)

        # Get min frames
        min_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
        min_frames = min(min_frames, max_frames, *[r.num_frames for r in valid_results])

        # Sample frames
        if min_frames > max_frames:
            frame_indices = list(range(0, min_frames, min_frames // max_frames))[:max_frames]
        else:
            frame_indices = list(range(min_frames))

        # Panel dimensions (consistent for all)
        panel_width = 320
        panel_height = 240

        combined_frames = []

        for frame_idx in frame_indices:
            panels = []

            for cap, r in zip(caps, valid_results):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Resize
                panel = cv2.resize(frame, (panel_width, panel_height))

                # Add species label
                label = f"{r.species_name.upper()}"
                cv2.rectangle(panel, (5, 5), (len(label) * 12 + 15, 30), (0, 0, 0), -1)
                cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                panels.append(panel)

            if panels:
                # Combine horizontally
                combined = np.hstack(panels)
                combined_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        # Release videos
        for cap in caps:
            cap.release()

        if combined_frames:
            output_path = self.output_dir / "species_comparison.gif"
            imageio.mimsave(str(output_path), combined_frames, duration=1.0/fps, loop=0)
            logger.info(f"Saved species comparison GIF: {output_path}")
            return output_path

        return None


# ============================================================
# Utility Functions
# ============================================================

def create_species_result(
    species_name: str,
    model_type: str,
    keypoints: np.ndarray,
    keypoint_names: List[str],
    video_path: Optional[Path] = None,
    fps: float = 30.0,
) -> SpeciesAnalysisResult:
    """
    Create SpeciesAnalysisResult from keypoint data.

    Args:
        species_name: Name of species
        model_type: Model type used
        keypoints: Keypoints array (frames, keypoints, 3)
        keypoint_names: Keypoint names
        video_path: Optional video path
        fps: Video frame rate

    Returns:
        SpeciesAnalysisResult object
    """
    from src.analysis.behavior import estimate_body_size
    from src.models.action_classifier import UnifiedActionClassifier

    num_frames = len(keypoints)

    # Body size estimation
    body_stats = estimate_body_size(keypoints, keypoint_names, model_type)

    # Action classification
    classifier = UnifiedActionClassifier(species=species_name, fps=fps)
    metrics = classifier.analyze(keypoints, keypoint_names)

    # Get trajectory
    center_kps = ["mouse_center", "back_middle", "neck"]
    trajectory = None
    for center_name in center_kps:
        if center_name in keypoint_names:
            idx = keypoint_names.index(center_name)
            trajectory = keypoints[:, idx, :2]
            break

    if trajectory is None:
        # Use mean of valid keypoints
        valid_mask = keypoints[:, :, 2] > 0.3
        trajectory = np.zeros((num_frames, 2))
        for i in range(num_frames):
            valid = valid_mask[i]
            if valid.any():
                trajectory[i] = keypoints[i, valid, :2].mean(axis=0)

    # Velocity
    velocity = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
    velocity = np.concatenate([[0], velocity])

    # Normalized velocity
    body_mean = body_stats.get("mean", 100)
    vel_norm = velocity / body_mean if body_mean > 0 else velocity

    # Action distribution
    action_dist = {}
    for action, stats in metrics.action_summary.get("actions", {}).items():
        action_dist[action] = stats.get("percentage", 0)

    return SpeciesAnalysisResult(
        species_name=species_name,
        model_type=model_type,
        num_frames=num_frames,
        video_path=video_path,
        body_size_mean=body_stats.get("mean", 0),
        body_size_std=body_stats.get("std", 0),
        body_size_min=body_stats.get("min", 0),
        body_size_max=body_stats.get("max", 0),
        body_size_per_frame=body_stats.get("per_frame"),
        action_labels=metrics.action_labels,
        action_distribution=action_dist,
        velocity_mean=velocity.mean(),
        velocity_std=velocity.std(),
        velocity_max=velocity.max(),
        velocity_per_frame=velocity,
        velocity_normalized_mean=vel_norm.mean(),
        velocity_normalized_std=vel_norm.std(),
        trajectory=trajectory,
        num_keypoints=len(keypoint_names),
        keypoint_names=keypoint_names,
    )


def compute_comparison_summary(
    results: List[SpeciesAnalysisResult],
) -> SpeciesComparisonSummary:
    """
    Compute summary statistics for species comparison.

    Args:
        results: List of SpeciesAnalysisResult

    Returns:
        SpeciesComparisonSummary object
    """
    if not results:
        return None

    species_names = [r.species_name for r in results]
    body_sizes = {r.species_name: r.body_size_mean for r in results}

    largest = max(body_sizes, key=body_sizes.get)
    smallest = min(body_sizes, key=body_sizes.get)
    size_ratio = body_sizes[largest] / body_sizes[smallest] if body_sizes[smallest] > 0 else 0

    # Activity (% walking + running)
    activity = {}
    for r in results:
        active = r.action_distribution.get("walking", 0) + r.action_distribution.get("running", 0)
        activity[r.species_name] = active

    most_active = max(activity, key=activity.get)
    least_active = min(activity, key=activity.get)

    # Action agreement (pairwise)
    agreement = {}
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            if r1.action_labels is not None and r2.action_labels is not None:
                min_len = min(len(r1.action_labels), len(r2.action_labels))
                agree = np.mean(r1.action_labels[:min_len] == r2.action_labels[:min_len])
                agreement[f"{r1.species_name}_vs_{r2.species_name}"] = agree

    return SpeciesComparisonSummary(
        species_compared=species_names,
        largest_species=largest,
        smallest_species=smallest,
        body_size_ratio=size_ratio,
        most_active_species=most_active,
        least_active_species=least_active,
        action_agreement=agreement,
    )
