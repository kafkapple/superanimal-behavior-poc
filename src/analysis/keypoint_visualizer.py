"""
Keypoint Comparison Visualizer Module.

Creates visualizations comparing different keypoint configurations:
- Side-by-side keypoint overlays
- Trajectory comparisons
- Action recognition performance by keypoint preset
- Animated GIF comparisons
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
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
# Keypoint Preset Definitions
# ============================================================

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

PRESET_DESCRIPTIONS = {
    "full": "All 27 keypoints - Detailed pose analysis, grooming detection, gait analysis",
    "standard": "11 keypoints - Open Field Test, general behavior tracking",
    "mars": "7 keypoints - Social interaction, multi-animal tracking (MARS-compatible)",
    "locomotion": "5 keypoints - Movement/speed analysis, basic trajectory",
    "minimal": "3 keypoints - Real-time tracking, high-speed video",
}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class KeypointAnalysisResult:
    """Results from keypoint analysis with a specific preset."""
    preset_name: str
    num_keypoints: int
    keypoint_names: List[str]
    keypoints: np.ndarray  # (frames, keypoints, 3)
    trajectory: np.ndarray  # (frames, 2)
    velocity: np.ndarray  # (frames,)
    action_labels: np.ndarray  # (frames,)
    action_distribution: Dict[str, float] = field(default_factory=dict)
    mean_confidence: float = 0.0
    detection_rate: float = 0.0
    # Action recognition performance metrics (vs reference/full preset)
    accuracy: float = 0.0  # Overall accuracy vs reference
    f1_scores: Dict[str, float] = field(default_factory=dict)  # Per-class F1
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    agreement_with_full: float = 0.0  # Agreement rate with full preset


@dataclass
class KeypointComparisonMetrics:
    """Metrics comparing different keypoint presets."""
    presets_compared: List[str]
    reference_preset: str  # Usually "full"
    trajectory_agreement: Dict[str, float]  # pairwise agreement
    action_agreement: Dict[str, float]  # pairwise agreement
    velocity_correlation: Dict[str, float]  # pairwise correlation
    # Performance metrics per preset
    accuracy_by_preset: Dict[str, float] = field(default_factory=dict)
    f1_by_preset: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Summary
    best_preset_by_accuracy: str = ""
    accuracy_drop_from_full: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Keypoint Visualizer Class
# ============================================================

class KeypointVisualizer:
    """Visualize and compare keypoint configurations."""

    SKELETON_CONNECTIONS = [
        # Head
        ("nose", "neck"), ("nose", "left_ear"), ("nose", "right_ear"),
        ("nose", "head_midpoint"),
        # Spine
        ("neck", "mid_back"), ("mid_back", "mouse_center"),
        ("neck", "mouse_center"),  # Fallback
        ("nose", "mouse_center"),  # Minimal preset
        ("mouse_center", "mid_backend"), ("mid_backend", "tail_base"),
        ("mouse_center", "tail_base"),  # Fallback
        # Body
        ("neck", "left_shoulder"), ("neck", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        # Tail
        ("tail_base", "tail1"), ("tail1", "tail2"), ("tail2", "tail3"),
        ("tail_base", "tail_end"),
    ]

    ACTION_COLORS = {
        0: "#3498db",  # stationary - blue
        1: "#2ecc71",  # walking - green
        2: "#e74c3c",  # running - red
        "stationary": "#3498db",
        "walking": "#2ecc71",
        "running": "#e74c3c",
    }

    def __init__(self, output_dir: Union[str, Path], dpi: int = 150):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def filter_keypoints(
        self,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        preset_name: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """Filter keypoints based on preset."""
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

    def _get_skeleton_connections(self, keypoint_names: List[str]) -> List[Tuple[int, int]]:
        """Get skeleton connections for given keypoint names."""
        connections = []
        for kp1, kp2 in self.SKELETON_CONNECTIONS:
            if kp1 in keypoint_names and kp2 in keypoint_names:
                connections.append((keypoint_names.index(kp1), keypoint_names.index(kp2)))
        return connections

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for keypoints."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def create_preset_comparison_figure(
        self,
        video_path: Path,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        presets: List[str] = ["full", "standard", "minimal"],
        frame_indices: List[int] = None,
        num_frames: int = 4,
        confidence_threshold: float = 0.3,
    ) -> Path:
        """
        Create static figure comparing keypoint presets.

        Args:
            video_path: Path to video
            keypoints: Full keypoints array (frames, keypoints, 3)
            all_keypoint_names: All keypoint names
            presets: Presets to compare
            frame_indices: Specific frames to use
            num_frames: Number of frames if frame_indices not specified
            confidence_threshold: Confidence threshold

        Returns:
            Path to saved figure
        """
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(keypoints))

        if frame_indices is None:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        num_presets = len(presets)
        num_rows = len(frame_indices)

        fig = plt.figure(figsize=(4 * num_presets, 4 * num_rows + 2))
        gs = gridspec.GridSpec(num_rows + 1, num_presets, height_ratios=[1] * num_rows + [0.4])

        fig.suptitle(
            f"Keypoint Preset Comparison: {video_path.stem}\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=14, fontweight="bold", y=0.98
        )

        # Process each frame and preset
        for row, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for col, preset_name in enumerate(presets):
                ax = fig.add_subplot(gs[row, col])

                filtered_kp, filtered_names = self.filter_keypoints(
                    keypoints, all_keypoint_names, preset_name
                )

                # Draw frame with keypoints
                frame_with_kp = self._draw_keypoints_on_image(
                    frame_rgb.copy(),
                    filtered_kp[frame_idx],
                    filtered_names,
                    confidence_threshold,
                )

                ax.imshow(frame_with_kp)
                ax.axis("off")

                if row == 0:
                    preset_kp = KEYPOINT_PRESETS.get(preset_name)
                    kp_count = len(filtered_names) if preset_kp else len(all_keypoint_names)
                    ax.set_title(f"{preset_name.upper()}\n({kp_count} keypoints)",
                                fontsize=11, fontweight="bold")

                if col == 0:
                    ax.text(-0.1, 0.5, f"Frame {frame_idx}", transform=ax.transAxes,
                           fontsize=10, rotation=90, va='center', ha='right')

        cap.release()

        # Add summary table
        ax_table = fig.add_subplot(gs[-1, :])
        ax_table.axis("off")

        table_data = []
        for preset_name in presets:
            preset = KEYPOINT_PRESETS.get(preset_name)
            if preset is None:
                kp_list = all_keypoint_names
            else:
                kp_list = [kp for kp in preset if kp in all_keypoint_names]

            desc = PRESET_DESCRIPTIONS.get(preset_name, "")
            table_data.append([
                preset_name.upper(),
                str(len(kp_list)),
                ", ".join(kp_list[:4]) + ("..." if len(kp_list) > 4 else ""),
                desc[:50] + ("..." if len(desc) > 50 else ""),
            ])

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Preset", "#KP", "Key Points", "Use Case"],
            loc="center",
            cellLoc="left",
            colWidths=[0.1, 0.08, 0.35, 0.47],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # Style header
        for i in range(len(presets)):
            for j in range(4):
                table[(0, j)].set_facecolor("#3498db")
                table[(0, j)].set_text_props(color="white", fontweight="bold")

        plt.tight_layout()

        output_path = self.output_dir / f"preset_comparison_{video_path.stem}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Saved preset comparison: {output_path}")
        return output_path

    def _draw_keypoints_on_image(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        confidence_threshold: float = 0.3,
    ) -> np.ndarray:
        """Draw keypoints on image (RGB format for matplotlib)."""
        # Convert to BGR for cv2 drawing
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        colors = self._generate_colors(len(keypoint_names))
        connections = self._get_skeleton_connections(keypoint_names)

        # Draw skeleton
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints):
                if keypoints[i, 2] > confidence_threshold and keypoints[j, 2] > confidence_threshold:
                    pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                    pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                    cv2.line(image_bgr, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw keypoints
        for idx, (name, color) in enumerate(zip(keypoint_names, colors)):
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf > confidence_threshold:
                    center = (int(x), int(y))
                    cv2.circle(image_bgr, center, 5, color, -1, cv2.LINE_AA)
                    cv2.circle(image_bgr, center, 2, (255, 255, 255), -1, cv2.LINE_AA)

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def create_trajectory_comparison(
        self,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        presets: List[str] = ["full", "standard", "minimal"],
        reference_keypoint: str = "mouse_center",
        video_name: str = "video",
    ) -> Path:
        """
        Create trajectory comparison for different presets.

        Args:
            keypoints: Full keypoints array
            all_keypoint_names: All keypoint names
            presets: Presets to compare
            reference_keypoint: Keypoint to use for trajectory
            video_name: Name for output file

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, len(presets), figsize=(5 * len(presets), 5))
        if len(presets) == 1:
            axes = [axes]

        fig.suptitle(f"Trajectory Comparison by Keypoint Preset: {video_name}",
                    fontsize=14, fontweight="bold")

        colors = plt.cm.viridis(np.linspace(0, 1, 256))

        for ax, preset_name in zip(axes, presets):
            filtered_kp, filtered_names = self.filter_keypoints(
                keypoints, all_keypoint_names, preset_name
            )

            # Get trajectory from reference keypoint or centroid
            if reference_keypoint in filtered_names:
                ref_idx = filtered_names.index(reference_keypoint)
                trajectory = filtered_kp[:, ref_idx, :2]
            else:
                # Use mean of all valid keypoints
                valid_mask = filtered_kp[:, :, 2] > 0.3
                trajectory = np.zeros((len(filtered_kp), 2))
                for i in range(len(filtered_kp)):
                    valid = valid_mask[i]
                    if valid.any():
                        trajectory[i] = filtered_kp[i, valid, :2].mean(axis=0)

            # Compute velocity for coloring
            velocity = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
            velocity = np.concatenate([[0], velocity])

            # Normalize velocity for coloring
            if velocity.max() > 0:
                vel_norm = velocity / velocity.max()
            else:
                vel_norm = np.zeros_like(velocity)

            # Plot trajectory with velocity coloring
            scatter = ax.scatter(
                trajectory[:, 0], trajectory[:, 1],
                c=velocity, cmap="viridis", s=15, alpha=0.7
            )

            # Mark start and end
            ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=12, label="Start")
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], "rs", markersize=12, label="End")

            # Title with keypoint count
            preset_kp = KEYPOINT_PRESETS.get(preset_name)
            kp_count = len(filtered_names) if preset_kp else len(all_keypoint_names)
            ax.set_title(f"{preset_name.upper()} ({kp_count} keypoints)")

            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.invert_yaxis()
            ax.legend(loc="upper right", fontsize=8)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=axes[-1], label="Velocity (px/frame)")
        plt.tight_layout()

        output_path = self.output_dir / f"trajectory_comparison_{video_name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved trajectory comparison: {output_path}")
        return output_path

    def create_action_recognition_comparison(
        self,
        results: List[KeypointAnalysisResult],
        video_name: str = "video",
    ) -> Path:
        """
        Create action recognition comparison across presets.

        Args:
            results: List of KeypointAnalysisResult for different presets
            video_name: Name for output file

        Returns:
            Path to saved figure
        """
        if not results:
            logger.warning("No results to visualize")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Prepare data
        preset_names = [r.preset_name for r in results]
        num_keypoints = [r.num_keypoints for r in results]

        # 1. Action distribution bar chart
        ax1 = axes[0, 0]
        x = np.arange(len(preset_names))
        width = 0.25
        actions = ["stationary", "walking", "running"]
        action_colors = [self.ACTION_COLORS[a] for a in actions]

        for i, action in enumerate(actions):
            values = [r.action_distribution.get(action, 0) for r in results]
            bars = ax1.bar(x + i * width, values, width, label=action.capitalize(),
                          color=action_colors[i])

        ax1.set_ylabel("Percentage (%)")
        ax1.set_title("Action Distribution by Keypoint Preset")
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)])
        ax1.legend()
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Timeline comparison (first 500 frames)
        ax2 = axes[0, 1]
        max_display = 500
        for i, result in enumerate(results):
            labels = result.action_labels[:max_display]
            y_offset = i * 1.2
            for j, label in enumerate(labels):
                color = self.ACTION_COLORS.get(int(label), "#95a5a6")
                ax2.axvline(j, ymin=(y_offset) / (len(results) * 1.2),
                           ymax=(y_offset + 1) / (len(results) * 1.2),
                           color=color, linewidth=0.5)

        ax2.set_yticks([i * 1.2 + 0.5 for i in range(len(results))])
        ax2.set_yticklabels([f"{r.preset_name}\n({r.num_keypoints} kp)" for r in results])
        ax2.set_xlabel("Frame")
        ax2.set_title("Action Timeline Comparison (First 500 frames)")
        ax2.set_xlim(0, max_display)

        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=self.ACTION_COLORS[a], lw=4, label=a.capitalize())
                         for a in actions]
        ax2.legend(handles=legend_elements, loc="upper right")

        # 3. Keypoint count vs confidence
        ax3 = axes[1, 0]
        confidences = [r.mean_confidence for r in results]
        detection_rates = [r.detection_rate * 100 for r in results]

        ax3.bar(x - 0.2, confidences, 0.4, label="Mean Confidence", color="#3498db")
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + 0.2, detection_rates, 0.4, label="Detection Rate (%)", color="#2ecc71")

        ax3.set_ylabel("Mean Confidence", color="#3498db")
        ax3_twin.set_ylabel("Detection Rate (%)", color="#2ecc71")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)])
        ax3.set_title("Keypoint Quality Metrics")
        ax3.legend(loc="upper left")
        ax3_twin.legend(loc="upper right")
        ax3.set_ylim(0, 1)
        ax3_twin.set_ylim(0, 100)

        # 4. Velocity statistics
        ax4 = axes[1, 1]
        mean_velocities = [r.velocity.mean() for r in results]
        std_velocities = [r.velocity.std() for r in results]
        max_velocities = [r.velocity.max() for r in results]

        bar_width = 0.25
        ax4.bar(x - bar_width, mean_velocities, bar_width, label="Mean", color="#3498db")
        ax4.bar(x, std_velocities, bar_width, label="Std", color="#f39c12")
        ax4.bar(x + bar_width, max_velocities, bar_width, label="Max", color="#e74c3c")

        ax4.set_ylabel("Velocity (px/frame)")
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)])
        ax4.set_title("Velocity Statistics by Preset")
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        fig.suptitle(f"Action Recognition Analysis: {video_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / f"action_comparison_{video_name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved action comparison: {output_path}")
        return output_path

    def create_performance_by_keypoint_count(
        self,
        results: List[KeypointAnalysisResult],
        video_name: str = "video",
    ) -> Path:
        """
        Create performance metrics comparison by keypoint count.

        Shows how accuracy and F1 scores change as keypoint count decreases.

        Args:
            results: List of KeypointAnalysisResult with performance metrics
            video_name: Name for output file

        Returns:
            Path to saved figure
        """
        if not results:
            logger.warning("No results to visualize")
            return None

        # Sort by keypoint count (descending)
        results_sorted = sorted(results, key=lambda x: x.num_keypoints, reverse=True)

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

        preset_names = [r.preset_name for r in results_sorted]
        num_keypoints = [r.num_keypoints for r in results_sorted]
        x = np.arange(len(preset_names))

        # Colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))

        # 1. Accuracy vs Keypoint Count
        ax1 = fig.add_subplot(gs[0, 0])
        accuracies = [r.accuracy * 100 for r in results_sorted]

        bars = ax1.bar(x, accuracies, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel("Accuracy (%)", fontsize=11)
        ax1.set_title("Action Recognition Accuracy by Keypoint Preset", fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)])
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Highlight best and worst
        best_idx = np.argmax(accuracies)
        worst_idx = np.argmin(accuracies)
        bars[best_idx].set_edgecolor('#27ae60')
        bars[best_idx].set_linewidth(3)
        bars[worst_idx].set_edgecolor('#e74c3c')
        bars[worst_idx].set_linewidth(3)

        # 2. F1 Scores by Action Class
        ax2 = fig.add_subplot(gs[0, 1])
        actions = ["stationary", "walking", "running"]
        action_colors = [self.ACTION_COLORS[a] for a in actions]
        width = 0.25

        for i, action in enumerate(actions):
            f1_values = [r.f1_scores.get(action, 0) for r in results_sorted]
            ax2.bar(x + i * width, f1_values, width, label=action.capitalize(),
                   color=action_colors[i], edgecolor='black', linewidth=0.5)

        ax2.set_ylabel("F1 Score", fontsize=11)
        ax2.set_title("F1 Score by Action Class", fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)])
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Accuracy Drop from Full Preset
        ax3 = fig.add_subplot(gs[1, 0])
        full_accuracy = results_sorted[0].accuracy * 100 if results_sorted else 100

        accuracy_drops = [full_accuracy - (r.accuracy * 100) for r in results_sorted]

        bars = ax3.bar(x, accuracy_drops, color=['#27ae60' if d <= 5 else '#f39c12' if d <= 15 else '#e74c3c' for d in accuracy_drops])
        ax3.set_ylabel("Accuracy Drop (%)", fontsize=11)
        ax3.set_title("Accuracy Drop from Full Preset (Lower = Better)", fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)])
        ax3.axhline(y=5, color='#27ae60', linestyle='--', alpha=0.7, label='Acceptable (5%)')
        ax3.axhline(y=15, color='#f39c12', linestyle='--', alpha=0.7, label='Marginal (15%)')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

        for bar, drop in zip(bars, accuracy_drops):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{drop:.1f}%', ha='center', va='bottom', fontsize=10)

        # 4. Agreement with Full Preset
        ax4 = fig.add_subplot(gs[1, 1])
        agreements = [r.agreement_with_full * 100 for r in results_sorted]

        bars = ax4.bar(x, agreements, color=colors, edgecolor='black')
        ax4.set_ylabel("Agreement Rate (%)", fontsize=11)
        ax4.set_title("Action Label Agreement with Full Preset", fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)])
        ax4.set_ylim(0, 105)
        ax4.axhline(y=90, color='#27ae60', linestyle='--', alpha=0.7, label='High Agreement (90%)')
        ax4.legend(loc='lower left', fontsize=9)
        ax4.grid(axis='y', alpha=0.3)

        for bar, agree in zip(bars, agreements):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{agree:.1f}%', ha='center', va='bottom', fontsize=10)

        # 5. Keypoint Count vs Performance Trade-off (scatter)
        ax5 = fig.add_subplot(gs[2, 0])

        scatter = ax5.scatter(num_keypoints, accuracies, c=colors, s=200, edgecolors='black', linewidths=2)

        # Add labels
        for i, (n, a, name) in enumerate(zip(num_keypoints, accuracies, preset_names)):
            ax5.annotate(name.upper(), (n, a), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

        ax5.set_xlabel("Number of Keypoints", fontsize=11)
        ax5.set_ylabel("Accuracy (%)", fontsize=11)
        ax5.set_title("Keypoint Count vs Accuracy Trade-off", fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(num_keypoints, accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(num_keypoints), max(num_keypoints), 100)
        ax5.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"Trend (slope: {z[0]:.2f}%/kp)")
        ax5.legend(loc='lower right')

        # 6. Summary Table
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')

        table_data = []
        for r in results_sorted:
            mean_f1 = np.mean(list(r.f1_scores.values())) if r.f1_scores else 0
            table_data.append([
                r.preset_name.upper(),
                str(r.num_keypoints),
                f"{r.accuracy * 100:.1f}%",
                f"{mean_f1:.3f}",
                f"{r.agreement_with_full * 100:.1f}%",
                f"{full_accuracy - r.accuracy * 100:.1f}%",
            ])

        table = ax6.table(
            cellText=table_data,
            colLabels=["Preset", "KP#", "Accuracy", "Mean F1", "Agreement", "Drop"],
            loc="center",
            cellLoc="center",
            colWidths=[0.18, 0.1, 0.15, 0.15, 0.17, 0.12],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style header
        for j in range(6):
            table[(0, j)].set_facecolor("#3498db")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        # Highlight best row
        for j in range(6):
            table[(1, j)].set_facecolor("#e8f8f5")

        fig.suptitle(f"Action Recognition Performance by Keypoint Count: {video_name}",
                    fontsize=14, fontweight='bold', y=0.98)

        output_path = self.output_dir / f"performance_by_keypoint_{video_name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor='white')
        plt.close(fig)

        logger.info(f"Saved performance by keypoint count: {output_path}")
        return output_path

    def create_keypoint_count_analysis(
        self,
        all_keypoint_names: List[str],
    ) -> Path:
        """
        Create analysis figure showing keypoint count trade-offs.

        Args:
            all_keypoint_names: All available keypoint names

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        presets = ["full", "standard", "mars", "locomotion", "minimal"]
        preset_kp_counts = []

        for preset in presets:
            preset_kps = KEYPOINT_PRESETS.get(preset)
            if preset_kps is None:
                count = len(all_keypoint_names)
            else:
                count = len([kp for kp in preset_kps if kp in all_keypoint_names])
            preset_kp_counts.append(count)

        # Bar chart of keypoint counts
        ax1 = axes[0]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(presets)))
        bars = ax1.bar(presets, preset_kp_counts, color=colors)

        ax1.set_ylabel("Number of Keypoints")
        ax1.set_title("Keypoint Count by Preset")
        ax1.set_xlabel("Preset")

        for bar, count in zip(bars, preset_kp_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')

        # Keypoint inclusion matrix
        ax2 = axes[1]

        # Get representative keypoints for visualization
        rep_keypoints = ["nose", "left_ear", "right_ear", "neck", "mouse_center",
                        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
                        "tail_base", "tail_end", "mid_back", "tail1", "tail2"]
        rep_keypoints = [kp for kp in rep_keypoints if kp in all_keypoint_names]

        # Create inclusion matrix
        matrix = np.zeros((len(rep_keypoints), len(presets)))
        for j, preset in enumerate(presets):
            preset_kps = KEYPOINT_PRESETS.get(preset)
            if preset_kps is None:
                preset_kps = all_keypoint_names
            for i, kp in enumerate(rep_keypoints):
                if kp in preset_kps:
                    matrix[i, j] = 1

        im = ax2.imshow(matrix, cmap="Blues", aspect="auto")

        ax2.set_xticks(np.arange(len(presets)))
        ax2.set_yticks(np.arange(len(rep_keypoints)))
        ax2.set_xticklabels([p.upper() for p in presets])
        ax2.set_yticklabels(rep_keypoints)
        ax2.set_title("Keypoint Inclusion Matrix")

        # Add grid
        ax2.set_xticks(np.arange(len(presets) + 1) - 0.5, minor=True)
        ax2.set_yticks(np.arange(len(rep_keypoints) + 1) - 0.5, minor=True)
        ax2.grid(which="minor", color="white", linestyle="-", linewidth=2)

        plt.tight_layout()

        output_path = self.output_dir / "keypoint_count_analysis.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved keypoint count analysis: {output_path}")
        return output_path

    def create_hierarchical_action_comparison(
        self,
        results: List[KeypointAnalysisResult],
        video_name: str = "video",
    ) -> Path:
        """
        Create hierarchical comparison showing action-level performance by keypoint preset.

        This visualization shows:
        1. Overall accuracy comparison across presets
        2. Per-action breakdown (stationary, walking, running)
        3. Confusion matrices for each preset
        4. Action-wise accuracy heatmap

        Args:
            results: List of KeypointAnalysisResult with performance metrics
            video_name: Name for output file

        Returns:
            Path to saved figure
        """
        if not results:
            logger.warning("No results to visualize")
            return None

        # Sort by keypoint count (descending)
        results_sorted = sorted(results, key=lambda x: x.num_keypoints, reverse=True)
        actions = ["stationary", "walking", "running"]
        action_colors = [self.ACTION_COLORS.get(a, '#95a5a6') for a in actions]

        # Create figure with hierarchical layout
        fig = plt.figure(figsize=(20, 16))

        # Main gridspec: 3 rows
        gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.2, 1.5, 1.3], hspace=0.3)

        # =====================================================================
        # Row 1: Overall Performance Overview
        # =====================================================================
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=0.25)

        preset_names = [r.preset_name.upper() for r in results_sorted]
        num_keypoints = [r.num_keypoints for r in results_sorted]
        x = np.arange(len(preset_names))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))

        # 1a. Overall Accuracy
        ax1 = fig.add_subplot(gs_top[0])
        accuracies = [r.accuracy * 100 for r in results_sorted]
        bars = ax1.bar(x, accuracies, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel("Accuracy (%)", fontsize=11)
        ax1.set_title("Overall Accuracy by Preset", fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)], fontsize=9)
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 1b. Mean F1 Score
        ax2 = fig.add_subplot(gs_top[1])
        mean_f1s = []
        for r in results_sorted:
            f1_vals = [r.f1_scores.get(a, 0) for a in actions]
            mean_f1s.append(np.mean(f1_vals) if f1_vals else 0)

        bars = ax2.bar(x, mean_f1s, color=colors, edgecolor='black', linewidth=1.2)
        ax2.set_ylabel("Mean F1 Score", fontsize=11)
        ax2.set_title("Mean F1 Score by Preset", fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)], fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', alpha=0.3)
        for bar, f1 in zip(bars, mean_f1s):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

        # 1c. Keypoint Count vs Accuracy (scatter with action-wise breakdown)
        ax3 = fig.add_subplot(gs_top[2])
        scatter = ax3.scatter(num_keypoints, accuracies, c=colors, s=200, edgecolors='black', linewidths=2, zorder=5)
        for n, a, name in zip(num_keypoints, accuracies, preset_names):
            ax3.annotate(name, (n, a), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')

        # Add trend line
        if len(num_keypoints) >= 2:
            z = np.polyfit(num_keypoints, accuracies, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(num_keypoints) - 1, max(num_keypoints) + 1, 100)
            ax3.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)

        ax3.set_xlabel("Number of Keypoints", fontsize=11)
        ax3.set_ylabel("Accuracy (%)", fontsize=11)
        ax3.set_title("Keypoint Count vs Accuracy", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(num_keypoints) + 3)

        # =====================================================================
        # Row 2: Per-Action Hierarchical Breakdown
        # =====================================================================
        gs_mid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1], wspace=0.2)

        for action_idx, action in enumerate(actions):
            ax = fig.add_subplot(gs_mid[action_idx])

            # Get F1, Precision, Recall for this action across presets
            f1_vals = [r.f1_scores.get(action, 0) for r in results_sorted]
            prec_vals = [r.precision.get(action, 0) for r in results_sorted]
            rec_vals = [r.recall.get(action, 0) for r in results_sorted]

            width = 0.25
            x_pos = np.arange(len(results_sorted))

            bars_f1 = ax.bar(x_pos - width, f1_vals, width, label='F1', color=action_colors[action_idx], alpha=0.9)
            bars_prec = ax.bar(x_pos, prec_vals, width, label='Precision', color=action_colors[action_idx], alpha=0.6)
            bars_rec = ax.bar(x_pos + width, rec_vals, width, label='Recall', color=action_colors[action_idx], alpha=0.3)

            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(f"{action.capitalize()} - F1/Precision/Recall", fontsize=11, fontweight='bold',
                        color=action_colors[action_idx])
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{p}\n({n})" for p, n in zip(preset_names, num_keypoints)], fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on F1 bars
            for bar, val in zip(bars_f1, f1_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        # =====================================================================
        # Row 3: Action-wise Performance Heatmap & Summary
        # =====================================================================
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2], wspace=0.25, width_ratios=[1.5, 1])

        # 3a. Action-wise F1 Heatmap
        ax_hm = fig.add_subplot(gs_bot[0])

        # Build heatmap data: rows=actions, cols=presets
        heatmap_data = np.zeros((len(actions), len(results_sorted)))
        for j, r in enumerate(results_sorted):
            for i, action in enumerate(actions):
                heatmap_data[i, j] = r.f1_scores.get(action, 0)

        im = ax_hm.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax_hm.set_xticks(np.arange(len(results_sorted)))
        ax_hm.set_yticks(np.arange(len(actions)))
        ax_hm.set_xticklabels([f"{p}\n({n} kp)" for p, n in zip(preset_names, num_keypoints)], fontsize=9)
        ax_hm.set_yticklabels([a.capitalize() for a in actions], fontsize=10)
        ax_hm.set_title("F1 Score Heatmap by Action & Preset", fontsize=12, fontweight='bold')

        # Add text annotations
        for i in range(len(actions)):
            for j in range(len(results_sorted)):
                val = heatmap_data[i, j]
                text_color = 'white' if val < 0.5 else 'black'
                ax_hm.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10,
                          color=text_color, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_hm, shrink=0.8)
        cbar.set_label('F1 Score', fontsize=10)

        # 3b. Summary Table
        ax_tbl = fig.add_subplot(gs_bot[1])
        ax_tbl.axis('off')

        # Build summary table
        table_data = []
        headers = ["Preset", "KP", "Acc%", "F1 (S)", "F1 (W)", "F1 (R)", "Avg F1"]

        for r in results_sorted:
            f1_s = r.f1_scores.get("stationary", 0)
            f1_w = r.f1_scores.get("walking", 0)
            f1_r = r.f1_scores.get("running", 0)
            avg_f1 = np.mean([f1_s, f1_w, f1_r])

            table_data.append([
                r.preset_name.upper(),
                str(r.num_keypoints),
                f"{r.accuracy * 100:.1f}",
                f"{f1_s:.3f}",
                f"{f1_w:.3f}",
                f"{f1_r:.3f}",
                f"{avg_f1:.3f}",
            ])

        table = ax_tbl.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['#2c3e50'] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        # Style header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(color='white', fontweight='bold')
            elif row > 0:
                # Highlight best accuracy row
                if table_data[row-1][2] == max([d[2] for d in table_data]):
                    cell.set_facecolor('#e8f5e9')

        ax_tbl.set_title("Performance Summary Table", fontsize=12, fontweight='bold', pad=20)

        # Main title
        fig.suptitle(f"Hierarchical Action Recognition Performance by Keypoint Preset\n{video_name}",
                    fontsize=14, fontweight='bold', y=0.98)

        output_path = self.output_dir / f"hierarchical_action_comparison_{video_name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor='white')
        plt.close(fig)

        logger.info(f"Saved hierarchical action comparison: {output_path}")
        return output_path

    def create_confusion_matrix_grid(
        self,
        results: List[KeypointAnalysisResult],
        video_name: str = "video",
    ) -> Path:
        """
        Create grid of confusion matrices for each keypoint preset.

        Args:
            results: List of KeypointAnalysisResult with confusion matrices
            video_name: Name for output file

        Returns:
            Path to saved figure
        """
        # Filter results with valid confusion matrices
        valid_results = [r for r in results if r.confusion_matrix is not None and r.confusion_matrix.size > 0]

        if not valid_results:
            logger.warning("No valid confusion matrices to visualize")
            return None

        n_presets = len(valid_results)
        n_cols = min(3, n_presets)
        n_rows = (n_presets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_presets == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        actions = ["stationary", "walking", "running"]

        for idx, result in enumerate(valid_results):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            cm = result.confusion_matrix
            # Normalize
            cm_normalized = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm_normalized = cm_normalized / row_sums

            im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

            n_classes = min(len(actions), cm.shape[0])
            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels([a[:4].upper() for a in actions[:n_classes]], fontsize=9)
            ax.set_yticklabels([a[:4].upper() for a in actions[:n_classes]], fontsize=9)

            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True (Full)", fontsize=10)
            ax.set_title(f"{result.preset_name.upper()} ({result.num_keypoints} kp)\nAcc: {result.accuracy*100:.1f}%",
                        fontsize=11, fontweight='bold')

            # Add text annotations
            for i in range(n_classes):
                for j in range(n_classes):
                    val = cm_normalized[i, j]
                    count = cm[i, j]
                    text_color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}\n({count})', ha='center', va='center',
                           fontsize=8, color=text_color)

        # Hide empty subplots
        for idx in range(n_presets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        fig.suptitle(f"Confusion Matrices by Keypoint Preset (vs Full as Reference)\n{video_name}",
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = self.output_dir / f"confusion_matrix_grid_{video_name}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor='white')
        plt.close(fig)

        logger.info(f"Saved confusion matrix grid: {output_path}")
        return output_path

    def create_preset_action_gifs(
        self,
        video_path: Path,
        results: List["KeypointAnalysisResult"],
        all_keypoint_names: List[str],
        keypoints: np.ndarray,
        max_frames_per_gif: int = 80,
        fps: float = 10.0,
        confidence_threshold: float = 0.3,
        max_gifs_per_combo: int = 2,
        min_segment_frames: int = 15,
    ) -> Dict[str, Dict[str, List[Path]]]:
        """
        Create GIFs for each preset × action combination.

        This generates a grid of GIFs showing how each action (stationary, walking, running)
        looks with different keypoint configurations (full, standard, mars, locomotion, minimal).

        Args:
            video_path: Path to video
            results: List of KeypointAnalysisResult for different presets
            all_keypoint_names: All keypoint names
            keypoints: Full keypoints array (frames, keypoints, 3)
            max_frames_per_gif: Maximum frames per GIF
            fps: GIF frame rate
            confidence_threshold: Confidence threshold for keypoints
            max_gifs_per_combo: Maximum GIFs per preset×action combination
            min_segment_frames: Minimum frames for action segment

        Returns:
            Nested dict: {preset_name: {action_name: [gif_paths]}}
        """
        try:
            import imageio
        except ImportError:
            logger.warning("imageio not installed - skipping preset action GIFs")
            return {}

        video_path = Path(video_path)
        video_name = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Action names mapping
        action_names = {0: "stationary", 1: "walking", 2: "running"}
        actions = ["stationary", "walking", "running"]

        # Extract action segments from the reference (first/full) result
        reference_result = results[0] if results else None
        if reference_result is None:
            logger.warning("No results provided for preset action GIFs")
            cap.release()
            return {}

        # Find continuous action segments
        segments_by_action = self._extract_action_segments(
            reference_result.action_labels,
            action_names,
            min_segment_frames
        )

        all_gif_paths = {}

        for result in results:
            preset_name = result.preset_name
            all_gif_paths[preset_name] = {}

            # Get filtered keypoints for this preset
            filtered_kp, filtered_names = self.filter_keypoints(
                keypoints, all_keypoint_names, preset_name
            )
            colors = self._generate_colors(len(filtered_names))

            for action_name in actions:
                all_gif_paths[preset_name][action_name] = []
                action_segments = segments_by_action.get(action_name, [])

                if not action_segments:
                    logger.debug(f"No segments for {preset_name}/{action_name}")
                    continue

                # Select best segments (longest ones, up to max_gifs_per_combo)
                sorted_segments = sorted(action_segments, key=lambda x: x[1] - x[0], reverse=True)

                for seg_idx, (start_frame, end_frame) in enumerate(sorted_segments[:max_gifs_per_combo]):
                    # Subsample if too long
                    segment_length = end_frame - start_frame
                    if segment_length > max_frames_per_gif:
                        step = segment_length // max_frames_per_gif
                        frame_indices = list(range(start_frame, end_frame, step))[:max_frames_per_gif]
                    else:
                        frame_indices = list(range(start_frame, min(end_frame, total_video_frames)))

                    frames = []

                    for frame_idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        if frame_idx < len(filtered_kp):
                            kp = filtered_kp[frame_idx]

                            # Draw skeleton
                            connections = self._get_skeleton_connections(filtered_names)
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

                        # Add labels
                        action_color = self.ACTION_COLORS.get(action_name, "#95a5a6")
                        # Convert hex to BGR
                        if isinstance(action_color, str):
                            action_color = tuple(int(action_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

                        # Preset label
                        label_text = f"{preset_name.upper()} ({len(filtered_names)} kp)"
                        cv2.rectangle(frame, (5, 5), (len(label_text) * 10 + 15, 30), (0, 0, 0), -1)
                        cv2.putText(frame, label_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Action label
                        cv2.rectangle(frame, (5, 35), (120, 60), action_color, -1)
                        cv2.putText(frame, action_name.upper(), (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Frame counter
                        cv2.putText(frame, f"Frame: {frame_idx}", (10, frame_height - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if frames:
                        output_name = f"preset_{preset_name}_{action_name}_{seg_idx + 1}_{video_name}.gif"
                        output_path = self.output_dir / output_name
                        imageio.mimsave(str(output_path), frames, duration=1.0/fps, loop=0)
                        all_gif_paths[preset_name][action_name].append(output_path)
                        logger.debug(f"Saved: {output_path}")

        cap.release()

        # Log summary
        total_gifs = sum(
            len(paths)
            for preset_dict in all_gif_paths.values()
            for paths in preset_dict.values()
        )
        logger.info(f"Generated {total_gifs} preset×action GIFs for {video_name}")

        return all_gif_paths

    def _extract_action_segments(
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
        unique_actions = np.unique(action_labels)

        for action_id in unique_actions:
            action_name = action_names.get(int(action_id), f"action_{action_id}")
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
                    segments[action_name].append((int(start), int(end)))

        return segments

    def create_comparison_gif(
        self,
        video_path: Path,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        presets: List[str] = ["full", "standard", "minimal"],
        max_frames: int = 100,
        fps: float = 8.0,
        confidence_threshold: float = 0.3,
    ) -> Optional[Path]:
        """
        Create side-by-side GIF comparing different presets.

        Args:
            video_path: Path to video
            keypoints: Full keypoints array
            all_keypoint_names: All keypoint names
            presets: Presets to compare
            max_frames: Maximum frames in GIF
            fps: GIF frame rate
            confidence_threshold: Confidence threshold

        Returns:
            Path to saved GIF or None
        """
        try:
            import imageio
        except ImportError:
            logger.warning("imageio not installed - skipping GIF creation")
            return None

        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(keypoints))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Subsample frames
        if total_frames > max_frames:
            step = total_frames // max_frames
            frame_indices = list(range(0, total_frames, step))[:max_frames]
        else:
            frame_indices = list(range(total_frames))

        # Prepare preset data
        preset_data = []
        for preset_name in presets:
            filtered_kp, filtered_names = self.filter_keypoints(
                keypoints, all_keypoint_names, preset_name
            )
            preset_data.append({
                "name": preset_name,
                "keypoints": filtered_kp,
                "names": filtered_names,
                "colors": self._generate_colors(len(filtered_names)),
            })

        # Panel dimensions
        num_presets = len(presets)
        panel_width = frame_width // 2
        panel_height = frame_height // 2

        combined_frames = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Create combined frame
            if num_presets <= 3:
                combined = np.zeros((panel_height, panel_width * num_presets, 3), dtype=np.uint8)
            else:
                rows = (num_presets + 1) // 2
                combined = np.zeros((panel_height * rows, panel_width * 2, 3), dtype=np.uint8)

            for i, preset in enumerate(preset_data):
                # Resize frame
                panel = cv2.resize(frame.copy(), (panel_width, panel_height))

                # Draw skeleton
                kp_frame = preset["keypoints"][frame_idx]
                connections = self._get_skeleton_connections(preset["names"])

                for idx1, idx2 in connections:
                    if idx1 < len(kp_frame) and idx2 < len(kp_frame):
                        if (kp_frame[idx1, 2] > confidence_threshold and
                            kp_frame[idx2, 2] > confidence_threshold):
                            pt1 = (int(kp_frame[idx1, 0] * panel_width / frame_width),
                                   int(kp_frame[idx1, 1] * panel_height / frame_height))
                            pt2 = (int(kp_frame[idx2, 0] * panel_width / frame_width),
                                   int(kp_frame[idx2, 1] * panel_height / frame_height))
                            cv2.line(panel, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw keypoints
                for j, color in enumerate(preset["colors"]):
                    if j < len(kp_frame):
                        x, y, conf = kp_frame[j]
                        if conf > confidence_threshold:
                            px = int(x * panel_width / frame_width)
                            py = int(y * panel_height / frame_height)
                            cv2.circle(panel, (px, py), 4, color, -1, cv2.LINE_AA)

                # Add label
                label = f"{preset['name'].upper()} ({len(preset['names'])} kp)"
                cv2.rectangle(panel, (5, 5), (len(label) * 9 + 15, 28), (0, 0, 0), -1)
                cv2.putText(panel, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Add frame counter
                cv2.putText(panel, f"Frame: {frame_idx}", (10, panel_height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                # Place in combined frame
                if num_presets <= 3:
                    combined[:, i * panel_width:(i + 1) * panel_width] = panel
                else:
                    row = i // 2
                    col = i % 2
                    combined[row * panel_height:(row + 1) * panel_height,
                            col * panel_width:(col + 1) * panel_width] = panel

            combined_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        cap.release()

        if combined_frames:
            output_path = self.output_dir / f"preset_comparison_{video_path.stem}.gif"
            imageio.mimsave(str(output_path), combined_frames, duration=1.0/fps, loop=0)
            logger.info(f"Saved comparison GIF: {output_path}")
            return output_path

        return None


# ============================================================
# Utility Functions
# ============================================================

def analyze_keypoint_preset(
    keypoints: np.ndarray,
    all_keypoint_names: List[str],
    preset_name: str,
    fps: float = 30.0,
) -> KeypointAnalysisResult:
    """
    Analyze keypoints with a specific preset configuration.

    Args:
        keypoints: Full keypoints array (frames, keypoints, 3)
        all_keypoint_names: All keypoint names
        preset_name: Name of preset to use
        fps: Video frame rate

    Returns:
        KeypointAnalysisResult object
    """
    from src.models.action_classifier import UnifiedActionClassifier

    visualizer = KeypointVisualizer(output_dir=".")  # temp
    filtered_kp, filtered_names = visualizer.filter_keypoints(
        keypoints, all_keypoint_names, preset_name
    )

    # Compute trajectory (from center or mean)
    center_kps = ["mouse_center", "back_middle", "neck"]
    trajectory = None

    for center_name in center_kps:
        if center_name in filtered_names:
            idx = filtered_names.index(center_name)
            trajectory = filtered_kp[:, idx, :2]
            break

    if trajectory is None:
        # Use mean of all valid keypoints
        valid_mask = filtered_kp[:, :, 2] > 0.3
        trajectory = np.zeros((len(filtered_kp), 2))
        for i in range(len(filtered_kp)):
            valid = valid_mask[i]
            if valid.any():
                trajectory[i] = filtered_kp[i, valid, :2].mean(axis=0)

    # Compute velocity
    velocity = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
    velocity = np.concatenate([[0], velocity])

    # Action classification
    classifier = UnifiedActionClassifier(species="mouse", fps=fps)
    metrics = classifier.analyze(filtered_kp, filtered_names)

    # Compute confidence and detection stats
    valid_conf = filtered_kp[:, :, 2] > 0.3
    mean_confidence = filtered_kp[:, :, 2][valid_conf].mean() if valid_conf.any() else 0
    detection_rate = valid_conf.any(axis=1).mean()

    return KeypointAnalysisResult(
        preset_name=preset_name,
        num_keypoints=len(filtered_names),
        keypoint_names=filtered_names,
        keypoints=filtered_kp,
        trajectory=trajectory,
        velocity=velocity,
        action_labels=metrics.action_labels,
        action_distribution=metrics.action_summary.get("actions", {}),
        mean_confidence=mean_confidence,
        detection_rate=detection_rate,
    )


def compare_presets(
    keypoints: np.ndarray,
    all_keypoint_names: List[str],
    presets: List[str] = ["full", "standard", "minimal"],
    fps: float = 30.0,
) -> List[KeypointAnalysisResult]:
    """
    Compare multiple keypoint presets.

    Args:
        keypoints: Full keypoints array
        all_keypoint_names: All keypoint names
        presets: Presets to compare
        fps: Video frame rate

    Returns:
        List of KeypointAnalysisResult for each preset
    """
    results = []
    for preset_name in presets:
        result = analyze_keypoint_preset(keypoints, all_keypoint_names, preset_name, fps)
        results.append(result)
    return results


ACTION_CLASS_NAMES = {0: "stationary", 1: "walking", 2: "running"}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float], np.ndarray]:
    """
    Compute classification metrics (accuracy, F1, precision, recall, confusion matrix).

    Args:
        y_true: Ground truth labels (reference)
        y_pred: Predicted labels
        class_names: Mapping from label id to class name

    Returns:
        Tuple of (accuracy, f1_scores, precision, recall, confusion_matrix)
    """
    if class_names is None:
        class_names = ACTION_CLASS_NAMES

    # Get all unique labels
    all_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    # Compute confusion matrix
    conf_matrix = np.zeros((n_labels, n_labels), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1

    # Compute accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0

    # Compute per-class metrics
    f1_scores = {}
    precision = {}
    recall = {}

    for label in all_labels:
        idx = label_to_idx[label]
        tp = conf_matrix[idx, idx]
        fp = np.sum(conf_matrix[:, idx]) - tp
        fn = np.sum(conf_matrix[idx, :]) - tp

        # Get class name (use string label if not in mapping)
        class_name = class_names.get(int(label), str(label))

        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision[class_name] = prec

        # Recall
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall[class_name] = rec

        # F1
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores[class_name] = f1

    return accuracy, f1_scores, precision, recall, conf_matrix


def compare_presets_with_metrics(
    keypoints: np.ndarray,
    all_keypoint_names: List[str],
    presets: List[str] = ["full", "standard", "mars", "locomotion", "minimal"],
    fps: float = 30.0,
    reference_preset: str = "full",
) -> Tuple[List[KeypointAnalysisResult], KeypointComparisonMetrics]:
    """
    Compare multiple keypoint presets with full accuracy/F1 metrics.

    Uses the reference preset (typically "full") as ground truth for
    computing accuracy, F1, precision, and recall for other presets.

    Args:
        keypoints: Full keypoints array (frames, keypoints, 3)
        all_keypoint_names: All keypoint names
        presets: Presets to compare
        fps: Video frame rate
        reference_preset: Preset to use as ground truth (default: "full")

    Returns:
        Tuple of (list of KeypointAnalysisResult, KeypointComparisonMetrics)
    """
    # Ensure reference preset is first
    if reference_preset in presets:
        presets = [reference_preset] + [p for p in presets if p != reference_preset]
    else:
        presets = [reference_preset] + presets

    # Get base results first
    results = []
    for preset_name in presets:
        result = analyze_keypoint_preset(keypoints, all_keypoint_names, preset_name, fps)
        results.append(result)

    # Get reference result
    reference_result = results[0]
    reference_labels = reference_result.action_labels

    # Compute metrics for each preset vs reference
    trajectory_agreement = {}
    action_agreement = {}
    velocity_correlation = {}
    accuracy_by_preset = {}
    f1_by_preset = {}
    accuracy_drop_from_full = {}

    for result in results:
        preset_name = result.preset_name

        # Compute agreement with reference
        if len(result.action_labels) == len(reference_labels):
            agreement = np.mean(result.action_labels == reference_labels)
            action_agreement[preset_name] = agreement

            # Compute full classification metrics
            accuracy, f1_scores, precision, recall, conf_matrix = compute_classification_metrics(
                reference_labels, result.action_labels
            )

            # Update result with metrics
            result.accuracy = accuracy
            result.f1_scores = f1_scores
            result.precision = precision
            result.recall = recall
            result.confusion_matrix = conf_matrix
            result.agreement_with_full = agreement

            # Store for comparison metrics
            accuracy_by_preset[preset_name] = accuracy
            f1_by_preset[preset_name] = f1_scores

            # Compute accuracy drop from full
            if preset_name != reference_preset:
                accuracy_drop_from_full[preset_name] = 1.0 - accuracy

        # Trajectory agreement (using normalized distance)
        if len(result.trajectory) == len(reference_result.trajectory):
            traj_diff = np.sqrt(np.sum((result.trajectory - reference_result.trajectory) ** 2, axis=1))
            # Normalize by image size (assume 640x480)
            normalized_diff = traj_diff / np.sqrt(640**2 + 480**2)
            trajectory_agreement[preset_name] = 1.0 - np.mean(normalized_diff)

        # Velocity correlation
        if len(result.velocity) == len(reference_result.velocity):
            if np.std(result.velocity) > 0 and np.std(reference_result.velocity) > 0:
                corr = np.corrcoef(result.velocity, reference_result.velocity)[0, 1]
                velocity_correlation[preset_name] = corr if not np.isnan(corr) else 0.0
            else:
                velocity_correlation[preset_name] = 1.0 if np.allclose(result.velocity, reference_result.velocity) else 0.0

    # Find best preset by accuracy (excluding full)
    non_full_accuracy = {k: v for k, v in accuracy_by_preset.items() if k != reference_preset}
    best_preset = max(non_full_accuracy.keys(), key=lambda k: non_full_accuracy[k]) if non_full_accuracy else reference_preset

    comparison_metrics = KeypointComparisonMetrics(
        presets_compared=presets,
        reference_preset=reference_preset,
        trajectory_agreement=trajectory_agreement,
        action_agreement=action_agreement,
        velocity_correlation=velocity_correlation,
        accuracy_by_preset=accuracy_by_preset,
        f1_by_preset=f1_by_preset,
        best_preset_by_accuracy=best_preset,
        accuracy_drop_from_full=accuracy_drop_from_full,
    )

    logger.info(f"Computed metrics for {len(results)} presets vs reference '{reference_preset}'")
    for preset_name, acc in accuracy_by_preset.items():
        logger.info(f"  {preset_name}: accuracy={acc:.3f}, agreement={action_agreement.get(preset_name, 0):.3f}")

    return results, comparison_metrics


# ============================================================
# Ground Truth Comparison GIF Generator
# ============================================================

class GroundTruthComparisonGifGenerator:
    """
    Generate GIFs that compare predictions with ground truth labels.
    Shows per-frame prediction correctness for each action class.
    """

    ACTION_COLORS_BGR = {
        "stationary": (219, 152, 52),   # Blue
        "walking": (113, 204, 46),      # Green
        "running": (60, 76, 231),       # Red
        0: (219, 152, 52),
        1: (113, 204, 46),
        2: (60, 76, 231),
    }

    CORRECT_COLOR = (0, 255, 0)     # Green
    INCORRECT_COLOR = (0, 0, 255)   # Red

    ACTION_NAMES = {0: "stationary", 1: "walking", 2: "running"}

    def __init__(self, output_dir: Union[str, Path]):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_preset_comparison_gif_with_gt(
        self,
        video_path: Path,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        preset_predictions: Dict[str, np.ndarray],
        preset_num_keypoints: Dict[str, int],
        ground_truth: np.ndarray,
        max_frames: int = 100,
        fps: float = 8.0,
        confidence_threshold: float = 0.3,
    ) -> Optional[Path]:
        """
        Create GIF comparing predictions across presets with ground truth overlay.

        Shows each preset's prediction vs ground truth with correct/incorrect indicator.

        Args:
            video_path: Path to video file
            keypoints: Full keypoints array (frames, keypoints, 3)
            all_keypoint_names: All keypoint names
            preset_predictions: Dict mapping preset_name -> predicted labels array
            preset_num_keypoints: Dict mapping preset_name -> number of keypoints
            ground_truth: Ground truth action labels
            max_frames: Maximum frames in GIF
            fps: GIF frame rate
            confidence_threshold: Keypoint confidence threshold

        Returns:
            Path to saved GIF or None if failed
        """
        try:
            import imageio
        except ImportError:
            logger.warning("imageio not installed - skipping GT comparison GIF")
            return None

        video_path = Path(video_path)
        video_name = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = min(total_video_frames, len(keypoints), len(ground_truth))

        # Subsample frames
        if total_frames > max_frames:
            step = total_frames // max_frames
            frame_indices = list(range(0, total_frames, step))[:max_frames]
        else:
            frame_indices = list(range(total_frames))

        presets = list(preset_predictions.keys())
        num_presets = len(presets)

        # Calculate panel dimensions
        panel_width = frame_width // 2
        panel_height = frame_height // 2

        # Layout: grid based on number of presets
        if num_presets <= 3:
            n_cols = num_presets
            n_rows = 1
        else:
            n_cols = 3
            n_rows = (num_presets + 2) // 3

        combined_width = panel_width * n_cols
        combined_height = panel_height * n_rows

        # Filter keypoints for each preset
        visualizer = KeypointVisualizer(output_dir=self.output_dir)
        preset_data = {}
        for preset_name in presets:
            filtered_kp, filtered_names = visualizer.filter_keypoints(
                keypoints, all_keypoint_names, preset_name
            )
            preset_data[preset_name] = {
                "keypoints": filtered_kp,
                "names": filtered_names,
                "colors": visualizer._generate_colors(len(filtered_names)),
                "num_kp": preset_num_keypoints.get(preset_name, len(filtered_names)),
            }

        combined_frames = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Get ground truth for this frame
            gt_label = ground_truth[frame_idx] if frame_idx < len(ground_truth) else 0
            gt_name = self.ACTION_NAMES.get(int(gt_label), f"class_{gt_label}")

            combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            for preset_idx, preset_name in enumerate(presets):
                row = preset_idx // n_cols
                col = preset_idx % n_cols

                # Resize frame for panel
                panel = cv2.resize(frame.copy(), (panel_width, panel_height))

                # Get prediction for this preset
                pred_labels = preset_predictions[preset_name]
                pred_label = pred_labels[frame_idx] if frame_idx < len(pred_labels) else 0
                pred_name = self.ACTION_NAMES.get(int(pred_label), f"class_{pred_label}")

                # Check correctness
                is_correct = (pred_label == gt_label)

                # Draw keypoints
                data = preset_data[preset_name]
                if frame_idx < len(data["keypoints"]):
                    kp_frame = data["keypoints"][frame_idx]
                    connections = visualizer._get_skeleton_connections(data["names"])

                    # Draw skeleton
                    for i, j in connections:
                        if i < len(kp_frame) and j < len(kp_frame):
                            if kp_frame[i, 2] > confidence_threshold and kp_frame[j, 2] > confidence_threshold:
                                pt1 = (int(kp_frame[i, 0] * panel_width / frame_width),
                                       int(kp_frame[i, 1] * panel_height / frame_height))
                                pt2 = (int(kp_frame[j, 0] * panel_width / frame_width),
                                       int(kp_frame[j, 1] * panel_height / frame_height))
                                cv2.line(panel, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

                    # Draw keypoints
                    for kp_idx, color in enumerate(data["colors"]):
                        if kp_idx < len(kp_frame):
                            x, y, conf = kp_frame[kp_idx]
                            if conf > confidence_threshold:
                                px = int(x * panel_width / frame_width)
                                py = int(y * panel_height / frame_height)
                                cv2.circle(panel, (px, py), 4, color, -1, cv2.LINE_AA)

                # Add preset info bar (top)
                preset_label = f"{preset_name.upper()} ({data['num_kp']} kp)"
                cv2.rectangle(panel, (0, 0), (panel_width, 25), (30, 30, 30), -1)
                cv2.putText(panel, preset_label, (5, 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # Add prediction vs ground truth bar (bottom)
                # Background color based on correctness
                bar_color = self.CORRECT_COLOR if is_correct else self.INCORRECT_COLOR
                cv2.rectangle(panel, (0, panel_height - 50), (panel_width, panel_height), (50, 50, 50), -1)

                # Prediction label with action color
                pred_color = self.ACTION_COLORS_BGR.get(pred_name, (150, 150, 150))
                cv2.rectangle(panel, (5, panel_height - 45), (panel_width // 2 - 5, panel_height - 25), pred_color, -1)
                cv2.putText(panel, f"Pred: {pred_name[:4].upper()}", (10, panel_height - 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Ground truth label with action color
                gt_color = self.ACTION_COLORS_BGR.get(gt_name, (150, 150, 150))
                cv2.rectangle(panel, (panel_width // 2 + 5, panel_height - 45),
                             (panel_width - 5, panel_height - 25), gt_color, -1)
                cv2.putText(panel, f"GT: {gt_name[:4].upper()}", (panel_width // 2 + 10, panel_height - 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Correctness indicator
                status_text = "CORRECT" if is_correct else "WRONG"
                cv2.rectangle(panel, (5, panel_height - 22), (panel_width - 5, panel_height - 5), bar_color, -1)
                cv2.putText(panel, status_text, (panel_width // 2 - 30, panel_height - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Place panel in combined frame
                y_start = row * panel_height
                y_end = (row + 1) * panel_height
                x_start = col * panel_width
                x_end = (col + 1) * panel_width
                combined[y_start:y_end, x_start:x_end] = panel

            # Add frame counter at bottom
            cv2.putText(combined, f"Frame: {frame_idx}", (10, combined_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            combined_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        cap.release()

        if combined_frames:
            output_path = self.output_dir / f"gt_comparison_{video_name}.gif"
            imageio.mimsave(str(output_path), combined_frames, duration=1.0/fps, loop=0)
            logger.info(f"Saved GT comparison GIF: {output_path} ({len(combined_frames)} frames)")
            return output_path

        return None

    def create_per_class_accuracy_gif(
        self,
        video_path: Path,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        preset_name: str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        max_frames: int = 100,
        fps: float = 8.0,
        confidence_threshold: float = 0.3,
    ) -> Optional[Path]:
        """
        Create GIF for a single preset showing per-class prediction accuracy.

        Shows running accuracy per action class as overlay on video.

        Args:
            video_path: Path to video file
            keypoints: Full keypoints array
            all_keypoint_names: All keypoint names
            preset_name: Name of the keypoint preset
            predictions: Predicted action labels
            ground_truth: Ground truth action labels
            max_frames: Maximum frames in GIF
            fps: GIF frame rate
            confidence_threshold: Keypoint confidence threshold

        Returns:
            Path to saved GIF or None if failed
        """
        try:
            import imageio
        except ImportError:
            logger.warning("imageio not installed - skipping per-class accuracy GIF")
            return None

        video_path = Path(video_path)
        video_name = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = min(total_video_frames, len(keypoints), len(ground_truth), len(predictions))

        # Subsample frames
        if total_frames > max_frames:
            step = total_frames // max_frames
            frame_indices = list(range(0, total_frames, step))[:max_frames]
        else:
            frame_indices = list(range(total_frames))

        # Filter keypoints
        visualizer = KeypointVisualizer(output_dir=self.output_dir)
        filtered_kp, filtered_names = visualizer.filter_keypoints(
            keypoints, all_keypoint_names, preset_name
        )
        colors = visualizer._generate_colors(len(filtered_names))

        # Track per-class accuracy over time
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}

        frames_out = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gt_label = ground_truth[frame_idx]
            pred_label = predictions[frame_idx]

            # Update running accuracy
            class_total[gt_label] = class_total.get(gt_label, 0) + 1
            if gt_label == pred_label:
                class_correct[gt_label] = class_correct.get(gt_label, 0) + 1

            # Draw keypoints
            if frame_idx < len(filtered_kp):
                kp = filtered_kp[frame_idx]
                connections = visualizer._get_skeleton_connections(filtered_names)

                for i, j in connections:
                    if i < len(kp) and j < len(kp):
                        if kp[i, 2] > confidence_threshold and kp[j, 2] > confidence_threshold:
                            pt1 = (int(kp[i, 0]), int(kp[i, 1]))
                            pt2 = (int(kp[j, 0]), int(kp[j, 1]))
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

                for idx, color in enumerate(colors):
                    if idx < len(kp):
                        x, y, conf = kp[idx]
                        if conf > confidence_threshold:
                            cv2.circle(frame, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)

            # Prediction vs GT indicator
            is_correct = (gt_label == pred_label)
            indicator_color = self.CORRECT_COLOR if is_correct else self.INCORRECT_COLOR

            gt_name = self.ACTION_NAMES.get(int(gt_label), "?")
            pred_name = self.ACTION_NAMES.get(int(pred_label), "?")

            # Top info bar
            cv2.rectangle(frame, (0, 0), (frame_width, 60), (30, 30, 30), -1)
            cv2.putText(frame, f"{preset_name.upper()} ({len(filtered_names)} kp)",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Pred: {pred_name.upper()} | GT: {gt_name.upper()}",
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, indicator_color, 1)

            # Per-class accuracy bar (right side)
            bar_x = frame_width - 180
            bar_y = 70
            bar_width = 170
            bar_height = 90

            cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + bar_width + 5, bar_y + bar_height + 5),
                         (30, 30, 30), -1)
            cv2.putText(frame, "Per-Class Accuracy", (bar_x, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for i, (cls_id, cls_name) in enumerate(self.ACTION_NAMES.items()):
                y_offset = bar_y + 30 + i * 20
                total = class_total.get(cls_id, 0)
                correct = class_correct.get(cls_id, 0)
                acc = correct / total if total > 0 else 0.0

                # Color for this class
                cls_color = self.ACTION_COLORS_BGR.get(cls_name, (150, 150, 150))

                # Accuracy bar
                acc_bar_width = int(80 * acc)
                cv2.rectangle(frame, (bar_x + 70, y_offset - 10), (bar_x + 70 + 80, y_offset + 5), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x + 70, y_offset - 10), (bar_x + 70 + acc_bar_width, y_offset + 5), cls_color, -1)

                cv2.putText(frame, f"{cls_name[:4]}:", (bar_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, cls_color, 1)
                cv2.putText(frame, f"{acc*100:.0f}%", (bar_x + 155, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Frame counter
            cv2.putText(frame, f"Frame: {frame_idx}", (10, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            frames_out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        if frames_out:
            output_path = self.output_dir / f"per_class_acc_{preset_name}_{video_name}.gif"
            imageio.mimsave(str(output_path), frames_out, duration=1.0/fps, loop=0)
            logger.info(f"Saved per-class accuracy GIF: {output_path}")
            return output_path

        return None

    def create_all_preset_gt_gifs(
        self,
        video_path: Path,
        keypoints: np.ndarray,
        all_keypoint_names: List[str],
        results: List[KeypointAnalysisResult],
        ground_truth: np.ndarray,
        max_frames: int = 80,
        fps: float = 8.0,
    ) -> Dict[str, Path]:
        """
        Create GT comparison GIFs for all presets.

        Args:
            video_path: Path to video
            keypoints: Full keypoints array
            all_keypoint_names: All keypoint names
            results: List of KeypointAnalysisResult for different presets
            ground_truth: Ground truth action labels
            max_frames: Maximum frames per GIF
            fps: GIF frame rate

        Returns:
            Dict mapping preset_name -> gif_path
        """
        gif_paths = {}

        # Create comparison GIF across all presets
        preset_predictions = {}
        preset_num_kp = {}
        for result in results:
            preset_predictions[result.preset_name] = result.action_labels
            preset_num_kp[result.preset_name] = result.num_keypoints

        comparison_gif = self.create_preset_comparison_gif_with_gt(
            video_path=video_path,
            keypoints=keypoints,
            all_keypoint_names=all_keypoint_names,
            preset_predictions=preset_predictions,
            preset_num_keypoints=preset_num_kp,
            ground_truth=ground_truth,
            max_frames=max_frames,
            fps=fps,
        )
        if comparison_gif:
            gif_paths["comparison"] = comparison_gif

        # Create per-preset accuracy GIFs
        for result in results:
            gif_path = self.create_per_class_accuracy_gif(
                video_path=video_path,
                keypoints=keypoints,
                all_keypoint_names=all_keypoint_names,
                preset_name=result.preset_name,
                predictions=result.action_labels,
                ground_truth=ground_truth,
                max_frames=max_frames,
                fps=fps,
            )
            if gif_path:
                gif_paths[result.preset_name] = gif_path

        return gif_paths
