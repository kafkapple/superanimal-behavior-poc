"""
Comprehensive Dashboard Generator for SuperAnimal Behavior Analysis.

Generates interactive HTML dashboards with:
- Keypoint comparison across presets
- Cross-species comparison
- Action recognition performance metrics
- Train/validation comparison
- GIF visualizations embedded in reports
"""
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO

logger = logging.getLogger(__name__)

# Optional cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2 not available - some dashboard features may be limited")


# ============================================================
# Data Classes for Structured Results
# ============================================================

@dataclass
class KeypointPresetResult:
    """Results for a single keypoint preset."""
    preset_name: str
    num_keypoints: int
    keypoint_names: List[str]
    action_accuracy: float = 0.0
    action_distribution: Dict[str, float] = field(default_factory=dict)
    mean_confidence: float = 0.0
    trajectory_distance: float = 0.0
    velocity_stats: Dict[str, float] = field(default_factory=dict)
    # New metrics for accuracy/F1 comparison
    f1_scores: Dict[str, float] = field(default_factory=dict)
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    agreement_with_full: float = 0.0  # Agreement rate with full preset


@dataclass
class SpeciesResult:
    """Results for a single species."""
    species_name: str
    model_type: str
    num_frames: int
    body_size_stats: Dict[str, float] = field(default_factory=dict)
    action_distribution: Dict[str, float] = field(default_factory=dict)
    velocity_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActionRecognitionResult:
    """Action recognition performance results."""
    model_name: str
    accuracy: float
    f1_scores: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    consistency_score: float = 0.0
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """Summary of a complete experiment."""
    experiment_name: str
    timestamp: str
    total_frames: int
    species: List[str]
    presets_tested: List[str]
    keypoint_results: List[KeypointPresetResult] = field(default_factory=list)
    species_results: List[SpeciesResult] = field(default_factory=list)
    action_results: List[ActionRecognitionResult] = field(default_factory=list)
    output_files: Dict[str, str] = field(default_factory=dict)


# ============================================================
# Dashboard HTML Templates
# ============================================================

DASHBOARD_CSS = """
<style>
    :root {
        --primary: #3498db;
        --secondary: #2ecc71;
        --warning: #f39c12;
        --danger: #e74c3c;
        --dark: #2c3e50;
        --light: #ecf0f1;
        --gray: #95a5a6;
    }

    * { box-sizing: border-box; }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
        color: var(--dark);
    }

    .dashboard-container {
        max-width: 1600px;
        margin: 0 auto;
        padding: 20px;
    }

    .dashboard-header {
        background: linear-gradient(135deg, var(--dark) 0%, #34495e 100%);
        color: white;
        padding: 30px 40px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .dashboard-header h1 {
        margin: 0 0 10px 0;
        font-size: 2.5em;
        font-weight: 700;
    }

    .dashboard-header .subtitle {
        color: var(--light);
        font-size: 1.1em;
        opacity: 0.9;
    }

    .dashboard-header .meta {
        display: flex;
        gap: 30px;
        margin-top: 20px;
        flex-wrap: wrap;
    }

    .meta-item {
        background: rgba(255,255,255,0.1);
        padding: 10px 20px;
        border-radius: 8px;
    }

    .meta-item .label {
        font-size: 0.85em;
        opacity: 0.8;
    }

    .meta-item .value {
        font-size: 1.4em;
        font-weight: 600;
    }

    .section {
        background: white;
        border-radius: 15px;
        padding: 25px 30px;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }

    .section h2 {
        color: var(--dark);
        border-bottom: 3px solid var(--primary);
        padding-bottom: 10px;
        margin-top: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section h2 .icon {
        font-size: 1.2em;
    }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }

    .metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card.orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
    .metric-card.red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .metric-card.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }

    .metric-card .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-card .metric-label {
        font-size: 0.95em;
        opacity: 0.9;
        margin-top: 8px;
    }

    .metric-card .metric-sublabel {
        font-size: 0.85em;
        opacity: 0.8;
        margin-top: 5px;
        font-weight: 500;
    }

    .metric-card.purple { background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); }

    .section h3 {
        color: var(--dark);
        margin-top: 25px;
        margin-bottom: 15px;
        font-size: 1.2em;
    }

    .section .description {
        color: var(--gray);
        font-size: 0.95em;
        margin-bottom: 15px;
        line-height: 1.5;
    }

    .section .legend {
        color: var(--gray);
        font-size: 0.85em;
        margin-top: 10px;
        font-style: italic;
    }

    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 0.95em;
    }

    .comparison-table th {
        background: var(--dark);
        color: white;
        padding: 15px;
        text-align: left;
        font-weight: 600;
    }

    .comparison-table td {
        padding: 12px 15px;
        border-bottom: 1px solid var(--light);
    }

    .comparison-table tr:hover {
        background: #f8f9fa;
    }

    .comparison-table .highlight {
        background: #e8f5e9;
        font-weight: 600;
    }

    .comparison-table .best {
        color: var(--secondary);
        font-weight: 700;
    }

    .bar-chart {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin: 20px 0;
    }

    .bar-item {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .bar-label {
        width: 120px;
        font-weight: 500;
        text-align: right;
    }

    .bar-container {
        flex: 1;
        background: var(--light);
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: 600;
        font-size: 0.85em;
        transition: width 0.5s ease;
    }

    .bar-fill.stationary { background: linear-gradient(90deg, #3498db, #2980b9); }
    .bar-fill.walking { background: linear-gradient(90deg, #2ecc71, #27ae60); }
    .bar-fill.running { background: linear-gradient(90deg, #e74c3c, #c0392b); }

    .gif-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    .gif-card {
        background: var(--light);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
    }

    .gif-card img {
        width: 100%;
        display: block;
    }

    .gif-card .gif-info {
        padding: 15px;
    }

    .gif-card .gif-title {
        font-weight: 600;
        color: var(--dark);
        margin-bottom: 5px;
    }

    .gif-card .gif-meta {
        font-size: 0.85em;
        color: var(--gray);
    }

    .plot-container {
        text-align: center;
        margin: 20px 0;
    }

    .plot-container img {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }

    .plot-title {
        font-weight: 600;
        color: var(--dark);
        margin-bottom: 15px;
        font-size: 1.1em;
    }

    .flex-row {
        display: flex;
        gap: 25px;
        flex-wrap: wrap;
    }

    .flex-col {
        flex: 1;
        min-width: 300px;
    }

    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
    }

    .badge.primary { background: var(--primary); color: white; }
    .badge.success { background: var(--secondary); color: white; }
    .badge.warning { background: var(--warning); color: white; }
    .badge.danger { background: var(--danger); color: white; }

    .progress-ring {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto;
    }

    .tabs {
        display: flex;
        gap: 5px;
        margin-bottom: 20px;
        border-bottom: 2px solid var(--light);
        padding-bottom: 10px;
    }

    .tab {
        padding: 10px 20px;
        background: transparent;
        border: none;
        cursor: pointer;
        font-size: 1em;
        color: var(--gray);
        border-radius: 8px 8px 0 0;
        transition: all 0.3s;
    }

    .tab:hover {
        background: var(--light);
        color: var(--dark);
    }

    .tab.active {
        background: var(--primary);
        color: white;
    }

    .tab-content {
        display: none;
    }

    .tab-content.active {
        display: block;
    }

    footer {
        text-align: center;
        color: var(--gray);
        padding: 30px;
        font-size: 0.9em;
    }

    footer a {
        color: var(--primary);
        text-decoration: none;
    }

    @media (max-width: 768px) {
        .dashboard-header { padding: 20px; }
        .dashboard-header h1 { font-size: 1.8em; }
        .section { padding: 15px 20px; }
        .flex-row { flex-direction: column; }
    }

    /* Lightbox Modal Styles */
    .lightbox-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        z-index: 10000;
        justify-content: center;
        align-items: center;
        cursor: zoom-out;
    }

    .lightbox-overlay.active {
        display: flex;
    }

    .lightbox-content {
        max-width: 95vw;
        max-height: 95vh;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 10px 50px rgba(0,0,0,0.5);
    }

    .lightbox-close {
        position: fixed;
        top: 20px;
        right: 30px;
        font-size: 40px;
        color: white;
        cursor: pointer;
        z-index: 10001;
        opacity: 0.8;
        transition: opacity 0.2s;
    }

    .lightbox-close:hover {
        opacity: 1;
    }

    .lightbox-caption {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        font-size: 1.1em;
        text-align: center;
        background: rgba(0,0,0,0.6);
        padding: 10px 25px;
        border-radius: 8px;
        max-width: 80%;
    }

    /* Clickable images/gifs */
    .zoomable {
        cursor: zoom-in;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .zoomable:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    .gif-card .zoomable {
        width: 100%;
        display: block;
    }

    .plot-container .zoomable {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
</style>
"""

DASHBOARD_JS = """
<script>
    // Lightbox functionality
    function openLightbox(imgSrc, caption) {
        const overlay = document.getElementById('lightbox-overlay');
        const content = document.getElementById('lightbox-content');
        const captionEl = document.getElementById('lightbox-caption');

        content.src = imgSrc;
        captionEl.textContent = caption || '';
        captionEl.style.display = caption ? 'block' : 'none';
        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeLightbox() {
        const overlay = document.getElementById('lightbox-overlay');
        overlay.classList.remove('active');
        document.body.style.overflow = 'auto';
    }

    // Close on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeLightbox();
        }
    });

    // Initialize zoomable images
    document.addEventListener('DOMContentLoaded', function() {
        // Add click handlers to all zoomable images
        document.querySelectorAll('.zoomable').forEach(function(img) {
            img.addEventListener('click', function() {
                const caption = this.getAttribute('data-caption') || this.alt || '';
                openLightbox(this.src, caption);
            });
        });
    });

    function showTab(tabId, button) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        // Deactivate all tabs
        document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
        // Show selected tab content
        document.getElementById(tabId).classList.add('active');
        // Activate clicked tab
        button.classList.add('active');
    }

    // Animate bars on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.width = entry.target.dataset.width;
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.bar-fill').forEach(bar => {
        const width = bar.style.width;
        bar.dataset.width = width;
        bar.style.width = '0%';
        observer.observe(bar);
    });
</script>
"""


# ============================================================
# Dashboard Generator
# ============================================================

class DashboardGenerator:
    """Generate comprehensive HTML dashboards for experiment results."""

    ACTION_COLORS = {
        "stationary": "#3498db",
        "walking": "#2ecc71",
        "running": "#e74c3c",
        "resting": "#3498db",
        "grooming": "#9b59b6",
        "unknown": "#95a5a6",
    }

    def __init__(self, output_dir: Union[str, Path]):
        """Initialize dashboard generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _image_to_base64(self, image_path: Path) -> str:
        """Convert image file to base64 string for embedding."""
        if not image_path or not Path(image_path).exists():
            return ""

        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()

        suffix = Path(image_path).suffix.lower()
        mime_types = {".gif": "image/gif", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime = mime_types.get(suffix, "image/png")

        return f"data:{mime};base64,{data}"

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{data}"

    def _create_comparison_bar_chart(
        self,
        data: Dict[str, Dict[str, float]],
        title: str,
        ylabel: str = "Value",
    ) -> str:
        """Create a comparison bar chart and return base64."""
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = list(data.keys())
        if not categories:
            plt.close(fig)
            return ""

        x = np.arange(len(categories))

        # Get all unique metrics
        all_metrics = set()
        for cat_data in data.values():
            all_metrics.update(cat_data.keys())
        metrics = sorted(list(all_metrics))

        width = 0.8 / len(metrics) if metrics else 0.8
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

        for i, metric in enumerate(metrics):
            values = [data[cat].get(metric, 0) for cat in categories]
            ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])

        ax.set_xlabel("Category")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_heatmap(
        self,
        matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str,
    ) -> str:
        """Create a heatmap visualization."""
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(matrix, cmap='Blues')

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = matrix[i, j]
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_keypoint_comparison_section(
        self,
        keypoint_results: List[KeypointPresetResult],
    ) -> str:
        """Generate HTML section for keypoint preset comparison."""
        if not keypoint_results:
            return ""

        # Create comparison chart
        # Convert action_distribution to simple percentage values
        action_data = {}
        for result in keypoint_results:
            preset_key = f"{result.preset_name}\n({result.num_keypoints} kp)"
            dist = result.action_distribution
            # Handle nested dict format (e.g., {"stationary": {"percentage": 10}})
            # or simple format (e.g., {"stationary": 10})
            simple_dist = {}
            for action, val in dist.items():
                if isinstance(val, dict):
                    simple_dist[action] = val.get("percentage", 0)
                else:
                    simple_dist[action] = val
            action_data[preset_key] = simple_dist

        chart_b64 = self._create_comparison_bar_chart(
            action_data,
            "Action Distribution by Keypoint Preset",
            "Percentage (%)"
        )

        # Create accuracy/F1 performance chart
        perf_chart_b64 = self._create_keypoint_performance_chart(keypoint_results)

        # Build HTML
        html = f'''
        <div class="section">
            <h2><span class="icon">🎯</span> Keypoint Preset Comparison</h2>

            <div class="card-grid">
        '''

        colors = ["blue", "green", "orange", "red", "purple"]
        for i, result in enumerate(keypoint_results):
            color = colors[i % len(colors)]
            acc_display = f"{result.action_accuracy * 100:.1f}%" if result.action_accuracy > 0 else "N/A"
            html += f'''
                <div class="metric-card {color}">
                    <div class="metric-value">{result.num_keypoints}</div>
                    <div class="metric-label">{result.preset_name.upper()} Keypoints</div>
                    <div class="metric-sublabel">Accuracy: {acc_display}</div>
                </div>
            '''

        html += '</div>'

        # Performance metrics table (Accuracy/F1)
        html += '''
            <h3>Action Recognition Performance by Keypoint Count</h3>
            <p class="description">Comparing action classification accuracy using different keypoint configurations.
            Full preset (27 keypoints) serves as reference ground truth.</p>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Preset</th>
                        <th>Keypoints</th>
                        <th>Accuracy</th>
                        <th>Agreement</th>
                        <th>F1 (Stationary)</th>
                        <th>F1 (Walking)</th>
                        <th>F1 (Running)</th>
                        <th>Mean F1</th>
                    </tr>
                </thead>
                <tbody>
        '''

        # Find best values for highlighting
        valid_results = [r for r in keypoint_results if r.action_accuracy > 0]
        best_acc = max((r.action_accuracy for r in valid_results), default=0)
        best_agree = max((r.agreement_with_full for r in valid_results), default=0)

        for result in keypoint_results:
            acc_class = "best" if result.action_accuracy == best_acc and result.action_accuracy > 0 else ""
            agree_class = "best" if result.agreement_with_full == best_agree and result.agreement_with_full > 0 else ""

            # Calculate mean F1
            f1_values = list(result.f1_scores.values()) if result.f1_scores else []
            mean_f1 = np.mean(f1_values) if f1_values else 0

            # Accuracy drop indicator
            acc_pct = result.action_accuracy * 100
            if acc_pct >= 95:
                acc_indicator = "🟢"
            elif acc_pct >= 85:
                acc_indicator = "🟡"
            else:
                acc_indicator = "🔴"

            html += f'''
                <tr>
                    <td><strong>{result.preset_name.upper()}</strong></td>
                    <td>{result.num_keypoints}</td>
                    <td class="{acc_class}">{acc_indicator} {result.action_accuracy * 100:.1f}%</td>
                    <td class="{agree_class}">{result.agreement_with_full * 100:.1f}%</td>
                    <td>{result.f1_scores.get("stationary", 0):.3f}</td>
                    <td>{result.f1_scores.get("walking", 0):.3f}</td>
                    <td>{result.f1_scores.get("running", 0):.3f}</td>
                    <td><strong>{mean_f1:.3f}</strong></td>
                </tr>
            '''

        html += '''
                </tbody>
            </table>
            <p class="legend">🟢 &gt;95% | 🟡 85-95% | 🔴 &lt;85% accuracy vs full preset</p>
        '''

        # Original action distribution table
        html += '''
            <h3>Action Distribution Comparison</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Preset</th>
                        <th>Keypoints</th>
                        <th>Stationary %</th>
                        <th>Walking %</th>
                        <th>Running %</th>
                        <th>Mean Confidence</th>
                    </tr>
                </thead>
                <tbody>
        '''

        # Find best values for highlighting
        best_conf = max(r.mean_confidence for r in keypoint_results) if keypoint_results else 0

        for result in keypoint_results:
            conf_class = "best" if result.mean_confidence == best_conf else ""
            # Handle nested dict format for action_distribution
            dist = result.action_distribution
            stat_val = dist.get("stationary", 0)
            stat_pct = stat_val.get("percentage", 0) if isinstance(stat_val, dict) else stat_val
            walk_val = dist.get("walking", 0)
            walk_pct = walk_val.get("percentage", 0) if isinstance(walk_val, dict) else walk_val
            run_val = dist.get("running", 0)
            run_pct = run_val.get("percentage", 0) if isinstance(run_val, dict) else run_val
            html += f'''
                <tr>
                    <td><strong>{result.preset_name.upper()}</strong></td>
                    <td>{result.num_keypoints}</td>
                    <td>{stat_pct:.1f}%</td>
                    <td>{walk_pct:.1f}%</td>
                    <td>{run_pct:.1f}%</td>
                    <td class="{conf_class}">{result.mean_confidence:.3f}</td>
                </tr>
            '''

        html += '''
                </tbody>
            </table>
        '''

        # Charts
        if perf_chart_b64:
            html += f'''
            <div class="plot-container">
                <div class="plot-title">Accuracy & F1 by Keypoint Count</div>
                <img src="{perf_chart_b64}" alt="Performance by Keypoint Count">
            </div>
            '''

        if chart_b64:
            html += f'''
            <div class="plot-container">
                <div class="plot-title">Action Distribution Comparison</div>
                <img src="{chart_b64}" alt="Keypoint Comparison Chart">
            </div>
            '''

        html += '</div>'
        return html

    def _create_keypoint_performance_chart(
        self,
        keypoint_results: List[KeypointPresetResult],
    ) -> Optional[str]:
        """Create performance metrics chart (accuracy/F1 by keypoint count)."""
        if not keypoint_results:
            return None

        try:
            # Sort by keypoint count (descending)
            results_sorted = sorted(keypoint_results, key=lambda x: x.num_keypoints, reverse=True)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            preset_names = [r.preset_name for r in results_sorted]
            num_keypoints = [r.num_keypoints for r in results_sorted]
            x = np.arange(len(preset_names))

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))

            # 1. Accuracy
            ax1 = axes[0]
            accuracies = [r.action_accuracy * 100 for r in results_sorted]
            bars = ax1.bar(x, accuracies, color=colors, edgecolor='black', linewidth=1.2)
            ax1.set_ylabel("Accuracy (%)", fontsize=11)
            ax1.set_title("Accuracy by Preset", fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{p}\n({n})" for p, n in zip(preset_names, num_keypoints)], fontsize=9)
            ax1.set_ylim(0, 105)
            ax1.grid(axis='y', alpha=0.3)
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

            # 2. F1 Scores by Action
            ax2 = axes[1]
            actions = ["stationary", "walking", "running"]
            action_colors = ['#3498db', '#f39c12', '#e74c3c']
            width = 0.25
            for i, action in enumerate(actions):
                f1_values = [r.f1_scores.get(action, 0) for r in results_sorted]
                ax2.bar(x + i * width, f1_values, width, label=action.capitalize(),
                       color=action_colors[i], edgecolor='black', linewidth=0.5)
            ax2.set_ylabel("F1 Score", fontsize=11)
            ax2.set_title("F1 by Action Class", fontsize=12, fontweight='bold')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels([f"{p}\n({n})" for p, n in zip(preset_names, num_keypoints)], fontsize=9)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.set_ylim(0, 1.05)
            ax2.grid(axis='y', alpha=0.3)

            # 3. Keypoint Count vs Accuracy Trade-off
            ax3 = axes[2]
            scatter = ax3.scatter(num_keypoints, accuracies, c=colors, s=150, edgecolors='black', linewidths=2)
            for n, a, name in zip(num_keypoints, accuracies, preset_names):
                ax3.annotate(name.upper(), (n, a), textcoords="offset points",
                            xytext=(0, 8), ha='center', fontsize=8, fontweight='bold')
            ax3.set_xlabel("Number of Keypoints", fontsize=11)
            ax3.set_ylabel("Accuracy (%)", fontsize=11)
            ax3.set_title("Keypoint Count vs Accuracy", fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Add trend line if enough points
            if len(num_keypoints) >= 2:
                z = np.polyfit(num_keypoints, accuracies, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(num_keypoints), max(num_keypoints), 100)
                ax3.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"Trend")
                ax3.legend(loc='lower right', fontsize=9)

            plt.tight_layout()

            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

            return f"data:image/png;base64,{img_b64}"

        except Exception as e:
            logger.error(f"Failed to create performance chart: {e}")
            return None

    def generate_cross_species_section(
        self,
        species_results: List[SpeciesResult],
    ) -> str:
        """Generate HTML section for cross-species comparison."""
        if not species_results:
            return ""

        # Create body size comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        species_names = [r.species_name for r in species_results]
        body_means = [r.body_size_stats.get("mean", 0) for r in species_results]
        body_stds = [r.body_size_stats.get("std", 0) for r in species_results]

        colors = plt.cm.Set2(np.linspace(0, 1, len(species_results)))

        # Body size bar chart
        bars = axes[0].bar(species_names, body_means, yerr=body_stds, capsize=5, color=colors)
        axes[0].set_ylabel("Body Size (pixels)")
        axes[0].set_title("Body Size Comparison")
        axes[0].grid(axis='y', alpha=0.3)

        for bar, mean, std in zip(bars, body_means, body_stds):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

        # Action distribution grouped bar
        x = np.arange(len(species_names))
        width = 0.25
        actions = ["stationary", "walking", "running"]
        action_colors = [self.ACTION_COLORS[a] for a in actions]

        for i, action in enumerate(actions):
            values = [r.action_distribution.get(action, 0) for r in species_results]
            axes[1].bar(x + i * width, values, width, label=action.capitalize(), color=action_colors[i])

        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title("Action Distribution by Species")
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(species_names)
        axes[1].legend()
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_b64 = self._fig_to_base64(fig)

        # Build HTML
        html = f'''
        <div class="section">
            <h2><span class="icon">🐾</span> Cross-Species Comparison</h2>

            <div class="card-grid">
        '''

        card_colors = ["blue", "green", "orange", "red"]
        for i, result in enumerate(species_results):
            color = card_colors[i % len(card_colors)]
            html += f'''
                <div class="metric-card {color}">
                    <div class="metric-value">{result.species_name}</div>
                    <div class="metric-label">{result.num_frames} frames | Body: {result.body_size_stats.get("mean", 0):.1f}px</div>
                </div>
            '''

        html += '</div>'

        # Comparison table
        html += '''
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Species</th>
                        <th>Model</th>
                        <th>Frames</th>
                        <th>Body Size</th>
                        <th>Stationary</th>
                        <th>Walking</th>
                        <th>Running</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for result in species_results:
            body_str = f"{result.body_size_stats.get('mean', 0):.1f} ± {result.body_size_stats.get('std', 0):.1f}"
            html += f'''
                <tr>
                    <td><strong>{result.species_name}</strong></td>
                    <td>{result.model_type}</td>
                    <td>{result.num_frames}</td>
                    <td>{body_str} px</td>
                    <td>{result.action_distribution.get("stationary", 0):.1f}%</td>
                    <td>{result.action_distribution.get("walking", 0):.1f}%</td>
                    <td>{result.action_distribution.get("running", 0):.1f}%</td>
                </tr>
            '''

        html += '''
                </tbody>
            </table>
        '''

        if chart_b64:
            html += f'''
            <div class="plot-container">
                <img src="{chart_b64}" alt="Cross-Species Comparison">
            </div>
            '''

        html += '</div>'
        return html

    def generate_action_recognition_section(
        self,
        action_results: List[ActionRecognitionResult],
    ) -> str:
        """Generate HTML section for action recognition performance."""
        if not action_results:
            return ""

        # Create performance comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        model_names = [r.model_name for r in action_results]
        accuracies = [r.accuracy * 100 for r in action_results]

        colors = plt.cm.Set2(np.linspace(0, 1, len(action_results)))

        # Accuracy bar chart
        bars = axes[0].bar(model_names, accuracies, color=colors)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Model Accuracy Comparison")
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis='y', alpha=0.3)

        for bar, acc in zip(bars, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # F1 scores grouped bar
        x = np.arange(len(model_names))
        width = 0.25
        actions = ["stationary", "walking", "running"]

        for i, action in enumerate(actions):
            values = [r.f1_scores.get(action, 0) for r in action_results]
            axes[1].bar(x + i * width, values, width, label=action.capitalize(),
                       color=self.ACTION_COLORS[action])

        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("F1 Scores by Action Class")
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_b64 = self._fig_to_base64(fig)

        # Build HTML
        html = f'''
        <div class="section">
            <h2><span class="icon">🎬</span> Action Recognition Performance</h2>

            <div class="card-grid">
        '''

        # Sort by accuracy for ranking
        sorted_results = sorted(action_results, key=lambda x: x.accuracy, reverse=True)

        for i, result in enumerate(sorted_results[:4]):
            rank_emoji = ["🥇", "🥈", "🥉", ""][i] if i < 3 else ""
            color = ["green", "blue", "orange", ""][i] if i < 3 else ""
            html += f'''
                <div class="metric-card {color}">
                    <div class="metric-value">{rank_emoji} {result.accuracy*100:.1f}%</div>
                    <div class="metric-label">{result.model_name}</div>
                </div>
            '''

        html += '</div>'

        # Detailed table
        html += '''
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>F1 (Stationary)</th>
                        <th>F1 (Walking)</th>
                        <th>F1 (Running)</th>
                        <th>Consistency</th>
                    </tr>
                </thead>
                <tbody>
        '''

        for i, result in enumerate(sorted_results):
            rank = i + 1
            highlight = "highlight" if rank == 1 else ""
            html += f'''
                <tr class="{highlight}">
                    <td>#{rank}</td>
                    <td><strong>{result.model_name}</strong></td>
                    <td>{result.accuracy*100:.1f}%</td>
                    <td>{result.f1_scores.get("stationary", 0):.3f}</td>
                    <td>{result.f1_scores.get("walking", 0):.3f}</td>
                    <td>{result.f1_scores.get("running", 0):.3f}</td>
                    <td>{result.consistency_score:.3f}</td>
                </tr>
            '''

        html += '''
                </tbody>
            </table>
        '''

        if chart_b64:
            html += f'''
            <div class="plot-container">
                <img src="{chart_b64}" alt="Action Recognition Performance">
            </div>
            '''

        html += '</div>'
        return html

    def generate_gif_gallery_section(
        self,
        gif_paths: Dict[str, List[Path]],
        title: str = "Action Visualizations",
    ) -> str:
        """Generate HTML section with embedded GIF gallery."""
        if not gif_paths:
            return ""

        html = f'''
        <div class="section">
            <h2><span class="icon">🎞️</span> {title}</h2>
            <div class="gif-gallery">
        '''

        for action_name, paths in gif_paths.items():
            for path in paths[:3]:  # Limit to 3 per action
                b64 = self._image_to_base64(path)
                if b64:
                    caption = f"{action_name.capitalize()} - {Path(path).stem}"
                    html += f'''
                        <div class="gif-card">
                            <img src="{b64}" alt="{action_name}" class="zoomable" data-caption="{caption}">
                            <div class="gif-info">
                                <div class="gif-title">{action_name.capitalize()}</div>
                                <div class="gif-meta">{Path(path).stem}</div>
                            </div>
                        </div>
                    '''

        html += '''
            </div>
        </div>
        '''
        return html

    def generate_preset_action_grid_section(
        self,
        gif_paths: Dict[str, List[Path]],
        title: str = "Keypoint Preset × Action Comparison",
    ) -> str:
        """
        Generate HTML section with preset × action GIF grid.

        Shows a grid where rows are presets and columns are actions,
        allowing visual comparison of how each action looks with different keypoint counts.
        """
        # Extract preset×action GIFs
        presets = ["full", "standard", "mars", "locomotion", "minimal"]
        actions = ["stationary", "walking", "running"]

        # Build a lookup dict
        preset_action_gifs = {}
        for category, paths in gif_paths.items():
            if category.startswith("preset_"):
                parts = category.split("_")
                if len(parts) >= 3:
                    preset = parts[1]
                    action = parts[2]
                    key = (preset, action)
                    if key not in preset_action_gifs:
                        preset_action_gifs[key] = []
                    preset_action_gifs[key].extend(paths)

        if not preset_action_gifs:
            return ""

        # Determine which presets/actions have GIFs
        active_presets = sorted(set(k[0] for k in preset_action_gifs.keys()),
                               key=lambda x: presets.index(x) if x in presets else 99)
        active_actions = sorted(set(k[1] for k in preset_action_gifs.keys()),
                               key=lambda x: actions.index(x) if x in actions else 99)

        if not active_presets or not active_actions:
            return ""

        # Action colors for column headers
        action_colors = {
            "stationary": "#3498db",
            "walking": "#2ecc71",
            "running": "#e74c3c",
        }

        html = f'''
        <div class="section">
            <h2><span class="icon">🎯</span> {title}</h2>
            <p class="section-desc">
                Visual comparison of action recognition across different keypoint configurations.
                Each row shows a keypoint preset, each column shows an action type.
            </p>
            <div class="preset-action-grid">
                <table class="pa-table">
                    <thead>
                        <tr>
                            <th class="pa-corner">Preset \\ Action</th>
        '''

        # Column headers (actions)
        for action in active_actions:
            color = action_colors.get(action, "#95a5a6")
            html += f'<th class="pa-action-header" style="background-color: {color};">{action.capitalize()}</th>'

        html += '</tr></thead><tbody>'

        # Rows (presets)
        preset_kp_counts = {
            "full": 27, "standard": 11, "mars": 7, "locomotion": 5, "minimal": 3
        }

        for preset in active_presets:
            kp_count = preset_kp_counts.get(preset, "?")
            html += f'''
                <tr>
                    <td class="pa-preset-header">
                        <strong>{preset.upper()}</strong>
                        <span class="kp-count">({kp_count} kp)</span>
                    </td>
            '''

            for action in active_actions:
                key = (preset, action)
                paths = preset_action_gifs.get(key, [])

                if paths:
                    # Use first GIF
                    b64 = self._image_to_base64(paths[0])
                    if b64:
                        caption = f"{preset.upper()} preset - {action.capitalize()}"
                        html += f'''
                            <td class="pa-cell">
                                <img src="{b64}" alt="{preset} {action}" class="pa-gif zoomable" data-caption="{caption}">
                            </td>
                        '''
                    else:
                        html += '<td class="pa-cell pa-empty">-</td>'
                else:
                    html += '<td class="pa-cell pa-empty">-</td>'

            html += '</tr>'

        html += '''
                    </tbody>
                </table>
            </div>
            <style>
                .preset-action-grid {
                    overflow-x: auto;
                    margin: 20px 0;
                }
                .pa-table {
                    border-collapse: collapse;
                    width: 100%;
                    background: white;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }
                .pa-corner {
                    background: #2c3e50;
                    color: white;
                    padding: 12px;
                    font-weight: bold;
                    text-align: center;
                }
                .pa-action-header {
                    color: white;
                    padding: 12px;
                    font-weight: bold;
                    text-align: center;
                    min-width: 150px;
                }
                .pa-preset-header {
                    background: #34495e;
                    color: white;
                    padding: 12px;
                    text-align: center;
                }
                .pa-preset-header .kp-count {
                    display: block;
                    font-size: 0.85em;
                    opacity: 0.8;
                    margin-top: 4px;
                }
                .pa-cell {
                    padding: 8px;
                    text-align: center;
                    border: 1px solid #ecf0f1;
                    vertical-align: middle;
                }
                .pa-gif {
                    max-width: 200px;
                    max-height: 150px;
                    border-radius: 4px;
                    border: 2px solid #ecf0f1;
                }
                .pa-empty {
                    color: #bdc3c7;
                    font-style: italic;
                }
                .section-desc {
                    color: #7f8c8d;
                    margin-bottom: 20px;
                    font-size: 0.95em;
                }
            </style>
        </div>
        '''
        return html

    def generate_action_class_comparison_section(
        self,
        gif_paths: Dict[str, List[Path]],
        keypoint_results: List[KeypointPresetResult] = None,
        title: str = "Action Class Comparison",
    ) -> str:
        """
        Generate HTML section with action class-based GIF comparison.

        Shows GIFs organized by action class (stationary, walking, running),
        allowing visual comparison of how each preset performs on each action type.
        Each action class gets its own category section with all presets side-by-side.
        """
        presets = ["full", "standard", "mars", "locomotion", "minimal"]
        actions = ["stationary", "walking", "running"]

        # Build a lookup dict: (preset, action) -> [paths]
        preset_action_gifs = {}
        for category, paths in gif_paths.items():
            if category.startswith("preset_"):
                parts = category.split("_")
                if len(parts) >= 3:
                    preset = parts[1]
                    action = parts[2]
                    key = (preset, action)
                    if key not in preset_action_gifs:
                        preset_action_gifs[key] = []
                    preset_action_gifs[key].extend(paths)

        if not preset_action_gifs:
            return ""

        # Get active presets and actions
        active_presets = sorted(set(k[0] for k in preset_action_gifs.keys()),
                               key=lambda x: presets.index(x) if x in presets else 99)
        active_actions = sorted(set(k[1] for k in preset_action_gifs.keys()),
                               key=lambda x: actions.index(x) if x in actions else 99)

        if not active_presets or not active_actions:
            return ""

        # Action class metadata
        action_info = {
            "stationary": {
                "icon": "🧘",
                "color": "#3498db",
                "bg_color": "#ebf5fb",
                "description": "Frames where the animal is not moving (speed < threshold)",
            },
            "walking": {
                "icon": "🚶",
                "color": "#2ecc71",
                "bg_color": "#eafaf1",
                "description": "Frames where the animal is walking at moderate speed",
            },
            "running": {
                "icon": "🏃",
                "color": "#e74c3c",
                "bg_color": "#fdedec",
                "description": "Frames where the animal is running at high speed",
            },
        }

        # Preset keypoint counts and F1 scores
        preset_kp_counts = {
            "full": 27, "standard": 11, "mars": 7, "locomotion": 5, "minimal": 3
        }

        # Extract F1 scores from keypoint_results if available
        preset_f1_by_action = {}
        if keypoint_results:
            for result in keypoint_results:
                preset_f1_by_action[result.preset_name] = result.f1_scores

        html = f'''
        <div class="section action-class-section">
            <h2><span class="icon">📊</span> {title}</h2>
            <p class="section-desc">
                Visual comparison of action recognition performance organized by action class.
                Each category shows how different keypoint presets perform for that specific behavior.
            </p>
        '''

        # Generate a card section for each action class
        for action in active_actions:
            info = action_info.get(action, {
                "icon": "❓",
                "color": "#95a5a6",
                "bg_color": "#f8f9fa",
                "description": "",
            })

            html += f'''
            <div class="action-category-card" style="border-left: 5px solid {info["color"]}; background: {info["bg_color"]};">
                <div class="action-category-header">
                    <span class="action-icon">{info["icon"]}</span>
                    <h3 class="action-title" style="color: {info["color"]};">{action.upper()}</h3>
                    <p class="action-desc">{info["description"]}</p>
                </div>
                <div class="preset-gif-row">
            '''

            # Show all presets for this action
            for preset in active_presets:
                key = (preset, action)
                paths = preset_action_gifs.get(key, [])
                kp_count = preset_kp_counts.get(preset, "?")

                # Get F1 score for this preset/action
                f1_score = None
                if preset in preset_f1_by_action:
                    f1_score = preset_f1_by_action[preset].get(action, None)

                f1_display = f"{f1_score:.2f}" if f1_score is not None else "-"
                f1_color = info["color"] if f1_score is not None and f1_score >= 0.8 else "#e74c3c" if f1_score is not None and f1_score < 0.5 else "#f39c12"

                html += f'''
                    <div class="preset-gif-card">
                        <div class="preset-gif-header">
                            <strong>{preset.upper()}</strong>
                            <span class="kp-badge">{kp_count} kp</span>
                        </div>
                '''

                if paths:
                    b64 = self._image_to_base64(paths[0])
                    if b64:
                        caption = f"{preset.upper()} preset - {action.capitalize()} (F1: {f1_display})"
                        html += f'''
                        <div class="preset-gif-container">
                            <img src="{b64}" alt="{preset} {action}" class="preset-gif-img zoomable" data-caption="{caption}">
                        </div>
                        '''
                    else:
                        html += '<div class="preset-gif-empty">No GIF</div>'
                else:
                    html += '<div class="preset-gif-empty">No GIF</div>'

                html += f'''
                        <div class="preset-gif-footer">
                            <span class="f1-label">F1:</span>
                            <span class="f1-value" style="color: {f1_color};">{f1_display}</span>
                        </div>
                    </div>
                '''

            html += '''
                </div>
            </div>
            '''

        # Add CSS for this section
        html += '''
            <style>
                .action-class-section {
                    margin-top: 30px;
                }
                .action-category-card {
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                }
                .action-category-header {
                    display: flex;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 15px;
                }
                .action-icon {
                    font-size: 2em;
                }
                .action-title {
                    margin: 0;
                    font-size: 1.5em;
                    font-weight: bold;
                }
                .action-desc {
                    width: 100%;
                    margin: 5px 0 0 0;
                    font-size: 0.9em;
                    color: #666;
                }
                .preset-gif-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    justify-content: flex-start;
                }
                .preset-gif-card {
                    background: white;
                    border-radius: 8px;
                    padding: 10px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                    min-width: 180px;
                    max-width: 220px;
                    flex: 1;
                }
                .preset-gif-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                    padding-bottom: 5px;
                    border-bottom: 1px solid #eee;
                }
                .kp-badge {
                    background: #ecf0f1;
                    color: #7f8c8d;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 0.8em;
                }
                .preset-gif-container {
                    text-align: center;
                    min-height: 120px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .preset-gif-img {
                    max-width: 100%;
                    max-height: 150px;
                    border-radius: 4px;
                    border: 1px solid #ddd;
                }
                .preset-gif-empty {
                    color: #bdc3c7;
                    font-style: italic;
                    text-align: center;
                    padding: 40px 10px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }
                .preset-gif-footer {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 8px;
                    margin-top: 8px;
                    padding-top: 8px;
                    border-top: 1px solid #eee;
                }
                .f1-label {
                    font-weight: bold;
                    color: #666;
                }
                .f1-value {
                    font-size: 1.2em;
                    font-weight: bold;
                }
            </style>
        </div>
        '''
        return html

    def generate_plots_section(
        self,
        plot_paths: Dict[str, Path],
        title: str = "Analysis Plots",
    ) -> str:
        """Generate HTML section with embedded plot images."""
        if not plot_paths:
            return ""

        html = f'''
        <div class="section">
            <h2><span class="icon">📊</span> {title}</h2>
            <div class="flex-row">
        '''

        for plot_name, path in plot_paths.items():
            b64 = self._image_to_base64(path)
            if b64:
                caption = plot_name.replace("_", " ").title()
                html += f'''
                <div class="flex-col">
                    <div class="plot-container">
                        <div class="plot-title">{caption}</div>
                        <img src="{b64}" alt="{plot_name}" class="zoomable" data-caption="{caption}">
                    </div>
                </div>
                '''

        html += '''
            </div>
        </div>
        '''
        return html

    def generate_full_dashboard(
        self,
        summary: ExperimentSummary,
        gif_paths: Dict[str, List[Path]] = None,
        plot_paths: Dict[str, Path] = None,
        additional_html: str = "",
    ) -> Path:
        """Generate complete HTML dashboard."""

        # Header
        header_html = f'''
        <div class="dashboard-header">
            <h1>SuperAnimal Behavior Analysis Dashboard</h1>
            <div class="subtitle">{summary.experiment_name}</div>
            <div class="meta">
                <div class="meta-item">
                    <div class="label">Generated</div>
                    <div class="value">{summary.timestamp}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Total Frames</div>
                    <div class="value">{summary.total_frames:,}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Species</div>
                    <div class="value">{", ".join(summary.species)}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Presets Tested</div>
                    <div class="value">{len(summary.presets_tested)}</div>
                </div>
            </div>
        </div>
        '''

        # Generate sections
        sections = []

        if summary.keypoint_results:
            sections.append(self.generate_keypoint_comparison_section(summary.keypoint_results))

        if summary.species_results:
            sections.append(self.generate_cross_species_section(summary.species_results))

        if summary.action_results:
            sections.append(self.generate_action_recognition_section(summary.action_results))

        if gif_paths:
            # Add action class comparison section (organized by action class)
            action_class_section = self.generate_action_class_comparison_section(
                gif_paths,
                keypoint_results=summary.keypoint_results,
            )
            if action_class_section:
                sections.append(action_class_section)
            # Add preset × action grid if available
            preset_action_grid = self.generate_preset_action_grid_section(gif_paths)
            if preset_action_grid:
                sections.append(preset_action_grid)
            # Add general GIF gallery
            sections.append(self.generate_gif_gallery_section(gif_paths))

        if plot_paths:
            sections.append(self.generate_plots_section(plot_paths))

        if additional_html:
            sections.append(additional_html)

        # Footer
        footer_html = f'''
        <footer>
            Generated by <strong>SuperAnimal Behavior Analysis Pipeline</strong><br>
            <a href="https://github.com/DeepLabCut/DeepLabCut">DeepLabCut 3.0</a> |
            Report generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </footer>
        '''

        # Lightbox overlay HTML
        lightbox_html = '''
        <!-- Lightbox Overlay for image zoom -->
        <div id="lightbox-overlay" class="lightbox-overlay" onclick="closeLightbox()">
            <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
            <img id="lightbox-content" class="lightbox-content" src="" alt="Zoomed image">
            <div id="lightbox-caption" class="lightbox-caption"></div>
        </div>
        '''

        # Assemble full HTML
        full_html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{summary.experiment_name} - Dashboard</title>
    {DASHBOARD_CSS}
</head>
<body>
    <div class="dashboard-container">
        {header_html}
        {"".join(sections)}
        {footer_html}
    </div>
    {lightbox_html}
    {DASHBOARD_JS}
</body>
</html>
        '''

        # Save dashboard
        output_path = self.output_dir / "dashboard.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        logger.info(f"Saved dashboard: {output_path}")
        return output_path


# ============================================================
# Utility Functions
# ============================================================

def collect_experiment_results(
    experiment_dir: Path,
    experiment_name: str = "Experiment",
) -> ExperimentSummary:
    """
    Collect results from experiment directory.

    Args:
        experiment_dir: Path to experiment output directory
        experiment_name: Name for the experiment

    Returns:
        ExperimentSummary object
    """
    experiment_dir = Path(experiment_dir)

    summary = ExperimentSummary(
        experiment_name=experiment_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_frames=0,
        species=[],
        presets_tested=[],
    )

    # Look for result files
    json_files = list(experiment_dir.rglob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Parse different result types based on filename/content
            if "keypoint" in json_file.stem.lower() or "preset" in json_file.stem.lower():
                # Keypoint comparison results
                pass
            elif "species" in json_file.stem.lower() or "cross" in json_file.stem.lower():
                # Cross-species results
                pass
            elif "action" in json_file.stem.lower() or "model" in json_file.stem.lower():
                # Action recognition results
                pass

        except Exception as e:
            logger.warning(f"Failed to parse {json_file}: {e}")

    return summary


def create_keypoint_comparison_gif(
    video_path: Path,
    keypoints_dict: Dict[str, Tuple[np.ndarray, List[str]]],
    output_path: Path,
    max_frames: int = 100,
    fps: float = 8.0,
    confidence_threshold: float = 0.3,
) -> Optional[Path]:
    """
    Create side-by-side GIF comparing different keypoint configurations.

    Args:
        video_path: Path to video
        keypoints_dict: Dict mapping preset name to (keypoints, keypoint_names)
        output_path: Output GIF path
        max_frames: Maximum frames
        fps: GIF frame rate
        confidence_threshold: Confidence threshold

    Returns:
        Path to saved GIF
    """
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not installed")
        return None

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Subsample frames
    num_kp_frames = min(total_frames, *[len(kp[0]) for kp in keypoints_dict.values()])
    if num_kp_frames > max_frames:
        step = num_kp_frames // max_frames
        frame_indices = list(range(0, num_kp_frames, step))[:max_frames]
    else:
        frame_indices = list(range(num_kp_frames))

    presets = list(keypoints_dict.keys())
    num_presets = len(presets)

    # Panel dimensions
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

        for i, preset_name in enumerate(presets):
            keypoints, keypoint_names = keypoints_dict[preset_name]

            # Resize frame
            panel = cv2.resize(frame.copy(), (panel_width, panel_height))

            # Draw keypoints
            if frame_idx < len(keypoints):
                kp = keypoints[frame_idx]
                colors = generate_keypoint_colors(len(keypoint_names))

                for j, (name, color) in enumerate(zip(keypoint_names, colors)):
                    if j < len(kp):
                        x, y, conf = kp[j]
                        if conf > confidence_threshold:
                            px = int(x * panel_width / frame_width)
                            py = int(y * panel_height / frame_height)
                            cv2.circle(panel, (px, py), 4, color, -1, cv2.LINE_AA)

            # Add label
            label = f"{preset_name.upper()} ({len(keypoint_names)} kp)"
            cv2.rectangle(panel, (5, 5), (len(label) * 9 + 15, 28), (0, 0, 0), -1)
            cv2.putText(panel, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

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
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), combined_frames, duration=1.0/fps, loop=0)
        logger.info(f"Saved comparison GIF: {output_path}")
        return output_path

    return None


def generate_keypoint_colors(num_keypoints: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for keypoints."""
    colors = []
    for i in range(num_keypoints):
        hue = int(180 * i / num_keypoints)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors
