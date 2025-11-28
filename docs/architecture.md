# SuperAnimal Behavior PoC - Architecture

## Overview

This document describes the architecture of the SuperAnimal Behavior Analysis Pipeline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pipeline Scripts                                 │
├─────────────────┬─────────────────────┬─────────────────────────────────┤
│     run.py      │ run_keypoint_       │     run_cross_species.py        │
│  (Single Video) │ comparison.py       │   (Multi-Species Comparison)    │
└────────┬────────┴──────────┬──────────┴────────────────┬────────────────┘
         │                   │                           │
         ▼                   ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Core Components                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ SuperAnimal     │  │ UnifiedAction   │  │ Report Generator        │  │
│  │ Predictor       │  │ Classifier      │  │ (HTML, GIF, Plots)      │  │
│  │                 │  │                 │  │                         │  │
│  │ - TopViewMouse  │  │ - Velocity-based│  │ - ActionGifGenerator    │  │
│  │ - Quadruped     │  │ - Body-length   │  │ - HTMLReportGenerator   │  │
│  │                 │  │   normalized    │  │ - KeypointComparison    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
│           │                    │                        │               │
│           ▼                    ▼                        ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Configuration System (Hydra)                  │   │
│  │  configs/                                                        │   │
│  │  ├── config.yaml        # Main config                           │   │
│  │  ├── model/             # Model presets (topviewmouse, quadruped)│   │
│  │  ├── species/           # Species-specific settings              │   │
│  │  ├── data/              # Data/video settings                    │   │
│  │  └── behavior/          # Behavior analysis settings             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Scripts

### 1. run.py - Single Video Analysis

Main entry point for analyzing a single video.

```bash
python run.py                           # Default sample video
python run.py input=/path/to/video.mp4  # Custom video
python run.py model=quadruped           # Different model
```

**Output:**
- Keypoint predictions (CSV/H5)
- Behavior metrics (JSON)
- Trajectory plots
- Action GIFs
- HTML report

### 2. run_keypoint_comparison.py - Keypoint Preset Comparison

Compare different keypoint configurations (Full, Standard, Minimal) on the same video.

```bash
python run_keypoint_comparison.py
python run_keypoint_comparison.py data.video.max_frames=200
```

**Output:**
- Side-by-side comparison GIF (animated)
- Static comparison report (PNG)
- Trajectory comparison per preset

### 3. run_cross_species.py - Cross-Species Comparison

Compare actions and body sizes across multiple species.

```bash
python run_cross_species.py                     # Default: mouse + dog
python run_cross_species.py species=[mouse,dog,horse]
```

**Output:**
- Action distribution comparison
- Body size comparison
- Per-species GIFs
- Cross-species HTML report

---

## Core Components

### SuperAnimalPredictor

Wrapper around DeepLabCut 3.0 SuperAnimal models.

```python
from src.models.predictor import SuperAnimalPredictor

predictor = SuperAnimalPredictor(
    model_type="topviewmouse",  # or "quadruped"
    model_name="hrnet_w32",
    device="cuda",
    use_keypoints=["nose", "mouse_center", "tail_base"],  # Optional filter
)

results = predictor.predict_video(
    video_path="video.mp4",
    output_dir="outputs/",
    max_frames=500,
)
```

**Available Models:**

| Model Type | Keypoints | Use Case |
|------------|-----------|----------|
| `topviewmouse` | 27 | Mice (top view) |
| `quadruped` | 39 | Dogs, horses, cats, etc. (side view) |

### UnifiedActionClassifier

Velocity-based action classification, normalized by body size.

```python
from src.models.action_classifier import UnifiedActionClassifier

classifier = UnifiedActionClassifier(
    species="mouse",
    fps=30.0,
)

# Override thresholds
classifier.thresholds = {
    "stationary": 0.5,  # body-lengths/sec
    "walking": 3.0,
}

metrics = classifier.analyze(keypoints, keypoint_names)
```

**Action Types:**
- `stationary`: < stationary_threshold body-lengths/sec
- `walking`: stationary_threshold ~ walking_threshold
- `running`: > walking_threshold

### Report Generators

#### ActionGifGenerator

```python
from src.analysis.report_generator import ActionGifGenerator

generator = ActionGifGenerator(output_dir="gifs/")
generator.generate_all_action_gifs(
    video_path=video_path,
    keypoints=keypoints,
    keypoint_names=names,
    action_labels=labels,
    action_names={0: "stationary", 1: "walking", 2: "running"},
    max_frames_per_gif=100,      # Max frames in each GIF
    segment_duration_sec=4.0,    # Target duration
    fps=8.0,                     # GIF frame rate
)
```

#### HTMLReportGenerator

```python
from src.analysis.report_generator import HTMLReportGenerator

generator = HTMLReportGenerator(output_dir="reports/")
generator.generate_behavior_report(
    video_name="mouse_sample",
    species="mouse",
    metrics=metrics_dict,
    action_gifs={"walking": [Path("walk.gif")]},
    plot_paths={"trajectory": Path("trajectory.png")},
)
```

---

## Configuration System

### Directory Structure

```
configs/
├── config.yaml          # Main config (defaults, paths, report settings)
├── model/
│   ├── topviewmouse.yaml   # Mouse model + keypoint presets
│   └── quadruped.yaml      # Quadruped model + keypoint presets
├── species/
│   ├── mouse.yaml          # Mouse-specific thresholds
│   ├── dog.yaml            # Dog-specific thresholds
│   └── horse.yaml          # Horse-specific thresholds
├── data/
│   └── sample.yaml         # Data paths, video settings
└── behavior/
    └── default.yaml        # Behavior analysis parameters
```

### Key Configuration Options

#### config.yaml

```yaml
# Report settings
report:
  html: true              # Generate HTML report
  gifs: true              # Generate action GIFs
  gifs_per_action: 2      # Max GIFs per action type
  gif_fps: 8              # GIF frame rate
  gif_max_frames: 100     # Max frames per GIF
  gif_duration_sec: 4.0   # Target GIF duration

# Species to compare (for run_cross_species.py)
species:
  - mouse
  - dog
```

#### model/topviewmouse.yaml

```yaml
name: superanimal_topviewmouse

# Keypoint presets
keypoint_presets:
  full: null              # All 27 keypoints
  standard:               # 11 keypoints (default)
    - nose
    - left_ear
    - right_ear
    - neck
    - mouse_center
    - left_shoulder
    - right_shoulder
    - left_hip
    - right_hip
    - tail_base
    - tail_end
  minimal:                # 3 keypoints
    - nose
    - mouse_center
    - tail_base

# Active preset
use_keypoints: ${model.keypoint_presets.standard}
```

#### species/mouse.yaml

```yaml
name: mouse
description: "Mouse (top view)"

model:
  type: topviewmouse
  use_keypoints: ${model.keypoint_presets.standard}

velocity_thresholds:
  stationary: 0.5   # body-lengths/sec
  walking: 3.0

center_keypoint: mouse_center
```

### Command-Line Overrides

```bash
# Override single value
python run.py report.gif_max_frames=150

# Override nested value
python run.py model.use_keypoints=null

# Override species threshold
python run_cross_species.py species.mouse.velocity_thresholds.walking=2.5

# Use different model preset
python run.py model=quadruped
```

---

## Data Flow

### Single Video Analysis (run.py)

```
Video Input
    │
    ▼
┌─────────────────────┐
│ SuperAnimalPredictor│ ──► Keypoints (frames, kp, 3)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ BehaviorAnalyzer    │ ──► Behavior Metrics
└─────────────────────┘
    │
    ├──► Trajectory Plot
    ├──► Action Timeline
    ├──► Action GIFs
    └──► HTML Report
```

### Cross-Species Comparison (run_cross_species.py)

```
Species List [mouse, dog, horse]
    │
    ├──► Mouse ──► topviewmouse model ──► Keypoints ──► Actions
    │                                                      │
    ├──► Dog   ──► quadruped model    ──► Keypoints ──► Actions
    │                                                      │
    └──► Horse ──► quadruped model    ──► Keypoints ──► Actions
                                                           │
                                                           ▼
                                              ┌─────────────────────┐
                                              │ CrossSpecies        │
                                              │ Comparator          │
                                              └─────────────────────┘
                                                           │
                                                           ▼
                                              - Action Distribution Chart
                                              - Velocity Profiles
                                              - Body Size Comparison
                                              - HTML Report
```

---

## Extending the System

### Adding a New Species

1. Create species config:

```yaml
# configs/species/cat.yaml
name: cat
description: "Cat (side view)"

model:
  type: quadruped

velocity_thresholds:
  stationary: 0.4
  walking: 2.0

center_keypoint: back_middle
```

2. Add sample video to downloader (optional):

```python
# src/data/downloader.py
SAMPLE_VIDEOS = {
    "cat_walking": {
        "urls": ["https://..."],
        "filename": "cat_sample.mp4",
        "animal_type": "quadruped",
        "species": "cat",
    },
}
```

3. Run comparison:

```bash
python run_cross_species.py species=[mouse,cat]
```

### Adding Custom Behaviors

Extend `UnifiedActionClassifier` for species-specific behaviors:

```python
class MouseBehaviorClassifier(UnifiedActionClassifier):
    def detect_grooming(self, keypoints, keypoint_names):
        # Custom grooming detection logic
        pass
```

---

## Performance Considerations

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| `max_frames` | Memory, processing time | 200-500 for PoC |
| `gif_max_frames` | GIF file size | 50-150 |
| `keypoint_presets` | Accuracy vs speed | `standard` for balance |
| `model_name` | Accuracy vs speed | `hrnet_w32` default |

---

## Troubleshooting

### Common Issues

1. **Video download fails (404)**
   - Check URL in `src/data/downloader.py`
   - Use local video: `python run.py input=/path/to/video.mp4`

2. **No keypoints detected**
   - Check video quality and animal visibility
   - Try different `bbox_threshold` in model config

3. **GIFs too short/long**
   - Adjust `report.gif_max_frames` and `report.gif_duration_sec`

4. **Wrong action classification**
   - Tune `velocity_thresholds` in species config
   - Check if body size estimation is accurate
