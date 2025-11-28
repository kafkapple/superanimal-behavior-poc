# Custom Dataset & Fine-tuning Guide

This guide explains how to use your own video data and fine-tune SuperAnimal models for better performance on your specific experimental setup.

## Table of Contents
1. [Using Custom Videos](#1-using-custom-videos)
2. [Directory Structure](#2-directory-structure)
3. [Running Inference on Custom Data](#3-running-inference-on-custom-data)
4. [Fine-tuning SuperAnimal Models](#4-fine-tuning-superanimal-models)
5. [Model Locations](#5-model-locations)
6. [Best Practices](#6-best-practices)

---

## 1. Using Custom Videos

### Supported Formats
- **Video**: MP4, AVI, MOV, MKV
- **Resolution**: Any (640x480 recommended for speed)
- **FPS**: 30 fps typical for behavior analysis
- **Duration**: Any (shorter videos process faster)

### Video Requirements
- Clear view of the animal
- Consistent lighting (avoid flickering)
- Minimal motion blur
- Top-view for mice (SuperAnimal-TopViewMouse)
- Side-view for quadrupeds (SuperAnimal-Quadruped)

---

## 2. Directory Structure

### Option A: Place videos in `data/raw/`

```
superanimal-behavior-poc/
├── data/
│   └── raw/
│       ├── my_experiment_01.mp4    # Your videos here
│       ├── my_experiment_02.mp4
│       └── mouse_session_001.mp4
```

### Option B: Specify custom path via command line

```bash
# Single video
python run.py data.video_path=/path/to/your/video.mp4

# Custom data directory
python run.py data.data_dir=/path/to/your/data/folder
```

### Option C: Create custom config

```yaml
# configs/data/my_experiment.yaml
data_dir: /absolute/path/to/my/data
videos:
  experiment_01:
    type: local
    path: experiment_01.mp4
    animal_type: topviewmouse
    description: "My custom experiment"
```

Use with:
```bash
python run.py data=my_experiment
```

---

## 3. Running Inference on Custom Data

### Basic Usage

```bash
# Activate environment
conda activate superanimal-dlc3

# Run on single video
python run.py data.video_path=/path/to/video.mp4

# With specific model
python run.py data.video_path=/path/to/video.mp4 model=topviewmouse

# Limit frames for testing
python run.py data.video_path=/path/to/video.mp4 data.video.max_frames=100
```

### Python API

```python
from pathlib import Path
from src.models.predictor import SuperAnimalPredictor
from src.analysis.behavior import BehaviorAnalyzer
from src.analysis.visualizer import Visualizer
from src.analysis.report_generator import KeypointComparisonReport

# Initialize predictor
predictor = SuperAnimalPredictor(
    model_type="topviewmouse",  # or "quadruped"
    model_name="hrnet_w32",
    video_adapt=False,  # Set True for better accuracy (slower)
    device="auto",  # "cuda", "mps", "cpu"
)

# Run inference
video_path = Path("/path/to/your/video.mp4")
output_dir = Path("./my_output")

results = predictor.predict_video(
    video_path=video_path,
    output_dir=output_dir,
    max_frames=None,  # Process all frames
)

# Get keypoints
keypoints = results["keypoints"]  # Shape: (frames, keypoints, 3)
keypoint_names = predictor.get_keypoint_names()

print(f"Processed {results['metadata']['num_frames']} frames")
print(f"Keypoints: {len(keypoint_names)}")
```

### Batch Processing Multiple Videos

```python
from pathlib import Path
from src.models.predictor import SuperAnimalPredictor

video_dir = Path("/path/to/videos")
output_dir = Path("./batch_output")

predictor = SuperAnimalPredictor(model_type="topviewmouse")

for video_path in video_dir.glob("*.mp4"):
    print(f"Processing: {video_path.name}")

    results = predictor.predict_video(
        video_path=video_path,
        output_dir=output_dir / video_path.stem,
    )

    print(f"  Frames: {results['metadata']['num_frames']}")
```

---

## 4. Fine-tuning SuperAnimal Models

Fine-tuning improves model performance for your specific experimental setup, animal strain, or camera angle.

### When to Fine-tune
- Model predictions have low confidence
- Animal appearance differs from training data (albino mice, specific strains)
- Camera setup is unusual (angle, resolution, lighting)
- Need higher accuracy for specific bodyparts

### Step 1: Create Annotation Project

```python
import deeplabcut

# Create new project for fine-tuning
config_path = deeplabcut.create_project(
    project="MyMouseExperiment",
    experimenter="YourName",
    videos=["/path/to/video1.mp4", "/path/to/video2.mp4"],
    working_directory="/path/to/projects",
    copy_videos=True,
)

print(f"Project created: {config_path}")
```

### Step 2: Extract Frames for Labeling

```python
import deeplabcut

config_path = "/path/to/projects/MyMouseExperiment-YourName-2025-01-01/config.yaml"

# Extract frames (k-means clustering for diversity)
deeplabcut.extract_frames(
    config_path,
    mode="automatic",
    algo="kmeans",
    userfeedback=False,
    crop=False,
)
```

### Step 3: Label Frames

```python
# Open labeling GUI
deeplabcut.label_frames(config_path)
```

Or use the napari-based labeler:
```python
deeplabcut.label_frames(config_path, use_napari=True)
```

**Labeling Tips:**
- Label 50-200 frames for good results
- Include diverse poses and positions
- Mark occluded keypoints as "not visible"
- Be consistent with keypoint placement

### Step 4: Create Training Dataset

```python
deeplabcut.create_training_dataset(
    config_path,
    net_type="hrnet_w32",  # Same as SuperAnimal
    augmenter_type="imgaug",
)
```

### Step 5: Fine-tune from SuperAnimal Weights

```python
# Fine-tune starting from SuperAnimal weights
deeplabcut.train_network(
    config_path,
    shuffle=1,
    trainingsetindex=0,
    maxiters=50000,  # Adjust based on dataset size
    saveiters=10000,
    displayiters=1000,
    # SuperAnimal transfer learning
    superanimal_name="superanimal_topviewmouse",
    superanimal_transfer_learning=True,
)
```

### Step 6: Evaluate Model

```python
# Evaluate on test set
deeplabcut.evaluate_network(
    config_path,
    plotting=True,
)

# Analyze videos with fine-tuned model
deeplabcut.analyze_videos(
    config_path,
    ["/path/to/test_video.mp4"],
    shuffle=1,
    save_as_csv=True,
)
```

### Step 7: Use Fine-tuned Model

```python
# Use your fine-tuned model instead of SuperAnimal
deeplabcut.analyze_videos(
    config_path,  # Your project config
    ["/path/to/new_video.mp4"],
    videotype="mp4",
    shuffle=1,
    save_as_csv=True,
)

# Create labeled video
deeplabcut.create_labeled_video(
    config_path,
    ["/path/to/new_video.mp4"],
    shuffle=1,
    draw_skeleton=True,
)
```

---

## 5. Model Locations

### SuperAnimal Pre-trained Models

Models are automatically downloaded to:
```
~/.deeplabcut/
├── models/
│   ├── superanimal_topviewmouse_hrnet_w32/
│   │   ├── detector/
│   │   │   └── fasterrcnn_resnet50_fpn_v2/
│   │   └── pose/
│   │       └── hrnet_w32/
│   └── superanimal_quadruped_hrnet_w32/
│       ├── detector/
│       └── pose/
```

### Your Fine-tuned Models

Located in your project directory:
```
/path/to/projects/MyMouseExperiment-YourName-2025-01-01/
├── config.yaml                    # Project configuration
├── dlc-models/
│   └── iteration-0/
│       └── MyMouseExperimentJan1-trainset95shuffle1/
│           ├── train/
│           │   ├── snapshot-50000.pb    # TF model (DLC 2.x)
│           │   └── snapshot-50000.pt    # PyTorch model (DLC 3.x)
│           └── test/
├── labeled-data/                  # Your annotations
│   ├── video1/
│   │   └── CollectedData_YourName.csv
│   └── video2/
├── training-datasets/
└── videos/
```

### Export/Share Models

```python
# Export model for sharing
deeplabcut.export_model(
    config_path,
    shuffle=1,
    make_tar=True,  # Creates portable archive
)
```

---

## 6. Best Practices

### Video Quality
- **Resolution**: 640x480 minimum, higher for fine details
- **Frame Rate**: 30 fps for general behavior, 60+ for fast movements
- **Lighting**: Consistent, avoid shadows
- **Background**: High contrast with animal

### Annotation Guidelines
- **Quantity**: 50-200 frames typically sufficient
- **Diversity**: Include all behaviors and positions
- **Consistency**: Same person labels all frames if possible
- **Occlusions**: Mark as not visible, don't guess

### Training Tips
- Start with SuperAnimal transfer learning
- Use data augmentation (rotation, scaling, brightness)
- Monitor training loss, stop if overfitting
- Validate on held-out videos

### Inference Optimization
- Use GPU (CUDA or MPS) for faster processing
- Reduce resolution for real-time applications
- Use minimal keypoint preset for speed
- Process in batches for multiple videos

---

## Quick Reference

### Command Line Examples

```bash
# Basic inference
python run.py data.video_path=/my/video.mp4

# Specific output directory
python run.py data.video_path=/my/video.mp4 output_dir=/my/output

# Use quadruped model
python run.py data.video_path=/my/video.mp4 model=quadruped

# Enable video adaptation (slower, more accurate)
python run.py data.video_path=/my/video.mp4 model.video_adapt=true

# Limit frames
python run.py data.video_path=/my/video.mp4 data.video.max_frames=500

# Use minimal keypoints
python run.py data.video_path=/my/video.mp4 model.use_keypoints=[nose,mouse_center,tail_base]

# Generate comparison report
python -c "
from src.analysis.report_generator import KeypointComparisonReport
# ... see examples above
"
```

### Python API Quick Start

```python
from src.models.predictor import SuperAnimalPredictor

# Quick inference
predictor = SuperAnimalPredictor(model_type="topviewmouse")
results = predictor.predict_video("/path/to/video.mp4")
keypoints = results["keypoints"]
```

---

## Troubleshooting

### Low Confidence Predictions
- Check video quality and lighting
- Try `video_adapt=True` for better accuracy
- Consider fine-tuning on your data

### Out of Memory
- Reduce video resolution
- Process fewer frames at a time
- Use CPU instead of GPU (slower but less memory)

### Slow Processing
- Use GPU acceleration
- Reduce resolution
- Use fewer keypoints
- Disable video adaptation

### Model Not Found
- Check internet connection (models download automatically)
- Verify `~/.deeplabcut/models/` directory
- Try reinstalling DeepLabCut

---

## References

- [DeepLabCut Documentation](https://deeplabcut.github.io/DeepLabCut/)
- [SuperAnimal Paper](https://www.nature.com/articles/s41467-024-48792-2)
- [Model Zoo](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html)
- [Fine-tuning Guide](https://deeplabcut.github.io/DeepLabCut/docs/recipes/nn.html)
