# Configuration Guide

This document describes all configuration options for the SuperAnimal Behavior Analysis Pipeline.

## Quick Reference

```bash
# Common overrides
python run.py report.gif_max_frames=150           # Longer GIFs
python run.py model.use_keypoints=null            # All keypoints
python run.py data.video.max_frames=300           # More frames
python run_cross_species.py species=[mouse,dog]   # Select species
```

---

## Configuration Files

### Main Config (configs/config.yaml)

```yaml
# ============================================================
# Default Model & Data
# ============================================================
defaults:
  - model: topviewmouse    # or quadruped
  - data: sample
  - behavior: default
  - _self_

# ============================================================
# Paths
# ============================================================
paths:
  root: ${hydra:runtime.cwd}
  data: ${paths.root}/data
  outputs: ${paths.root}/outputs
  experiments: ${paths.outputs}/experiments/${experiment_id}

# ============================================================
# General Settings
# ============================================================
seed: 42
device: auto    # auto, cuda, mps, cpu
verbose: true
input: null     # Override with video path

# ============================================================
# Species (for run_cross_species.py)
# ============================================================
species:
  - mouse
  - dog

# ============================================================
# Report Settings
# ============================================================
report:
  html: true              # Generate HTML report
  gifs: true              # Generate action GIF animations
  gifs_per_action: 2      # Max GIF samples per action type
  gif_fps: 8              # GIF animation frame rate
  gif_max_frames: 100     # Max frames per GIF (longer = more detail)
  gif_duration_sec: 4.0   # Target duration for each GIF segment
```

---

## Model Configuration

### TopViewMouse (configs/model/topviewmouse.yaml)

```yaml
name: superanimal_topviewmouse
model_name: hrnet_w32
detector: fasterrcnn_resnet50_fpn

# Model parameters
video_adapt: false
pseudo_threshold: 0.1
bbox_threshold: 0.9
confidence_threshold: 0.3

# ============================================================
# Keypoint Presets
# ============================================================
keypoint_presets:
  # Full: All 27 keypoints for detailed pose analysis
  full: null

  # Standard (11): General behavior analysis
  standard:
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

  # MARS (7): Social behavior analysis
  mars:
    - nose
    - left_ear
    - right_ear
    - neck
    - left_hip
    - right_hip
    - tail_base

  # Locomotion (6): Movement analysis
  locomotion:
    - nose
    - neck
    - mouse_center
    - tail_base
    - tail_end

  # Minimal (3): Basic tracking
  minimal:
    - nose
    - mouse_center
    - tail_base

# Active keypoint configuration (default: standard)
use_keypoints: ${model.keypoint_presets.standard}

# All 27 available keypoints
keypoints:
  - nose
  - left_ear
  - right_ear
  - left_ear_tip
  - right_ear_tip
  - left_eye
  - right_eye
  - neck
  - mid_back
  - mouse_center
  - mid_backend
  - mid_backend2
  - mid_backend3
  - tail_base
  - tail1
  - tail2
  - tail3
  - tail4
  - tail5
  - left_shoulder
  - left_midside
  - left_hip
  - right_shoulder
  - right_midside
  - right_hip
  - tail_end
  - head_midpoint

num_keypoints: 27
```

### Quadruped (configs/model/quadruped.yaml)

```yaml
name: superanimal_quadruped
model_name: hrnet_w32
detector: fasterrcnn_resnet50_fpn

# Model parameters
video_adapt: false
pseudo_threshold: 0.1
bbox_threshold: 0.9
confidence_threshold: 0.3

# ============================================================
# Keypoint Presets
# ============================================================
keypoint_presets:
  full: null    # All 39 keypoints

  # Standard (15): General pose analysis
  standard:
    - nose
    - right_eye
    - left_eye
    - right_earbase
    - left_earbase
    - neck_base
    - back_base
    - back_end
    - tail_base
    - left_front_paw
    - right_front_paw
    - left_back_paw
    - right_back_paw
    - left_front_elbow
    - right_front_elbow

  # Minimal (7): Basic tracking
  minimal:
    - nose
    - neck_base
    - back_middle
    - tail_base
    - left_front_paw
    - right_front_paw
    - left_back_paw

use_keypoints: ${model.keypoint_presets.standard}

num_keypoints: 39
```

---

## Species Configuration

Species configs define per-animal settings for cross-species comparison.

### Mouse (configs/species/mouse.yaml)

```yaml
name: mouse
description: "Mouse (top view)"

# Model configuration
model:
  type: topviewmouse
  use_keypoints: ${model.keypoint_presets.standard}

# Sample video
sample:
  name: mouse_topview

# Action classification thresholds (body-lengths/sec)
velocity_thresholds:
  stationary: 0.5   # < 0.5 = stationary
  walking: 3.0      # 0.5 - 3.0 = walking, > 3.0 = running

# Body center keypoint for trajectory
center_keypoint: mouse_center
center_fallback: [mid_back, neck]

# Species-specific behaviors
behaviors:
  grooming:
    enabled: true
    velocity_threshold: 0.3
```

### Dog (configs/species/dog.yaml)

```yaml
name: dog
description: "Dog (side view)"

model:
  type: quadruped
  use_keypoints: ${model.keypoint_presets.standard}

sample:
  name: dog_walking

velocity_thresholds:
  stationary: 0.5
  walking: 2.5

center_keypoint: back_middle
center_fallback: [neck_base, back_base]

behaviors:
  tail_wagging:
    enabled: true
    frequency_threshold: 2.0
```

### Horse (configs/species/horse.yaml)

```yaml
name: horse
description: "Horse (side view)"

model:
  type: quadruped

sample:
  name: horse_running

# Horses have longer strides
velocity_thresholds:
  stationary: 0.3
  walking: 1.5

center_keypoint: back_middle

behaviors:
  gait:
    enabled: true
    trot_threshold: 2.0
    canter_threshold: 3.5
    gallop_threshold: 5.0
```

---

## Data Configuration

### Sample Data (configs/data/sample.yaml)

```yaml
data_dir: ${paths.data}

dirs:
  raw: ${data_dir}/raw
  processed: ${data_dir}/processed

# Video processing settings
video:
  max_frames: 500     # Limit frames for PoC
  fps: null           # Use original FPS
  resize: null        # [width, height] or null

# Output format
output:
  save_video: true
  save_keypoints: true
  keypoints_format: csv   # csv, h5, json
  visualization: true
```

---

## Command-Line Examples

### Basic Usage

```bash
# Run with default settings
python run.py

# Use custom video
python run.py input=/path/to/video.mp4

# Use quadruped model for dogs
python run.py model=quadruped input=dog_video.mp4
```

### Keypoint Configuration

```bash
# Use all keypoints
python run.py model.use_keypoints=null

# Use minimal preset
python run.py 'model.use_keypoints=${model.keypoint_presets.minimal}'

# Custom keypoint list
python run.py 'model.use_keypoints=[nose,mouse_center,tail_base]'
```

### Report Settings

```bash
# Longer GIFs (more frames, longer duration)
python run.py report.gif_max_frames=150 report.gif_duration_sec=6.0

# Disable GIFs
python run.py report.gifs=false

# More GIF samples per action
python run.py report.gifs_per_action=4
```

### Cross-Species Comparison

```bash
# Compare specific species
python run_cross_species.py species=[mouse,dog]

# Include horse
python run_cross_species.py species=[mouse,dog,horse]

# Override species threshold
python run_cross_species.py species.mouse.velocity_thresholds.walking=2.5
```

### Processing Limits

```bash
# Process fewer frames (faster)
python run.py data.video.max_frames=100

# Process more frames (more detail)
python run.py data.video.max_frames=1000
```

---

## Environment Variables

```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python run.py device=cpu

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
python run.py device=cuda
```

---

## Configuration Priority

1. Command-line overrides (highest)
2. Config file values
3. Default values (lowest)

Example:
```bash
# config.yaml has gif_max_frames: 100
# This overrides to 150
python run.py report.gif_max_frames=150
```
