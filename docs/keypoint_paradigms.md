# Mouse Keypoint Paradigms for Behavior Analysis

This document describes the commonly used keypoint configurations for mouse behavior analysis, based on published research and best practices.

## Overview

The number of keypoints used depends on the analysis goals, video quality, and computational constraints. SuperAnimal-TopViewMouse provides 27 keypoints, but subsets can be used for specific purposes.

## Keypoint Presets

### 1. Full (27 keypoints) - Detailed Pose Analysis

**Use cases**: Fine-grained posture analysis, grooming pattern detection, gait analysis

| Region | Keypoints | Count |
|--------|-----------|-------|
| Head | nose, left_ear, right_ear, left_ear_tip, right_ear_tip, left_eye, right_eye, head_midpoint | 8 |
| Body | neck, mid_back, mouse_center, mid_backend, mid_backend2, mid_backend3 | 6 |
| Limbs | left_shoulder, right_shoulder, left_midside, right_midside, left_hip, right_hip | 6 |
| Tail | tail_base, tail1, tail2, tail3, tail4, tail5, tail_end | 7 |

---

### 2. Standard (11 keypoints) - General Behavior Analysis

**Use cases**: Open Field Test, anxiety/depression behavioral assessment, general locomotion

```yaml
standard:
  - nose          # Head direction, exploration
  - left_ear      # Head orientation
  - right_ear     # Head orientation
  - neck          # Body start point
  - mouse_center  # Center of mass, velocity reference
  - left_shoulder # Body shape
  - right_shoulder
  - left_hip      # Lower body
  - right_hip
  - tail_base     # Body end, direction
  - tail_end      # Tail movement
```

**Suitable for**:
- Total distance traveled
- Average/max velocity
- Thigmotaxis (wall preference - anxiety index)
- Zone occupancy time
- Rearing detection (with height info)

**References**:
- [Open Field Test - PMC4354627](https://pmc.ncbi.nlm.nih.gov/articles/PMC4354627/)

---

### 3. MARS (7 keypoints) - Social Behavior Analysis

**Use cases**: Multi-animal tracking, social interaction analysis

Based on [MARS (Mouse Action Recognition System)](https://elifesciences.org/articles/63720):

```yaml
mars:
  - nose          # Sniffing detection
  - left_ear      # Head direction
  - right_ear
  - neck          # Body reference
  - left_hip      # Lower body position
  - right_hip
  - tail_base     # Body end point
```

**Suitable for**:
- Attack behavior detection
- Mounting detection
- Close investigation
- Following/chasing behavior
- Anogenital/face sniffing

---

### 4. Locomotion (6 keypoints) - Movement Analysis

**Use cases**: Gait analysis, speed tracking, stride-level analysis

Based on [Stride-level analysis - Cell Reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC8796662/):

```yaml
locomotion:
  - nose          # Movement direction
  - neck          # Upper body position
  - mouse_center  # Center reference
  - tail_base     # Most stable point - velocity surrogate
  - tail_end      # Tail dynamics
```

**Key insight**: The `tail_base` keypoint is highly stable and serves as an excellent surrogate for overall mouse speed.

**Suitable for**:
- Velocity/acceleration profiles
- Stride cycle analysis
- Lateral displacement measurement
- Path tortuosity

---

### 5. Minimal (3 keypoints) - Basic Tracking

**Use cases**: Real-time tracking, low-resolution video, fast processing, simple location tracking

```yaml
minimal:
  - nose          # Head position, direction
  - mouse_center  # Center of mass location
  - tail_base     # Body orientation
```

**Suitable for**:
- Basic position tracking
- Simple velocity calculation
- Direction of movement
- Zone entry/exit detection

**Advantages**:
- Fastest processing
- Works with lower resolution
- Robust even with partial occlusions
- Sufficient for many standard behavioral tests

---

## Behavior-Specific Recommendations

| Behavior | Minimum Keypoints | Recommended Preset |
|----------|-------------------|-------------------|
| Distance/Velocity | mouse_center OR tail_base | minimal (3) |
| Direction/Rotation | nose + tail_base | minimal (3) |
| Thigmotaxis | mouse_center | minimal (3) |
| Grooming | nose + ears + shoulders | standard (11) |
| Rearing | nose + neck + center | locomotion (6) |
| Social Interaction | nose + ears + tail_base | mars (7) |
| Detailed Gait | full spine + tail | standard (11+) |
| Fine Posture | all body parts | full (27) |

---

## Configuration

### Using in Config Files

```yaml
# configs/model/topviewmouse.yaml
use_keypoints: ${keypoint_presets.standard}  # Use preset name
# OR
use_keypoints:  # Custom list
  - nose
  - mouse_center
  - tail_base
```

### Using in Code

```python
from src.models.predictor import SuperAnimalPredictor

# Using preset
predictor = SuperAnimalPredictor(
    model_type="topviewmouse",
    use_keypoints=["nose", "mouse_center", "tail_base"]  # minimal
)

# Or use filter after prediction
filtered_kp, filtered_names = predictor.filter_keypoints(
    keypoints=full_keypoints,
    all_keypoint_names=all_names
)
```

---

## Confidence Threshold Guidelines

| Video Quality | Recommended Threshold |
|---------------|----------------------|
| High (HD, good lighting) | 0.5 - 0.7 |
| Medium | 0.3 - 0.5 |
| Low (noisy, occlusions) | 0.1 - 0.3 |

---

## References

1. **DeepLabCut SuperAnimal**: [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48792-2)
2. **MARS System**: [eLife 2021](https://elifesciences.org/articles/63720)
3. **Stride-level Analysis**: [Cell Reports 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8796662/)
4. **Keypoint-MoSeq**: [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02318-2)
5. **Open Field Test**: [J Vis Exp 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4354627/)
6. **Grooming Detection**: [eLife 2021](https://elifesciences.org/articles/63207)
