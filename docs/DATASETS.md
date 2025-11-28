# Datasets & Keypoint Presets - Complete Reference

이 프로젝트에서 사용하는 데이터셋, 키포인트 프리셋, 외부 참조 데이터셋의 종합 문서입니다.

## Table of Contents
1. [Project Datasets](#1-project-datasets) - 프로젝트 데이터셋
2. [Keypoint Presets](#2-keypoint-presets) - 키포인트 프리셋 설정
3. [External Pose Datasets](#3-external-pose-datasets) - 외부 데이터셋 레퍼런스
4. [Data Pipeline](#4-data-pipeline) - 데이터 파이프라인

---

# 1. Project Datasets

## Overview

| Dataset | Label Type | Classes | Keypoints | Use Case |
|---------|------------|---------|-----------|----------|
| **locomotion_sample** | Locomotion | 4 | 27 (TopViewMouse) | 이동 행동 분석 |
| **mars_sample** | Social | 4 | 7 (MARS format) | 사회적 상호작용 |

## 1.1 Locomotion Sample Dataset

SuperAnimal TopViewMouse 키포인트 기반 이동 행동 분류용 데이터셋

### Label System (Locomotion)
```yaml
classes: 4
class_names:
  0: stationary   # 정지 상태
  1: walking      # 걷기
  2: running      # 달리기
  3: other        # 기타
```

### Statistics
```yaml
total_samples: 1000
train_samples: 700      # 70%
val_samples: 150        # 15%
test_samples: 150       # 15%
```

### Usage
```bash
./run_evaluation.sh --dataset locomotion_sample
```

---

## 1.2 MARS Sample Dataset

MARS (Multi-Agent Rodent System) 기반 사회적 상호작용 행동 데이터셋

### Label System (Social)
```yaml
classes: 4
class_names:
  0: other          # 기타 행동
  1: attack         # 공격 행동
  2: mount          # 마운트 행동
  3: investigation  # 탐색 행동
```

### Reference
- Paper: [MARS - eLife 2021](https://elifesciences.org/articles/63720)
- Website: https://neuroethology.github.io/MARS/

---

## 1.3 Dual Label System

프로젝트는 두 가지 라벨 시스템을 자동 감지합니다:

```python
DATASET_TYPES = {
    "mars_sample": {
        "label_type": "social",
        "class_names": ["other", "attack", "mount", "investigation"],
    },
    "locomotion_sample": {
        "label_type": "locomotion",
        "class_names": ["stationary", "walking", "running", "other"],
    },
}
```

### Label Mapping (Social → Locomotion)
```python
MARS_TO_SIMPLE = {
    "other": "other",
    "attack": "running",        # High activity
    "mount": "stationary",      # Low mobility
    "investigation": "walking", # Moderate activity
}
```

---

# 2. Keypoint Presets

SuperAnimal-TopViewMouse는 27개 키포인트를 제공하지만, 분석 목적에 따라 서브셋을 사용합니다.

## 2.1 Preset Overview

| Preset | Count | Use Case | Accuracy Trade-off |
|--------|-------|----------|-------------------|
| **Full** | 27 | 정밀 자세 분석 | 최고 정확도 |
| **Standard** | 11 | Open Field Test | 균형 |
| **MARS** | 7 | 사회적 상호작용 | 다중 동물 |
| **Locomotion** | 5 | 이동/보행 분석 | 속도 중심 |
| **Minimal** | 3 | 기본 추적 | 최고 속도 |

---

## 2.2 Full (27 keypoints)

**Use cases**: 정밀 자세 분석, 그루밍 감지, 보행 분석

| Region | Keypoints | Count |
|--------|-----------|-------|
| Head | nose, left_ear, right_ear, left_ear_tip, right_ear_tip, left_eye, right_eye, head_midpoint | 8 |
| Body | neck, mid_back, mouse_center, mid_backend, mid_backend2, mid_backend3 | 6 |
| Limbs | left_shoulder, right_shoulder, left_midside, right_midside, left_hip, right_hip | 6 |
| Tail | tail_base, tail1, tail2, tail3, tail4, tail5, tail_end | 7 |

---

## 2.3 Standard (11 keypoints)

**Use cases**: Open Field Test, 불안/우울 행동 평가, 일반 이동

```yaml
standard:
  - nose          # Head direction
  - left_ear      # Head orientation
  - right_ear
  - neck          # Body start
  - mouse_center  # Center of mass
  - left_shoulder
  - right_shoulder
  - left_hip
  - right_hip
  - tail_base     # Body end
  - tail_end      # Tail movement
```

**측정 가능**: 총 이동 거리, 평균/최대 속도, Thigmotaxis, Zone 체류 시간

---

## 2.4 MARS (7 keypoints)

**Use cases**: 다중 동물 추적, 사회적 상호작용 분석

```yaml
mars:
  - nose          # Sniffing detection
  - left_ear      # Head direction
  - right_ear
  - neck          # Body reference
  - left_hip      # Lower body
  - right_hip
  - tail_base     # Body end
```

**측정 가능**: Attack, Mount, Investigation, Following/Chasing

---

## 2.5 Locomotion (5 keypoints)

**Use cases**: 보행 분석, 속도 추적, Stride 분석

```yaml
locomotion:
  - nose          # Movement direction
  - neck          # Upper body
  - mouse_center  # Center reference
  - tail_base     # Most stable - velocity proxy
  - tail_end      # Tail dynamics
```

**Key insight**: `tail_base`가 가장 안정적이며 속도 추정에 최적

---

## 2.6 Minimal (3 keypoints)

**Use cases**: 실시간 추적, 저해상도, 빠른 처리

```yaml
minimal:
  - nose          # Head position
  - mouse_center  # Center of mass
  - tail_base     # Body orientation
```

**장점**: 가장 빠름, 저해상도 호환, 부분 가림에도 견고

---

## 2.7 Behavior별 추천 Preset

| Behavior | Minimum Keypoints | Recommended Preset |
|----------|-------------------|-------------------|
| Distance/Velocity | mouse_center OR tail_base | minimal (3) |
| Direction/Rotation | nose + tail_base | minimal (3) |
| Thigmotaxis | mouse_center | minimal (3) |
| Grooming | nose + ears + shoulders | standard (11) |
| Rearing | nose + neck + center | locomotion (5) |
| Social Interaction | nose + ears + tail_base | mars (7) |
| Detailed Gait | full spine + tail | standard (11+) |
| Fine Posture | all body parts | full (27) |

---

## 2.8 Configuration

### Config Files
```yaml
# configs/model/topviewmouse.yaml
use_keypoints: ${keypoint_presets.standard}
# OR custom list
use_keypoints:
  - nose
  - mouse_center
  - tail_base
```

### Python Code
```python
from src.models.predictor import SuperAnimalPredictor

predictor = SuperAnimalPredictor(
    model_type="topviewmouse",
    use_keypoints=["nose", "mouse_center", "tail_base"]  # minimal
)
```

---

## 2.9 Confidence Threshold Guidelines

| Video Quality | Recommended Threshold |
|---------------|----------------------|
| High (HD, good lighting) | 0.5 - 0.7 |
| Medium | 0.3 - 0.5 |
| Low (noisy, occlusions) | 0.1 - 0.3 |

---

# 3. External Pose Datasets

외부에서 사용 가능한 Pose Estimation 데이터셋 레퍼런스입니다.

## 3.1 Animal Pose Datasets

| Dataset | Size | Keypoints | License | Use Case |
|---------|------|-----------|---------|----------|
| **AP-10K** | 10K images, 60 species | 17 | Apache 2.0 | Cross-species |
| **APT-36K** | 36K frames, 30 species | 17 + tracking | CC BY-NC 4.0 | Multi-animal tracking |
| **SuperAnimal** | Pre-trained | 27/39 | - | Zero-shot (프로젝트 기본) |
| **MARS** | Social behavior | 7 × 2 mice | CC BY 4.0 | Social interaction |
| **CalMS21** | Benchmark | 7 | - | Behavior classification |
| **Animal3D** | 3.4K images | 26 + 3D | - | 3D pose |

## 3.2 Human Pose Datasets

| Dataset | Size | Keypoints | Use Case |
|---------|------|-----------|----------|
| **COCO** | 250K+ persons | 17 | Industry standard |
| **COCO-WholeBody** | Same + annotations | 133 | Full body + face + hands |
| **MPII** | 25K images | 16 | Activity recognition |
| **Human3.6M** | 3.6M frames | 32 (3D) | 3D pose |

---

# 4. Data Pipeline

## Loading Data

```python
from src.data.datasets import get_dataset, get_dataset_info

# Get info
info = get_dataset_info("locomotion_sample")
print(f"Classes: {info['num_classes']}")

# Load splits
dataset = get_dataset("locomotion_sample")
X_train = dataset.get_train_data()
```

## Data Format

```python
@dataclass
class BehaviorSequence:
    video_id: str
    keypoints: np.ndarray     # (num_frames, num_keypoints, 3)
    labels: np.ndarray        # (num_frames,)
    label_names: List[str]
    fps: float = 30.0
```

## Train/Val/Test Split

```yaml
train: 70%    # Model training
val: 15%      # Hyperparameter tuning
test: 15%     # Final evaluation
```

Stratified sampling으로 모든 split에 모든 클래스가 포함됩니다.

## Download Instructions

```bash
# Project datasets (auto-download)
./run_evaluation.sh

# External datasets
python -m src.scripts.download_datasets --dataset mars
```

### Manual Download
- **MARS**: https://neuroethology.github.io/MARS/
- **CalMS21**: https://data.caltech.edu/records/s0vdx-0k302
- **AP-10K**: https://github.com/AlexTheBad/AP-10K

---

## File Locations

```
src/data/
├── datasets.py              # Dataset loaders
└── __init__.py

data/                        # Git ignored
├── locomotion_sample/       # Auto-generated
├── mars_sample/             # Manual download
└── calms21/                 # Manual download
```

---

## References

1. **SuperAnimal**: [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48792-2)
2. **MARS System**: [eLife 2021](https://elifesciences.org/articles/63720)
3. **Stride-level Analysis**: [Cell Reports 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8796662/)
4. **Keypoint-MoSeq**: [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02318-2)
5. **Open Field Test**: [J Vis Exp 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4354627/)
6. **AP-10K**: [NeurIPS 2021](https://arxiv.org/abs/2108.12617)
7. **COCO**: [ECCV 2014](https://cocodataset.org/)
