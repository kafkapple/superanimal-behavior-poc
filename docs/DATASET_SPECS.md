# Behavior Datasets - Technical Specifications

행동 분석 데이터셋들의 상세 기술 스펙입니다.

## Overview

| Dataset | Label Type | Classes | Keypoints | Split | Source |
|---------|------------|---------|-----------|-------|--------|
| locomotion_sample | Locomotion | 4 | 27 (TopViewMouse) | Train/Val/Test | Synthetic |
| mars_sample | Social | 4 | 7 (MARS format) | Train/Val/Test | MARS |
| CalMS21 | Social | 4 | 7 | Train/Val/Test | Caltech |

---

## 1. Locomotion Sample Dataset

### Description
SuperAnimal TopViewMouse 키포인트 기반 이동 행동 분류용 합성 데이터셋

### Label System
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

train_distribution:
  stationary: ~213 (30%)
  walking: ~195 (28%)
  running: ~144 (21%)
  other: ~148 (21%)
```

### Keypoint Schema (Full - 27 points)
```yaml
keypoints:
  - nose
  - left_ear
  - right_ear
  - left_ear_tip
  - right_ear_tip
  - left_eye
  - right_eye
  - head_midpoint
  - neck
  - mid_back
  - mouse_center
  - mid_backend
  - mid_backend2
  - mid_backend3
  - tailbase
  - tail1
  - tail2
  - tail3
  - tail4
  - tailend
  - left_front_paw
  - right_front_paw
  - left_rear_paw
  - right_rear_paw
  - left_front_paw_tip
  - right_front_paw_tip
  - left_rear_paw_tip
```

### Keypoint Presets

| Preset | Count | Keypoints | Use Case |
|--------|-------|-----------|----------|
| full | 27 | All | 정밀 분석 |
| minimal | 3 | nose, tailbase, tailend | 빠른 처리 |
| locomotion | 5 | nose, neck, mid_back, tailbase, tailend | 이동 분석 |

### Data Format
```json
{
  "keypoints": [
    [[x1, y1, conf1], [x2, y2, conf2], ...]  // Frame 1
  ],
  "labels": [0, 1, 1, 2, ...],               // Per-frame labels
  "label_map": {
    "0": "stationary",
    "1": "walking",
    "2": "running",
    "3": "other"
  }
}
```

### File Structure
```
data/locomotion_sample/
├── train/
│   ├── keypoints.npy       # (N, num_keypoints, 3)
│   ├── labels.npy          # (N,)
│   └── annotations.json    # Metadata + label_map
├── val/
│   └── ...
└── test/
    └── ...
```

---

## 2. MARS Sample Dataset

### Description
MARS (Multi-Agent Rodent System) 기반 사회적 상호작용 행동 데이터셋

### Label System
```yaml
classes: 4
class_names:
  0: other          # 기타 행동
  1: attack         # 공격 행동
  2: mount          # 마운트 행동
  3: investigation  # 탐색 행동
```

### Reference
- Paper: [MARS: Learning from unlabeled behavioral video](https://neuroethology.github.io/MARS/)
- Data: Requires download from official source

### Keypoint Schema (MARS - 7 points)
```yaml
keypoints:
  - nose
  - left_ear
  - right_ear
  - neck
  - left_side
  - right_side
  - tailbase
```

### Dual-Mouse Format
```json
{
  "mouse1_keypoints": [...],  // Resident mouse
  "mouse2_keypoints": [...],  // Intruder mouse
  "labels": [...]             // Interaction labels
}
```

---

## 3. CalMS21 Dataset

### Description
Caltech Mouse Social Behavior (CalMS21) 벤치마크 데이터셋

### Label System
```yaml
classes: 4
class_names:
  0: other
  1: attack
  2: mount
  3: investigation
```

### Reference
- Paper: [CalMS21: A Multi-Agent Behavior Challenge](https://data.caltech.edu/records/s0vdx-0k302)
- Benchmark: CVPR 2021 MABe Challenge

### Download
```bash
# 공식 다운로드 (수동)
# https://data.caltech.edu/records/s0vdx-0k302

# 스크립트 사용
python -m src.scripts.download_datasets --dataset calms21
```

---

## Dual Label System Architecture

### Label Type Detection
```python
DATASET_TYPES = {
    "mars_sample": {
        "label_type": "social",
        "num_classes": 4,
        "class_names": ["other", "attack", "mount", "investigation"],
    },
    "locomotion_sample": {
        "label_type": "locomotion",
        "num_classes": 4,
        "class_names": ["stationary", "walking", "running", "other"],
    },
}
```

### Label Mapping (Social → Locomotion)
```python
MARS_TO_SIMPLE = {
    MARSBehavior.OTHER: SimpleBehavior.OTHER,
    MARSBehavior.ATTACK: SimpleBehavior.RUNNING,      # High activity
    MARSBehavior.MOUNT: SimpleBehavior.STATIONARY,    # Low mobility
    MARSBehavior.INVESTIGATION: SimpleBehavior.WALKING,  # Moderate
}
```

---

## Data Pipeline

### Loading
```python
from src.data.datasets import get_dataset, get_dataset_info

# Auto-detect label type
info = get_dataset_info("locomotion_sample")
print(f"Classes: {info['class_names']}")

# Load data
dataset = get_dataset("locomotion_sample")
X_train, y_train = dataset.get_train_data()
```

### Preprocessing
```python
# 1. Stratified Split
# - 모든 split에 모든 클래스 포함
# - Shuffled indices로 균형 유지

# 2. Feature Extraction
# - Keypoint coordinates (x, y)
# - Confidence scores
# - Optional: velocity, acceleration

# 3. Normalization
# - Min-max scaling per sequence
# - Body-length normalization
```

### Augmentation (Future)
```python
# Planned augmentations
- Random scaling
- Random rotation
- Temporal jittering
- Keypoint dropout
```

---

## Evaluation Protocol

### Train/Val/Test Split
```yaml
train: 70%    # Model training
val: 15%      # Hyperparameter tuning
test: 15%     # Final evaluation (held-out)
```

### Stratification
```python
# Ensure all classes in each split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
```

### Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| F1 Macro | Mean F1 across all classes |
| F1 per Class | Class-wise F1 scores |
| Confusion Matrix | True vs Predicted |

---

## File Locations

```
src/data/
├── datasets.py              # Dataset loaders
│   ├── MARSDatasetLoader    # MARS format
│   ├── CalMS21DatasetLoader # CalMS21 format
│   ├── download_locomotion_sample()  # Sample data generator
│   ├── get_dataset()        # Unified loader
│   └── get_dataset_info()   # Dataset metadata
└── __init__.py

data/
├── locomotion_sample/       # Generated on first run
│   ├── train/
│   ├── val/
│   └── test/
├── mars_sample/             # Download required
└── calms21/                 # Download required

configs/data/
└── datasets.yaml            # Dataset configurations
```

---

## Usage Examples

### Load Dataset
```python
from src.data.datasets import get_dataset, get_dataset_info

# Get info
info = get_dataset_info("locomotion_sample")
print(f"Classes: {info['num_classes']}")
print(f"Names: {info['class_names']}")

# Load splits
dataset = get_dataset("locomotion_sample")
print(f"Train: {len(dataset.train)} sequences")
print(f"Test: {len(dataset.test)} sequences")
```

### Access Data
```python
# Get arrays
X_train = np.concatenate([seq.keypoints for seq in dataset.train])
y_train = np.concatenate([seq.labels for seq in dataset.train])

print(f"X shape: {X_train.shape}")  # (N, num_keypoints, 3)
print(f"y shape: {y_train.shape}")  # (N,)
```

### Create Custom Dataset
```python
from src.data.datasets import BehaviorSequence, DatasetSplit

# Create sequence
seq = BehaviorSequence(
    video_id="my_video",
    keypoints=keypoints,  # (T, K, 3)
    labels=labels,        # (T,)
    label_names=["stationary", "walking", "running", "other"],
    fps=30.0,
)

# Create split
split = DatasetSplit(
    train=[seq1, seq2],
    val=[seq3],
    test=[seq4],
    label_map={0: "stationary", 1: "walking", 2: "running", 3: "other"},
)
```

---

## Download & Setup

### Quick Start
```bash
# Locomotion sample (auto-generated)
python run_evaluation.py --dataset locomotion_sample

# MARS sample (requires download)
python -m src.scripts.download_datasets --dataset mars
python run_evaluation.py --dataset mars_sample
```

### Manual Download
```bash
# MARS Dataset
# 1. Visit: https://neuroethology.github.io/MARS/
# 2. Download annotation files
# 3. Place in data/mars_sample/

# CalMS21 Dataset
# 1. Visit: https://data.caltech.edu/records/s0vdx-0k302
# 2. Download zip files
# 3. Extract to data/calms21/
```
