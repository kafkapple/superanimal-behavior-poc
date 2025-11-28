# Models - Technical Specifications

이 프로젝트에서 사용하는 모든 모델의 상세 기술 스펙입니다.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Full Analysis Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Video Input                                                            │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────┐               │
│   │        STAGE 1: Keypoint Extraction                  │               │
│   │        (Pose Estimation Models)                      │               │
│   │                                                      │               │
│   │   ┌─────────────┐    ┌─────────────┐                │               │
│   │   │ SuperAnimal │ OR │ YOLO Pose   │                │               │
│   │   │ (DLC 3.0)   │    │ (YOLOv8)    │                │               │
│   │   └─────────────┘    └─────────────┘                │               │
│   │                                                      │               │
│   │   Output: Keypoints (frames, num_kp, 3)             │               │
│   └─────────────────────────────────────────────────────┘               │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────┐               │
│   │        STAGE 2: Action Recognition                   │               │
│   │        (Behavior Classification Models)              │               │
│   │                                                      │               │
│   │   ┌───────────┐ ┌─────┐ ┌──────┐ ┌───────────┐     │               │
│   │   │Rule-Based │ │ MLP │ │ LSTM │ │Transformer│     │               │
│   │   │(Baseline) │ │     │ │      │ │           │     │               │
│   │   └───────────┘ └─────┘ └──────┘ └───────────┘     │               │
│   │                                                      │               │
│   │   Output: Behavior Labels (stationary/walking/...)   │               │
│   └─────────────────────────────────────────────────────┘               │
│       │                                                                  │
│       ▼                                                                  │
│   Visualization & Reports                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# Part 1: Keypoint Extraction Models (Pose Estimation)

비디오에서 동물의 키포인트(관절점)를 추출하는 모델들입니다.

## 1.1 SuperAnimal (DeepLabCut 3.0)

### Overview
| 항목 | 내용 |
|------|------|
| **Framework** | DeepLabCut 3.0 |
| **Architecture** | HRNet-W32 + Faster R-CNN |
| **Pre-trained** | 45+ 종의 동물 데이터 |
| **Zero-shot** | Fine-tuning 없이 사용 가능 |
| **Paper** | [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48792-2) |

### Supported Model Types

| Model Type | Keypoints | View | Target Species |
|------------|-----------|------|----------------|
| **TopViewMouse** | 27 | Top-down | Mouse, Rat |
| **Quadruped** | 39 | Side | Dog, Cat, Horse, 45+ species |

### Architecture Details

```
┌─────────────────────────────────────────────────────────────┐
│                    SuperAnimal Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Frame (H × W × 3)                                   │
│       │                                                      │
│       ▼                                                      │
│   ┌───────────────────────────────┐                         │
│   │  Detector: Faster R-CNN       │                         │
│   │  Backbone: ResNet50-FPN-v2    │                         │
│   │  Output: Bounding Boxes       │                         │
│   └───────────────────────────────┘                         │
│       │                                                      │
│       ▼                                                      │
│   ┌───────────────────────────────┐                         │
│   │  Pose Estimator: HRNet-W32    │                         │
│   │  Multi-scale Feature Fusion   │                         │
│   │  Output: Keypoint Heatmaps    │                         │
│   └───────────────────────────────┘                         │
│       │                                                      │
│       ▼                                                      │
│   Keypoints: (num_animals, num_keypoints, 3)                │
│              [x, y, confidence]                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Parameters

```yaml
# TopViewMouse
superanimal_name: superanimal_topviewmouse
model_name: hrnet_w32
detector_name: fasterrcnn_resnet50_fpn_v2
num_keypoints: 27

# Quadruped
superanimal_name: superanimal_quadruped
model_name: hrnet_w32
detector_name: fasterrcnn_resnet50_fpn_v2
num_keypoints: 39

# Common options
video_adapt: false        # True for better accuracy (slower)
scale_list: null          # Multi-scale inference
device: auto              # cuda, mps, cpu
```

### TopViewMouse Keypoints (27)

| Region | Keypoints |
|--------|-----------|
| Head (8) | nose, left_ear, right_ear, left_ear_tip, right_ear_tip, left_eye, right_eye, head_midpoint |
| Body (6) | neck, mid_back, mouse_center, mid_backend, mid_backend2, mid_backend3 |
| Tail (7) | tailbase, tail1, tail2, tail3, tail4, tailend |
| Limbs (6) | left_front_paw, right_front_paw, left_rear_paw, right_rear_paw, left_front_paw_tip, right_front_paw_tip |

### Quadruped Keypoints (39)

| Region | Keypoints |
|--------|-----------|
| Head | nose, left_eye, right_eye, left_earbase, right_earbase, chin |
| Spine | neck, throat, withers, back, belly, tail_base, tail1, tail2, tailend |
| Front Legs | left/right_front_elbow, left/right_front_knee, left/right_front_paw |
| Back Legs | left/right_back_elbow, left/right_back_knee, left/right_back_paw |

### Usage

```python
from src.models.predictor import SuperAnimalPredictor

predictor = SuperAnimalPredictor(
    model_type="topviewmouse",  # or "quadruped"
    device="auto",
)

results = predictor.predict_video(
    video_path="video.mp4",
    output_dir="output/",
    max_frames=500,
)

keypoints = results["keypoints"]  # (frames, 27, 3)
```

### File Location
```
src/models/predictor.py          # SuperAnimalPredictor class
~/.deeplabcut/models/            # Downloaded model weights
```

---

## 1.2 YOLO Pose (YOLOv8)

### Overview
| 항목 | 내용 |
|------|------|
| **Framework** | Ultralytics YOLOv8 |
| **Architecture** | YOLO + Pose Head |
| **Pre-trained** | COCO (Human), Custom (Animal) |
| **Speed** | Real-time capable |
| **Install** | `pip install ultralytics` |

### Supported Models

| Model | Size | mAP | Speed |
|-------|------|-----|-------|
| yolov8n-pose | 6.7M | 50.4 | Fastest |
| yolov8s-pose | 11.6M | 60.0 | Fast |
| yolov8m-pose | 26.4M | 65.0 | Medium |
| yolov8l-pose | 44.4M | 67.6 | Slow |

### YOLO Keypoints (17 - Human/Generic)

```yaml
keypoints:
  - nose
  - left_eye, right_eye
  - left_ear, right_ear
  - left_shoulder, right_shoulder
  - left_elbow, right_elbow
  - left_wrist, right_wrist
  - left_hip, right_hip
  - left_knee, right_knee
  - left_ankle, right_ankle
```

### Usage

```python
from src.models.yolo_pose import YOLOPosePredictor

predictor = YOLOPosePredictor(
    model_name="yolov8n-pose",
    device="auto",
    conf_threshold=0.5,
)

results = predictor.predict_video(
    video_path="video.mp4",
    output_dir="output/",
    max_frames=500,
)
```

### SuperAnimal vs YOLO Pose Comparison

| Feature | SuperAnimal | YOLO Pose |
|---------|-------------|-----------|
| **Keypoints (Mouse)** | 27 | 17 (needs mapping) |
| **Keypoints (Quadruped)** | 39 | 17 (needs mapping) |
| **Zero-shot Animals** | Excellent | Limited |
| **Speed** | Moderate | Fast |
| **Accuracy (Animals)** | High | Moderate |
| **Pre-trained Species** | 45+ | Human-centric |

### File Location
```
src/models/yolo_pose.py          # YOLOPosePredictor class
```

---

# Part 2: Action Recognition Models (Behavior Classification)

키포인트에서 행동을 분류하는 모델들입니다.

## Model Summary

| Model | Type | Input | Temporal | Training | Best Accuracy |
|-------|------|-------|----------|----------|---------------|
| **Rule-Based** | Heuristic | Frame | No | None | 32.0% |
| **MLP** | Neural Net | Frame/Window | Limited | Supervised | 92.7% |
| **LSTM** | RNN | Sequence | Yes | Supervised | **96.1%** |
| **Transformer** | Attention | Sequence | Yes | Supervised | 85.2% |

---

## 2.1 Rule-Based Classifier (Baseline)

### Overview
학습 없이 속도 기반 규칙으로 행동을 분류하는 베이스라인 모델입니다.

### Classification Rules

```
┌─────────────────────────────────────────────────────────────┐
│              Velocity-Based Classification                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Normalized Velocity (body-length/sec)                     │
│                                                              │
│   0 ────────┬────────────────────┬────────────────→ ∞      │
│             │                    │                          │
│        < 0.5 BL/s          0.5-3.0 BL/s        > 3.0 BL/s  │
│             │                    │                    │     │
│        stationary            walking             running    │
│         (class 0)           (class 1)          (class 2)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Parameters

```yaml
fps: 30.0
thresholds:
  stationary: 0.5    # body-lengths/sec
  walking: 3.0       # body-lengths/sec
```

### Pros/Cons

| Pros | Cons |
|------|------|
| 학습 불필요 | 고정된 규칙 |
| 빠른 추론 | 복잡한 행동 불가 |
| 해석 가능 | 낮은 정확도 (32%) |

### Usage

```python
from src.models.action_models import RuleBasedClassifier

model = RuleBasedClassifier(
    num_classes=4,
    fps=30.0,
    thresholds={"stationary": 0.5, "walking": 3.0},
)

result = model.predict(X_test)
```

---

## 2.2 MLP Classifier

### Architecture

```
Input (input_dim × window_size)
    │
    ▼
Linear(in, 128) → ReLU → Dropout(0.3)
    │
    ▼
Linear(128, 64) → ReLU → Dropout(0.3)
    │
    ▼
Linear(64, num_classes)
    │
    ▼
Output (num_classes)
```

### Parameters

```yaml
input_dim: 14           # 7 keypoints × 2 (x, y)
hidden_dims: [128, 64]
window_size: 1          # 1 = frame-by-frame
learning_rate: 0.001
epochs: 50
batch_size: 64
dropout: 0.3
optimizer: Adam
loss: CrossEntropyLoss
```

### Performance

| Keypoint Preset | Accuracy | F1 Macro |
|-----------------|----------|----------|
| full (27) | 92.7% | 92.0% |
| minimal (3) | 76.0% | 75.7% |
| locomotion (5) | 82.7% | 78.7% |

---

## 2.3 LSTM Classifier

### Architecture

```
Input (batch, seq_len, input_dim)
    │
    ▼
LSTM(input_dim=14, hidden=128, layers=2, bidirectional=True)
    │
    ▼
Output: (batch, seq_len, 256)  # 128 × 2
    │
    ▼
Dropout(0.3) → Linear(256, 128) → ReLU
    │
    ▼
Dropout(0.3) → Linear(128, num_classes)
    │
    ▼
Output (batch, seq_len, num_classes)
```

### Parameters

```yaml
input_dim: 14
hidden_dim: 128
num_layers: 2
bidirectional: true
dropout: 0.3
seq_len: 64
learning_rate: 0.001
epochs: 50
batch_size: 32
optimizer: Adam
loss: CrossEntropyLoss
```

### Performance (Best Model)

| Keypoint Preset | Accuracy | F1 Macro |
|-----------------|----------|----------|
| full (27) | 95.3% | 95.4% |
| **minimal (3)** | **96.1%** | **95.6%** |
| locomotion (5) | 94.5% | 95.0% |

### Per-Class F1 (minimal_lstm)

| Class | F1 Score |
|-------|----------|
| stationary | 0.957 |
| walking | 0.962 |
| running | 0.923 |
| other | 0.982 |

---

## 2.4 Transformer Classifier

### Architecture

```
Input (batch, seq_len, input_dim)
    │
    ▼
Linear(input_dim, d_model) + PositionalEncoding
    │
    ▼
TransformerEncoder(
    num_layers=3,
    d_model=128,
    nhead=4,
    dim_feedforward=256
)
    │
    ▼
Linear(d_model, num_classes)
    │
    ▼
Output (batch, seq_len, num_classes)
```

### Parameters

```yaml
input_dim: 14
d_model: 128
nhead: 4
num_layers: 3
dim_feedforward: 256
dropout: 0.1
seq_len: 64
learning_rate: 0.0001    # Smaller LR
epochs: 50
batch_size: 32
```

### Performance

| Keypoint Preset | Accuracy | F1 Macro |
|-----------------|----------|----------|
| full (27) | 82.8% | 82.5% |
| minimal (3) | 85.2% | 86.4% |
| locomotion (5) | 68.8% | 65.2% |

**Note**: Transformer는 작은 데이터셋에서 LSTM보다 성능이 낮음. 더 많은 데이터 필요.

---

## Model Comparison Summary

### Overall Performance

```
┌─────────────────────────────────────────────────────────────┐
│                  Model Performance Comparison                │
├─────────────────┬──────────┬─────────┬─────────────────────┤
│ Model           │ Accuracy │ F1      │ Training Time       │
├─────────────────┼──────────┼─────────┼─────────────────────┤
│ minimal_lstm    │ 96.1%    │ 95.6%   │ ~60s                │
│ full_lstm       │ 95.3%    │ 95.4%   │ ~60s                │
│ full_mlp        │ 92.7%    │ 92.0%   │ ~30s                │
│ minimal_trans   │ 85.2%    │ 86.4%   │ ~90s                │
│ full_trans      │ 82.8%    │ 82.5%   │ ~90s                │
│ rule_based      │ 32.0%    │ 12.1%   │ 0s                  │
└─────────────────┴──────────┴─────────┴─────────────────────┘
```

### Key Findings

1. **LSTM이 최고 성능**: 시계열 패턴 학습에 효과적
2. **Minimal keypoints로 충분**: 3개 키포인트로도 96% 정확도 달성
3. **Rule-based는 baseline only**: 학습 기반 모델 대비 크게 낮음
4. **Transformer는 더 많은 데이터 필요**: 현재 데이터셋에서 underfitting

---

## File Locations

```
src/models/
├── predictor.py         # SuperAnimalPredictor (Keypoint)
├── yolo_pose.py         # YOLOPosePredictor (Keypoint)
├── action_models.py     # Action Recognition Models
│   ├── BaseActionClassifier
│   ├── RuleBasedClassifier
│   ├── MLPClassifier
│   ├── LSTMClassifier
│   └── TransformerClassifier
├── action_classifier.py # UnifiedActionClassifier (rule-based wrapper)
└── baseline.py          # Additional baselines

src/evaluation/
├── comprehensive_evaluation.py  # Model comparison
└── metrics.py                   # Evaluation metrics

outputs/evaluation/
├── evaluation_results.json      # Results
└── models/                      # Trained weights
    ├── full_mlp.pt
    ├── full_lstm.pt
    └── full_transformer.pt
```

---

## Usage Examples

### Full Pipeline

```python
# Stage 1: Keypoint Extraction
from src.models.predictor import SuperAnimalPredictor

keypoint_model = SuperAnimalPredictor(model_type="topviewmouse")
results = keypoint_model.predict_video("video.mp4")
keypoints = results["keypoints"]

# Stage 2: Action Recognition
from src.models.action_models import LSTMClassifier

action_model = LSTMClassifier(
    num_classes=4,
    class_names=['stationary', 'walking', 'running', 'other'],
)

# Train
action_model.fit(X_train, y_train, X_val, y_val)

# Predict
result = action_model.predict(keypoints)
print(f"Predictions: {result.predictions}")
```

### Evaluation

```python
metrics = action_model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"F1 Macro: {metrics.f1_macro:.4f}")
print(f"Per-class F1: {metrics.f1_per_class}")
```
