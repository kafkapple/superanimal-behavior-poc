# Action Recognition Models - Technical Specifications

행동 인식 모델들의 상세 기술 스펙입니다.

## Overview

| Model | Type | Input | Temporal | Training | Best Use Case |
|-------|------|-------|----------|----------|---------------|
| Rule-Based | Heuristic | Frame | No | None | Baseline, 빠른 테스트 |
| MLP | Neural Network | Frame/Window | Limited | Supervised | 단순 분류 |
| LSTM | RNN | Sequence | Yes | Supervised | 시계열 패턴 |
| Transformer | Attention | Sequence | Yes | Supervised | 장거리 의존성 |

---

## 1. Rule-Based Classifier (Baseline)

### Architecture
- 학습 없이 손으로 정의된 속도 기반 규칙 사용
- Body-length 정규화 속도로 분류

### Parameters
```yaml
fps: 30.0                    # 비디오 프레임 레이트
thresholds:
  stationary: 0.5            # body-lengths/sec 이하 → stationary
  walking: 3.0               # 0.5 ~ 3.0 → walking, 3.0+ → running
```

### Classification Rules
```
velocity < 0.5 bl/s  → stationary (class 0)
0.5 ≤ velocity < 3.0 → walking (class 1)
velocity ≥ 3.0       → running (class 2)
```

### Pros/Cons
| Pros | Cons |
|------|------|
| 학습 불필요 | 고정된 규칙 |
| 빠른 추론 | 복잡한 행동 불가 |
| 해석 가능 | 낮은 정확도 |

---

## 2. MLP Classifier

### Architecture
```
Input (input_dim * window_size)
    ↓
Linear(in, 128) → ReLU → Dropout(0.3)
    ↓
Linear(128, 64) → ReLU → Dropout(0.3)
    ↓
Linear(64, num_classes)
    ↓
Output (num_classes)
```

### Parameters
```yaml
input_dim: 14                # 7 keypoints × 2 (x, y)
hidden_dims: [128, 64]       # Hidden layer sizes
window_size: 1               # Temporal context (1 = frame-by-frame)
learning_rate: 0.001
epochs: 50
batch_size: 64
dropout: 0.3
optimizer: Adam
loss: CrossEntropyLoss
```

### Input Processing
- Window 기반 temporal context 지원
- `window_size=1`: 프레임 단위 분류
- `window_size=15`: 앞뒤 7프레임 context 포함

### Pros/Cons
| Pros | Cons |
|------|------|
| 빠른 학습/추론 | 제한된 temporal modeling |
| 적은 메모리 | 장거리 의존성 불가 |
| 안정적 성능 | 순서 정보 제한 |

---

## 3. LSTM Classifier

### Architecture
```
Input (batch, seq_len, input_dim)
    ↓
LSTM(input_dim, 128, num_layers=2, bidirectional=True)
    ↓
Output: (batch, seq_len, 256)  # 128 × 2 (bidirectional)
    ↓
Dropout(0.3)
    ↓
Linear(256, 128) → ReLU
    ↓
Dropout(0.3)
    ↓
Linear(128, num_classes)
    ↓
Output (batch, seq_len, num_classes)
```

### Parameters
```yaml
input_dim: 14                # 7 keypoints × 2 (x, y)
hidden_dim: 128              # LSTM hidden size
num_layers: 2                # LSTM layers
bidirectional: true          # 양방향 LSTM
dropout: 0.3
seq_len: 64                  # 훈련용 시퀀스 길이
learning_rate: 0.001
epochs: 50
batch_size: 32
optimizer: Adam
loss: CrossEntropyLoss
```

### Sequence Processing
- 고정 길이 시퀀스로 분할 (50% overlap)
- 추론 시 chunk 단위 처리 후 병합
- Edge padding for short sequences

### Pros/Cons
| Pros | Cons |
|------|------|
| 시계열 패턴 학습 | 긴 시퀀스에서 gradient vanishing |
| 양방향 context | 순차 처리 (병렬화 어려움) |
| 안정적 성능 | 메모리 사용량 높음 |

---

## 4. Transformer Classifier

### Architecture
```
Input (batch, seq_len, input_dim)
    ↓
Linear(input_dim, d_model) + PositionalEncoding
    ↓
TransformerEncoder(
    num_layers=3,
    d_model=128,
    nhead=4,
    dim_feedforward=256
)
    ↓
Linear(d_model, num_classes)
    ↓
Output (batch, seq_len, num_classes)
```

### Parameters
```yaml
input_dim: 14                # 7 keypoints × 2 (x, y)
d_model: 128                 # Transformer hidden size
nhead: 4                     # Attention heads
num_layers: 3                # Encoder layers
dim_feedforward: 256         # FFN hidden size
dropout: 0.1
seq_len: 64                  # 훈련용 시퀀스 길이
learning_rate: 0.0001        # 작은 learning rate
epochs: 50
batch_size: 32
optimizer: Adam
loss: CrossEntropyLoss
```

### Positional Encoding
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Pros/Cons
| Pros | Cons |
|------|------|
| 장거리 의존성 | 많은 데이터 필요 |
| 병렬 처리 | 높은 계산 비용 |
| Self-attention | Small dataset에서 불안정 |

---

## Model Comparison Results

### Locomotion Dataset (1000 samples)

| Model | Accuracy | F1 Macro | Training Time |
|-------|----------|----------|---------------|
| **LSTM** | **95.3%** | **95.4%** | ~60s |
| MLP | 92.7% | 92.0% | ~30s |
| Transformer | 82.8% | 82.5% | ~90s |
| Rule-Based | 32.0% | 12.1% | 0s |

### Per-Class F1 Scores

| Model | Stationary | Walking | Running | Other |
|-------|------------|---------|---------|-------|
| LSTM | 0.93 | 0.96 | 0.96 | 0.96 |
| MLP | 0.92 | 0.95 | 0.91 | 0.90 |
| Transformer | 0.67 | 0.86 | 0.83 | 0.94 |
| Rule-Based | 0.48 | 0.00 | 0.00 | 0.00 |

### Keypoint Preset Comparison

| Preset | LSTM Acc | MLP Acc | Transformer Acc |
|--------|----------|---------|-----------------|
| full (27) | 95.3% | 92.7% | 82.8% |
| minimal (3) | **96.1%** | 76.0% | 85.2% |
| locomotion (5) | 94.5% | 82.7% | 68.8% |

---

## Training Details

### Data Preparation
```python
# Train/Val/Test Split
train: 70%
val: 15%
test: 15%

# Stratified sampling
- 각 split에 모든 클래스 포함
- Shuffled indices로 class imbalance 방지
```

### Training Loop
```python
# Common settings
- Early stopping: patience=10 (not implemented in current version)
- Learning rate scheduler: None (future work)
- Gradient clipping: None (future work)
```

### Feature Engineering
```python
# Input features per frame
- Keypoint coordinates: (x, y) × num_keypoints
- Optional: velocity, acceleration (computed internally)

# Normalization
- Min-max scaling per sequence
- Body-length normalization for velocity
```

---

## Usage Examples

### Training
```python
from src.models.action_models import LSTMClassifier

model = LSTMClassifier(
    num_classes=4,
    class_names=['stationary', 'walking', 'running', 'other'],
    input_dim=14,
    hidden_dim=128,
    epochs=50,
)

history = model.fit(X_train, y_train, X_val, y_val)
```

### Inference
```python
result = model.predict(X_test)
print(f"Predictions: {result.predictions}")
print(f"Probabilities: {result.probabilities}")
```

### Evaluation
```python
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"F1 Macro: {metrics.f1_macro:.4f}")
```

---

## File Locations

```
src/models/
├── action_models.py         # 모든 모델 정의
│   ├── BaseActionClassifier # 추상 베이스 클래스
│   ├── RuleBasedClassifier  # 규칙 기반
│   ├── MLPClassifier        # MLP
│   ├── LSTMClassifier       # LSTM
│   └── TransformerClassifier # Transformer
└── __init__.py              # 모듈 export

src/evaluation/
├── comprehensive_evaluation.py  # 모델 비교 평가
└── metrics.py                   # 평가 메트릭

outputs/evaluation/
├── evaluation_results.json      # 전체 결과
└── models/                      # 학습된 가중치
    ├── full_mlp.pt
    ├── full_lstm.pt
    └── full_transformer.pt
```
