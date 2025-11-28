# 이론적 배경 및 파이프라인 구조

## 목차
1. [개요](#개요)
2. [핵심 이론](#핵심-이론)
3. [파이프라인 구조](#파이프라인-구조)
4. [핵심 모듈](#핵심-모듈)
5. [메트릭 및 평가](#메트릭-및-평가)
6. [모델 비교](#모델-비교)

---

## 개요

이 프로젝트는 **DeepLabCut 3.0의 SuperAnimal** 모델을 활용하여 동물 행동을 분석합니다.

### 핵심 아이디어

```
비디오 입력 → 키포인트 추출 → 궤적 분석 → 행동 분류 → 시각화/보고서
```

**왜 SuperAnimal인가?**
- 45+ 종의 동물에서 사전훈련된 foundation model
- Zero-shot 추론 가능 (fine-tuning 없이 사용)
- Body-size 정규화로 종간 비교 가능

---

## 핵심 이론

### 1. Pose Estimation (자세 추정)

#### 1.1 Top-Down vs Bottom-Up

```
┌─────────────────────────────────────────────────────────────────┐
│                     Top-Down Approach                            │
│  (SuperAnimal, YOLO Pose 사용)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [입력 이미지] → [객체 탐지] → [각 객체별 키포인트 추정]          │
│                     │                    │                       │
│                     ▼                    ▼                       │
│              Bounding Box          HRNet/ResNet                  │
│           (Faster R-CNN)         (Keypoint Head)                │
│                                                                  │
│   장점: 정확도 높음, 다중 객체 처리 용이                          │
│   단점: 탐지 실패시 키포인트도 실패                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Bottom-Up Approach                            │
│  (OpenPose 스타일)                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [입력 이미지] → [모든 키포인트 탐지] → [객체별 그룹화]           │
│                         │                    │                   │
│                         ▼                    ▼                   │
│                   Part Affinity          Greedy                  │
│                     Fields             Matching                  │
│                                                                  │
│   장점: 객체 수에 무관한 속도                                     │
│   단점: 그룹화 오류 가능                                          │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.2 HRNet (High-Resolution Network)

SuperAnimal이 사용하는 백본 네트워크:

```
                    ┌───────────────────┐
                    │   Input Image     │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Stage 1 │     │ Stage 2 │     │ Stage 3 │
        │  1/4    │ ←→  │  1/8    │ ←→  │  1/16   │
        │ (High)  │     │ (Med)   │     │ (Low)   │
        └─────────┘     └─────────┘     └─────────┘
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Multi-Scale      │
                    │  Feature Fusion   │
                    └───────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Keypoint Heatmap │
                    │  (27 or 39 ch)    │
                    └───────────────────┘

핵심: 고해상도 표현을 유지하면서 다중 스케일 정보 융합
```

### 2. Body-Size Normalization (체크기 정규화)

#### 2.1 왜 필요한가?

```
문제: 절대 픽셀 속도로는 종간 비교 불가

Mouse (100px body):  10 px/frame = 빠름 (0.1 body-length/frame)
Horse (500px body):  10 px/frame = 느림 (0.02 body-length/frame)

해결: Body-length 단위로 정규화

             실제 속도 (px/frame)
정규화 속도 = ─────────────────────
             Body Size (px)

결과: 종과 무관하게 행동 비교 가능
```

#### 2.2 Body Size 추정

```python
# TopViewMouse: nose → tail_base 거리
body_size = distance(keypoints["nose"], keypoints["tail_base"])

# Quadruped: nose → tail_base 거리
body_size = distance(keypoints["nose"], keypoints["tail_base"])

# 프레임별 중앙값으로 노이즈 제거
body_size_stable = np.median(body_sizes_per_frame)
```

### 3. Action Classification (행동 분류)

#### 3.1 Velocity-Based Rule Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                 Velocity-Based Classification                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   정규화 속도 (body-length/sec)                                  │
│                                                                  │
│   0 ────────┬────────────────────┬────────────────────→ ∞       │
│             │                    │                               │
│        stationary            walking               running       │
│      (< 0.5 BL/s)        (0.5-3.0 BL/s)         (> 3.0 BL/s)   │
│                                                                  │
│   임계값은 종별로 조정 가능 (configs/species/*.yaml)             │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2 Temporal Smoothing

```
문제: 프레임별 분류는 노이즈 발생 (떨림, 오검출)

해결: Moving Average Filter

                   1   n
    v_smooth(t) = ─── Σ v(t-i)
                   n  i=0

    n = smoothing_window (기본값: 5 frames)

효과:
- 급격한 행동 변화 완화
- 실제 행동 전환만 반영
- Consistency Score 향상
```

### 4. Cross-Species Comparison (종간 비교)

```
┌─────────────────────────────────────────────────────────────────┐
│                  Cross-Species Normalization                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Mouse (TopViewMouse)        Dog (Quadruped)                    │
│  ┌─────────────────┐         ┌─────────────────┐                │
│  │ 27 keypoints    │         │ 39 keypoints    │                │
│  │ Top view        │         │ Side view       │                │
│  │ ~100px body     │         │ ~300px body     │                │
│  └────────┬────────┘         └────────┬────────┘                │
│           │                           │                          │
│           ▼                           ▼                          │
│  ┌─────────────────────────────────────────────┐                │
│  │         Body-Size Normalization              │                │
│  │         velocity → body-length/sec           │                │
│  └─────────────────────┬───────────────────────┘                │
│                        │                                         │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────┐                │
│  │      Unified Action Classification           │                │
│  │   stationary | walking | running | other     │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  결과: 종과 무관하게 동일한 기준으로 행동 비교 가능               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 파이프라인 구조

### 실행 모드

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_all.sh - 실행 모드                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  기본 모드 (./run_all.sh)                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │   Debug     │ →  │    Full     │ →  │   Model     │ →  │  Report  │ │
│  │   Mode      │    │   Mode      │    │ Comparison  │    │  Gen     │ │
│  │  (50 frm)   │    │ (300 frm)   │    │             │    │          │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│  - 단일 종 (mouse)                                                       │
│  - 단일 프리셋 (standard)                                                │
│  - 예상 시간: ~10분                                                      │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Comprehensive 모드 (./run_all.sh -c)                                    │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐ │
│  │ Debug   │ → │  Full   │ → │ Model   │ → │ Compre- │ → │  Report  │ │
│  │         │   │         │   │ Compare │   │ hensive │   │          │ │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └──────────┘ │
│                                                 │                       │
│                    ┌───────────────────────────┴────────────────┐      │
│                    │          Comprehensive Analysis            │      │
│                    ├────────────────────────────────────────────┤      │
│                    │  Species:    Mouse, Dog, Horse (3종)       │      │
│                    │  Presets:    Full, Standard, MARS,         │      │
│                    │              Locomotion, Minimal (5개)     │      │
│                    │  Models:     SuperAnimal, YOLO, Baselines  │      │
│                    │  조합:       15개 (3×5) + 모델 비교         │      │
│                    │  예상 시간:  ~30분                          │      │
│                    └────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

실행 명령어:
  ./run_all.sh          기본 모드 (Debug → Full → Model Comparison)
  ./run_all.sh -d       Debug만 (~2분)
  ./run_all.sh -f       Full만, Debug 스킵 (~8분)
  ./run_all.sh -c       Comprehensive 전체 분석 (~30분)

주의: Comprehensive 모드는 기본이 아님. 명시적으로 -c 플래그 필요.
```

### 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        run_all.sh - Full Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │   Debug     │ →  │    Full     │ →  │   Model     │ →  │  Report  │ │
│  │   Mode      │    │   Mode      │    │ Comparison  │    │  Gen     │ │
│  │  (50 frm)   │    │ (300 frm)   │    │             │    │          │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│         │                  │                  │                  │      │
│         ▼                  ▼                  ▼                  ▼      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Stage 1: run.py                              │   │
│  │  Video → Keypoints → Behavior → GIFs → HTML Report              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                Stage 2: run_keypoint_comparison.py               │   │
│  │  Full(27) vs Standard(11) vs Minimal(3) Keypoint Comparison     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                Stage 3: run_cross_species.py                     │   │
│  │  Mouse vs Dog vs Horse - Action & Body Size Comparison          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               Stage 4: run_model_comparison.py                   │   │
│  │  SuperAnimal vs YOLO vs Baselines - Quantitative Metrics        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
Video/Images
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. Keypoint Extraction                        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │ SuperAnimal │   │  YOLO Pose  │   │   MMPose    │            │
│  │ (Primary)   │   │ (Compare)   │   │ (Optional)  │            │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘            │
│         └─────────────────┼─────────────────┘                    │
│                           ▼                                      │
│              keypoints: (frames, num_kp, 3)                      │
│              [x, y, confidence]                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. Trajectory Analysis                         │
│                                                                  │
│  Center Extraction:                                              │
│    mouse: mouse_center                                           │
│    quadruped: back_middle                                        │
│                                                                  │
│  Body Size Estimation:                                           │
│    body_size = dist(nose, tail_base)                            │
│                                                                  │
│  Velocity Calculation:                                           │
│    v(t) = |pos(t) - pos(t-1)| / body_size * fps                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3. Action Classification                         │
│                                                                  │
│  Rule-Based (Primary):                                           │
│    v < 0.5 BL/s  → stationary                                   │
│    v < 3.0 BL/s  → walking                                      │
│    v >= 3.0 BL/s → running                                      │
│                                                                  │
│  Baselines (Comparison):                                         │
│    Random, Majority, SimpleThreshold, CentroidOnly              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. Evaluation & Report                        │
│                                                                  │
│  Metrics:                                                        │
│    - Keypoint: PCK, OKS                                         │
│    - Action: Accuracy, F1, Consistency                          │
│                                                                  │
│  Outputs:                                                        │
│    - HTML Report with embedded GIFs                             │
│    - JSON metrics                                                │
│    - Comparison plots                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 핵심 모듈

### 1. SuperAnimalPredictor (`src/models/predictor.py`)

```python
class SuperAnimalPredictor:
    """DeepLabCut 3.0 SuperAnimal 모델 래퍼"""

    def __init__(
        self,
        model_type: str,      # "topviewmouse" | "quadruped"
        model_name: str,      # "hrnet_w32" (default)
        device: str,          # "auto" | "cuda" | "mps" | "cpu"
        use_keypoints: list,  # 키포인트 필터링 (optional)
    ):
        # DLC SuperAnimal 모델 로드

    def predict_video(self, video_path, max_frames) -> Dict:
        # 비디오 추론
        # Returns: {"keypoints": (F, K, 3), "metadata": {...}}

    def predict_images(self, image_paths) -> Dict:
        # 이미지 추론
```

### 2. UnifiedActionClassifier (`src/models/action_classifier.py`)

```python
class UnifiedActionClassifier:
    """Body-size 정규화 기반 종간 행동 분류기"""

    def __init__(
        self,
        species: str,           # "mouse" | "dog" | "horse"
        fps: float,             # 비디오 FPS
        smoothing_window: int,  # 시간 스무딩 윈도우
    ):
        self.thresholds = VELOCITY_THRESHOLDS[species]

    def analyze(self, keypoints, keypoint_names) -> ActionMetrics:
        # 1. 중심점 궤적 추출
        trajectory = self.extract_center_trajectory(keypoints)

        # 2. Body size 추정 및 속도 정규화
        body_size = self.estimate_body_size(keypoints)
        velocity = self.compute_normalized_velocity(trajectory, body_size)

        # 3. 행동 분류
        actions = self.classify_actions(velocity)

        return ActionMetrics(trajectory, velocity, actions, summary)
```

### 3. Baseline Models (`src/models/baseline.py`)

```python
# 비교용 베이스라인 모델들

class RandomBaseline:
    """랜덤 예측 (최소 성능 기준)"""

class MajorityBaseline:
    """다수결 클래스 예측"""

class SimpleThresholdBaseline:
    """Body-size 정규화 없이 픽셀 속도만 사용"""
    # → Body-size 정규화의 효과 검증용

class CentroidOnlyBaseline:
    """전체 키포인트 대신 중심점만 사용"""
    # → 전체 스켈레톤 정보의 가치 검증용
```

### 4. Evaluation Metrics (`src/evaluation/`)

```python
# metrics.py
def compute_classification_metrics(preds, gt) -> ClassificationMetrics:
    # Accuracy, Precision, Recall, F1, Confusion Matrix

def compute_consistency_metrics(labels) -> ConsistencyMetrics:
    # Segment length, Transition rate, Smoothness score

# model_comparison.py
def compute_pck(preds, gt, thresholds) -> Dict[str, float]:
    # PCK@0.05, PCK@0.1, PCK@0.2

def compute_oks(preds, gt, sigmas) -> float:
    # COCO Object Keypoint Similarity
```

---

## 메트릭 및 평가

### Keypoint Detection Metrics

| 메트릭 | 설명 | 범위 | 좋은 값 |
|--------|------|------|---------|
| **PCK@0.1** | 정답 대비 10% 거리 내 keypoint 비율 | 0-1 | > 0.7 |
| **PCK@0.2** | 정답 대비 20% 거리 내 keypoint 비율 | 0-1 | > 0.85 |
| **OKS** | COCO 표준 Object Keypoint Similarity | 0-1 | > 0.6 |
| **Detection Rate** | 유효 탐지 프레임 비율 | 0-1 | > 0.95 |

### Action Recognition Metrics

| 메트릭 | 설명 | 범위 | 좋은 값 |
|--------|------|------|---------|
| **Accuracy** | 전체 프레임 정확도 | 0-1 | > 0.8 |
| **F1 Score** | 클래스별 Precision-Recall 조화평균 | 0-1 | > 0.75 |
| **Agreement Rate** | 모델 간 일치율 | 0-1 | > 0.7 |
| **Consistency** | 시간적 일관성 (급변 페널티) | 0-1 | > 0.8 |

### Baseline 비교 해석

```
예상 결과:

                     Accuracy    Agreement    Consistency
─────────────────────────────────────────────────────────
SuperAnimal Model      ~85%         100%         ~0.9
YOLO Pose              ~70%         ~70%         ~0.85
SimpleThreshold        ~60%         ~55%         ~0.7
CentroidOnly           ~75%         ~65%         ~0.85
Majority               ~45%         ~45%         1.0
Random                 ~33%         ~33%         ~0.5

해석:
- SuperAnimal > SimpleThreshold: Body-size 정규화 효과 입증
- SuperAnimal > CentroidOnly: 전체 스켈레톤 정보의 가치
- SuperAnimal > Random/Majority: 모델이 실제로 학습함
```

---

## 모델 비교

### 지원 모델

| 모델 | 유형 | 키포인트 | 설치 |
|------|------|----------|------|
| **SuperAnimal-TopViewMouse** | Animal (Top) | 27 | 기본 |
| **SuperAnimal-Quadruped** | Animal (Side) | 39 | 기본 |
| **YOLO Pose** | Human/Animal | 17 | `pip install ultralytics` |
| **MMPose** | Animal | 다양 | `pip install mmpose` |

### 모델별 특성

```
┌─────────────────────────────────────────────────────────────────┐
│                    SuperAnimal (DeepLabCut 3.0)                  │
├─────────────────────────────────────────────────────────────────┤
│ 장점:                                                            │
│   - 동물 특화 (45+ 종 학습)                                       │
│   - 풍부한 키포인트 (27-39개)                                     │
│   - Zero-shot 성능 우수                                          │
│                                                                  │
│ 단점:                                                            │
│   - 추론 속도 (HRNet 백본)                                       │
│   - 설치 복잡성 (DLC 의존)                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        YOLO Pose                                 │
├─────────────────────────────────────────────────────────────────┤
│ 장점:                                                            │
│   - 빠른 추론 속도                                               │
│   - 설치 간편 (ultralytics)                                      │
│   - 실시간 처리 가능                                             │
│                                                                  │
│ 단점:                                                            │
│   - 주로 인간 학습 (동물 성능 제한적)                             │
│   - 적은 키포인트 (17개)                                         │
│   - 동물 특화 학습 필요                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 참고 문헌

1. **SuperAnimal**: Ye et al., "SuperAnimal pretrained pose estimation models for behavioral analysis", Nature Communications, 2024
2. **DeepLabCut**: Mathis et al., "DeepLabCut: markerless pose estimation", Nature Neuroscience, 2018
3. **HRNet**: Sun et al., "Deep High-Resolution Representation Learning for Visual Recognition", CVPR 2019
4. **YOLO Pose**: Ultralytics YOLOv8 Documentation
5. **PCK/OKS**: COCO Keypoint Evaluation Metrics
