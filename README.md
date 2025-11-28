# SuperAnimal Behavior Analysis PoC

DeepLabCut [SuperAnimal](https://www.nature.com/articles/s41467-024-48792-2) 사전훈련 모델을 활용한 동물 행동 분석 프로젝트입니다.

## Pipeline Overview

```
Video → [Keypoint Extraction] → [Action Recognition] → Behavior Labels
         (SuperAnimal/YOLO)     (LSTM/MLP/Rule-Based)
```

| Step | Model | Output |
|------|-------|--------|
| **1. Keypoint Extraction** | SuperAnimal (27/39 kpts), YOLO Pose | 키포인트 좌표 |
| **2. Action Recognition** | LSTM (96%), MLP (93%), Rule-Based | 행동 분류 |

---

## Quick Start

### 1. 환경 설치

```bash
# Clone repository
git clone https://github.com/kafkapple/superanimal-behavior-poc.git
cd superanimal-behavior-poc

# Create conda environment
conda env create -f environment.yml
conda activate superanimal-poc
```

### 2. 실행

```bash
# 빠른 테스트 (~3분)
./run_all.sh --debug

# 표준 실행 (~10분)
./run_all.sh

# 전체 분석 (~30분)
./run_all.sh --full

# 평가만 실행
./run_all.sh --eval-only
```

### 3. 결과 확인

```bash
# 평가 결과
cat outputs/evaluation/evaluation_results.json

# 시각화
open outputs/*/report/dashboard.html
```

---

## Entry Points

| 스크립트 | 설명 | 사용법 |
|----------|------|--------|
| `run_all.sh` | **통합 실행** (전체 파이프라인) | `./run_all.sh --debug` |
| `run.py` | 단일 비디오 분석 | `python run.py` |
| `run_keypoint_comparison.py` | 키포인트 프리셋 비교 | `python run_keypoint_comparison.py` |
| `run_cross_species.py` | Cross-species 비교 | `python run_cross_species.py` |
| `run_evaluation.py` | Action Recognition 평가 | `python run_evaluation.py --mode quick` |

### 실행 모드

| 모드 | 명령어 | 키포인트 | 모델 평가 | 시간 |
|------|--------|----------|-----------|------|
| **Debug** | `--debug` | 50 frames | demo | ~3분 |
| **Standard** | (기본) | 200 frames | quick | ~10분 |
| **Full** | `--full` | 300 frames | full | ~30분 |

---

## Project Structure

```
superanimal-behavior-poc/
├── run_all.sh                    # 통합 실행 스크립트
├── run.py                        # 단일 비디오 분석
├── run_keypoint_comparison.py    # 키포인트 프리셋 비교
├── run_cross_species.py          # Cross-species 비교
├── run_evaluation.py             # Action Recognition 평가
│
├── src/
│   ├── models/
│   │   ├── predictor.py          # SuperAnimal keypoint 추출
│   │   ├── yolo_pose.py          # YOLO Pose 추출
│   │   └── action_models.py      # LSTM, MLP, Transformer, Rule-Based
│   ├── data/
│   │   └── datasets.py           # 데이터셋 로더
│   ├── analysis/
│   │   ├── dashboard.py          # HTML 대시보드 생성
│   │   └── comprehensive_report.py
│   └── evaluation/
│       └── comprehensive_evaluation.py
│
├── configs/                      # Hydra 설정 파일
├── docs/                         # 상세 문서
│   ├── MODEL_SPECS.md            # 모델 스펙
│   └── DATASETS.md               # 데이터셋 & 키포인트 프리셋
│
├── data/                         # 데이터 (git ignored)
└── outputs/                      # 출력 (git ignored)
```

---

## Configuration

### 모델 선택

```bash
# TopViewMouse (기본) - 마우스 top-view 분석
python run.py model=topviewmouse

# Quadruped - 4족 동물 (개, 고양이, 말 등)
python run.py model=quadruped

# 커스텀 비디오 사용
python run.py input=/path/to/video.mp4 model=quadruped
```

### 주요 파라미터

```bash
# 프레임 수 제한
python run.py data.video.max_frames=100

# Device 선택
python run.py device=cuda    # NVIDIA GPU
python run.py device=mps     # Apple Silicon
python run.py device=cpu     # CPU

# Confidence threshold 조정
python run.py model.confidence_threshold=0.5

# 키포인트 프리셋 선택
python run.py model.use_keypoints='${model.keypoint_presets.minimal}'
```

### Config 파일 직접 수정

```yaml
# configs/config.yaml
defaults:
  - model: quadruped    # topviewmouse 또는 quadruped
  - data: sample

device: auto            # auto, cuda, mps, cpu
seed: 42

# configs/model/topviewmouse.yaml
video_adapt: false      # true: 더 정확하지만 느림
confidence_threshold: 0.3
use_keypoints: ${model.keypoint_presets.standard}  # full, standard, mars, locomotion, minimal
```

---

## Models

### Keypoint Extraction (Pose Estimation)

| Model | Keypoints | Species | Config | Command |
|-------|-----------|---------|--------|---------|
| **TopViewMouse** | 27 | Mouse (top-view) | `model=topviewmouse` | `python run.py` |
| **Quadruped** | 39 | Dog, Cat, Horse | `model=quadruped` | `python run.py model=quadruped` |
| **YOLO Pose** | 17 | General | - | `--models yolo_pose` |

### TopViewMouse 키포인트 (27개)

| Region | Keypoints |
|--------|-----------|
| Head (8) | nose, left_ear, right_ear, left_ear_tip, right_ear_tip, left_eye, right_eye, head_midpoint |
| Body (6) | neck, mid_back, mouse_center, mid_backend, mid_backend2, mid_backend3 |
| Tail (7) | tail_base, tail1, tail2, tail3, tail4, tail5, tail_end |
| Limbs (6) | left_shoulder, right_shoulder, left_hip, right_hip, left_midside, right_midside |

### Quadruped 키포인트 (39개)

| Region | Keypoints |
|--------|-----------|
| Head | nose, upper_jaw, lower_jaw, left_eye, right_eye, left_ear, right_ear |
| Body | neck_base, neck_end, back_base, back_middle, back_end, belly_bottom |
| Tail | tail_base, tail_middle, tail_end |
| Front Legs | left/right_front_paw, elbow, knee, hoof |
| Back Legs | left/right_back_paw, elbow, knee, hoof |

### Action Recognition (Behavior Classification)

| Model | Accuracy | F1 | Use Case |
|-------|----------|-----|----------|
| **LSTM** | 96.1% | 95.6% | 시계열 패턴 (Best) |
| **MLP** | 92.7% | 92.0% | 프레임 단위 분류 |
| **Transformer** | 85.2% | 86.4% | 장거리 의존성 |
| **Rule-Based** | 32.0% | 12.1% | Baseline only |

---

## Datasets

### Label Systems

| Dataset | Type | Classes |
|---------|------|---------|
| **locomotion_sample** | Locomotion | stationary, walking, running, other |
| **mars_sample** | Social | attack, mount, investigation, other |

### Keypoint Presets

| Preset | Keypoints | Use Case |
|--------|-----------|----------|
| **full** | 27 | 정밀 분석 |
| **standard** | 11 | Open Field Test |
| **minimal** | 3 | 빠른 추적 |

자세한 내용은 [docs/DATASETS.md](docs/DATASETS.md) 참조.

---

## Output Structure

```
outputs/
├── evaluation/
│   ├── evaluation_results.json   # 평가 결과
│   └── models/                   # 학습된 모델
│       ├── full_lstm.pt
│       └── full_mlp.pt
│
└── <experiment>/
    ├── keypoints/                # 추출된 키포인트
    ├── visualizations/           # 시각화
    └── report/
        └── dashboard.html        # HTML 대시보드
```

---

## Documentation

| 문서 | 내용 |
|------|------|
| [docs/MODEL_SPECS.md](docs/MODEL_SPECS.md) | 모델 아키텍처 & 파라미터 |
| [docs/DATASETS.md](docs/DATASETS.md) | 데이터셋 & 키포인트 프리셋 |
| [docs/architecture.md](docs/architecture.md) | 시스템 아키텍처 |
| [docs/configuration.md](docs/configuration.md) | 설정 가이드 |

---

## References

- [SuperAnimal - Nature Communications 2024](https://www.nature.com/articles/s41467-024-48792-2)
- [DeepLabCut Documentation](https://deeplabcut.github.io/DeepLabCut/)
- [MARS Dataset - eLife 2021](https://elifesciences.org/articles/63720)

---

## License

MIT License
