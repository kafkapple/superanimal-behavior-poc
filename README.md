# SuperAnimal Behavior Analysis PoC

DeepLabCut의 [SuperAnimal](https://www.nature.com/articles/s41467-024-48792-2) 사전훈련 모델을 활용한 동물 행동 분석 PoC (Proof of Concept) 프로젝트입니다.

## Quick Start - 전체 파이프라인 실행

### 통합 실행 스크립트

`./run_all.sh`와 `python run_comprehensive.py`는 동일한 옵션을 지원합니다.

```bash
# 1. 환경 설치
conda env create -f environment.yml
conda activate superanimal-poc

# 2. 빠른 테스트 (~2분)
./run_all.sh --debug
# 또는
./run_all.sh -d

# 3. ⭐ 모든 조합 빠른 테스트 (~5분) - 추천!
./run_all.sh --debug-full
# 또는
./run_all.sh -df

# 4. 표준 분석 (~10분, 기본값)
./run_all.sh

# 5. 완전한 분석 (~30분)
./run_all.sh --all
# 또는
./run_all.sh -a
```

### 커스텀 설정

```bash
# 특정 종만 분석
./run_all.sh --species mouse,dog

# 특정 프리셋만 사용
./run_all.sh --presets full,standard,minimal

# 프레임 수 제한
./run_all.sh --all --max-frames 100

# 조합 사용
./run_all.sh --species mouse,dog,horse --presets full,minimal --max-frames 50

# 기존 결과에서 시각화만 재생성
./run_all.sh --visualize-only --input outputs/comprehensive/20241127_123456

# 상세 로그 출력
./run_all.sh --debug-full --verbose
```

### 실행 모드 비교표

| 모드 | 명령어 | 프레임 | 종 | 프리셋 | 모델 | 시간 |
|------|--------|--------|-----|--------|------|------|
| **Debug** | `-d`, `--debug` | 50 | 1 (mouse) | 2개 | SuperAnimal | ~2분 |
| **Debug-Full** ⭐ | `-df`, `--debug-full` | 20 | 3종 모두 | 5개 모두 | 모든 모델 | ~5분 |
| **Standard** | (기본) | 200 | 2 (mouse, dog) | 3개 | SuperAnimal | ~10분 |
| **Full** | `-a`, `--all` | 300 | 3종 모두 | 5개 모두 | 모든 모델 | ~30분 |

### 실험 결과 확인

```bash
# 결과 디렉토리 구조
outputs/full_pipeline/<timestamp>/
├── experiment_results/          # 실험별 결과
│   ├── single_video_mouse/      # GIF, HTML 보고서
│   ├── keypoint_comparison/     # 프리셋 비교 GIF
│   └── cross_species/           # 종간 비교
├── visualizations/              # 📊 시각화 모음
│   ├── keypoint_comparison/     # 키포인트 프리셋 비교 차트
│   │   ├── performance_metrics.json   # 🆕 Accuracy/F1 메트릭 (JSON)
│   │   └── *_performance.png          # 🆕 성능 비교 차트
│   ├── species_comparison/      # 종간 비교 차트
│   └── action_performance/      # 모델 성능 비교 차트
├── comprehensive/               # (-c 플래그 사용 시)
│   ├── all_species_comparison/  # 3종 비교 결과
│   └── preset_*/                # 각 프리셋별 결과
├── evaluation/
│   └── baseline_comparison.json # 정량 메트릭 (Accuracy, F1, etc.)
├── report/
│   └── dashboard.html           # 📈 종합 대시보드 (인터랙티브)
├── summary.json                 # 전체 요약
└── final_dashboard.html         # 🎯 최종 대시보드 (브라우저 자동 오픈)
```

### 대시보드 직접 열기

```bash
# macOS
open outputs/full_pipeline/<timestamp>/final_dashboard.html

# Linux
xdg-open outputs/full_pipeline/<timestamp>/final_dashboard.html

# Windows
start outputs/full_pipeline/<timestamp>/final_dashboard.html
```

---

## 주요 기능

- **다중 입력 지원**: 비디오 파일 또는 이미지 시퀀스
- **Cross-Species 비교**: Mouse, Dog, Horse 등 여러 종 동시 분석
- **Body Size 추정**: 종별 체크 크기 정량화 및 비교
- **HTML 보고서**: 대화형 보고서 + Action GIF 애니메이션
- **GPU 가속**: CUDA (Linux/Windows) 및 MPS (Apple Silicon) 지원
- **Hierarchical Action Comparison**: 키포인트 프리셋별 action label 성능 계층 비교

## 지원 플랫폼

| 플랫폼 | GPU 지원 | 테스트 상태 |
|--------|----------|-------------|
| **macOS (Apple Silicon)** | MPS | Supported |
| **macOS (Intel)** | CPU only | Supported |
| **Linux** | CUDA | Supported |
| **Windows** | CUDA | Supported |

GPU 자동 감지:
```bash
python run.py device=auto  # 자동 감지 (CUDA > MPS > CPU)
python run.py device=mps   # Apple Silicon GPU 강제
python run.py device=cuda  # NVIDIA GPU 강제
python run.py device=cpu   # CPU 강제
```

## 3-Stage Pipeline

| Stage | 기능 | 실행 명령 |
|-------|------|----------|
| **Stage 1** | 동물 키포인트 추출 + 행동 분석 + HTML 보고서 | `python run.py` |
| **Stage 2** | 키포인트 프리셋 비교 (Full/Standard/Minimal) | `python run_keypoint_comparison.py` |
| **Stage 3** | Cross-Species 행동 + Body Size 비교 | `python run_cross_species.py` |
| **Batch** | 전체 실험 + 베이스라인 비교 | `./run_all.sh` |

---

## 빠른 시작

### 1. 환경 설치

```bash
cd superanimal-behavior-poc
conda env create -f environment.yml
conda activate superanimal-poc

# (선택) 인간 포즈 추정용
pip install mediapipe
```

### 2. Stage별 실행

```bash
# Stage 1: 기본 동물 행동 분석 + HTML 보고서
python run.py data.video.max_frames=100

# Stage 2: 키포인트 프리셋 비교 보고서
python run_keypoint_comparison.py data.video.max_frames=100

# Stage 3: Cross-Species 비교 (mouse + dog)
python run_cross_species.py data.video.max_frames=100

# Stage 3: 3종 비교 (zsh에서는 따옴표 필수)
python run_cross_species.py "species=[mouse,dog,horse]" data.video.max_frames=50
```

### 3. 커스텀 입력

```bash
# 비디오 파일
python run.py input=/path/to/my/video.mp4

# 이미지 디렉토리 (파일명 숫자 기준 정렬)
python run.py input=/path/to/frames/
# 예: frame_001.jpg, frame_002.jpg, ... 순서로 처리
```

---

## Stage 1: Animal Keypoint & Behavior Analysis

동물 비디오에서 키포인트를 추출하고 행동을 분류합니다.

### 실행

```bash
# 기본 실행 (샘플 비디오 자동 다운로드)
python run.py

# 프레임 제한 (빠른 테스트)
python run.py data.video.max_frames=100

# Quadruped 모델 (개, 고양이 등)
python run.py model=quadruped

# Video adaptation (더 정확, 더 느림)
python run.py model.video_adapt=true

# HTML 보고서 + Action GIF 생성
python run.py report.html=true report.gifs=true

# GPU 장치 지정
python run.py device=mps    # Apple Silicon
python run.py device=cuda   # NVIDIA GPU
python run.py device=cpu    # CPU only
```

### 출력

```
outputs/experiments/{timestamp}_{model}/
├── report_{video}.html           # HTML 보고서 (대화형)
├── gifs/                         # Action별 GIF 애니메이션
│   ├── video_walking_1.gif
│   ├── video_running_1.gif
│   └── video_resting_1.gif
├── predictions/
│   ├── *.h5                      # 키포인트 (DLC 형식)
│   ├── keypoints_coordinates.csv # 키포인트 (CSV)
│   └── *_labeled.mp4             # 레이블 비디오
├── keypoint_frames/              # 키포인트 오버레이 이미지
├── videos/
│   └── *_keypoints.mp4           # 오버레이 비디오
├── plots/
│   ├── trajectory.png            # 이동 궤적
│   ├── velocity_profile.png      # 속도 프로파일
│   ├── behavior_timeline.png     # 행동 타임라인
│   └── analysis_report.png       # 종합 보고서
├── behavior_metrics.csv          # 프레임별 행동 데이터
└── .hydra/                       # Hydra 설정 로그
```

### 지원 모델

| 모델 | 설명 | 키포인트 | 대상 |
|------|------|----------|------|
| SuperAnimal-TopViewMouse | 상단 뷰 | 27개 | 생쥐, 쥐 |
| SuperAnimal-Quadruped | 측면 뷰 | 39개 | 개, 고양이, 말 등 45+ 종 |

---

## Stage 2: Keypoint Preset Comparison

동일 비디오에 다양한 키포인트 프리셋을 적용하여 비교합니다.

### 실행

```bash
# 기본 실행
python run_keypoint_comparison.py

# 프레임 제한
python run_keypoint_comparison.py data.video.max_frames=100
```

### 키포인트 프리셋

| 프리셋 | 개수 | 용도 |
|--------|------|------|
| **Full** | 27 | 정밀 자세 분석, 그루밍 감지 |
| **Standard** | 11 | Open Field Test, 일반 행동 |
| **MARS** | 7 | 사회적 상호작용, 다중 동물 |
| **Locomotion** | 5 | 이동/보행 분석 |
| **Minimal** | 3 | 기본 추적, 실시간 처리 |

### Action Recognition 성능 비교 (Accuracy/F1)

키포인트 수에 따른 action recognition 성능 차이를 정량적으로 비교합니다.
**Full 프리셋(27개)을 기준(ground truth)으로** 각 프리셋의 accuracy, F1 score를 계산합니다.

```
=== Performance by Keypoint Count ===
Preset      | Keypoints | Accuracy | Agreement | Mean F1
------------|-----------|----------|-----------|--------
Full        | 27        | 100.0%   | 100.0%    | 1.000
Standard    | 11        | 95.2%    | 95.2%     | 0.948
MARS        | 7         | 91.8%    | 91.8%     | 0.912
Locomotion  | 5         | 88.5%    | 88.5%     | 0.879
Minimal     | 3         | 82.3%    | 82.3%     | 0.815
```

대시보드에서 확인 가능:
- **Accuracy by Preset**: 각 프리셋별 정확도 막대 그래프
- **F1 by Action Class**: stationary, walking, running 별 F1 점수
- **Keypoint Count vs Accuracy**: 키포인트 수와 정확도 관계 (트렌드 라인 포함)
- **Accuracy Drop from Full**: Full 대비 정확도 하락률

### Hierarchical Action Label Comparison (NEW)

Action label 별로 키포인트 프리셋 성능을 계층적으로 비교합니다:

**시각화 출력:**
1. **hierarchical_action_comparison.png**: 3단 계층적 비교 차트
   - Row 1: Overall accuracy, Mean F1, Keypoint-Accuracy trade-off
   - Row 2: Per-action breakdown (stationary, walking, running의 F1/Precision/Recall)
   - Row 3: F1 Heatmap (action x preset) + Summary table

2. **confusion_matrix_grid.png**: 각 프리셋별 Confusion Matrix
   - Full preset을 기준으로 다른 프리셋의 action 분류 오차 시각화
   - 정규화된 비율 + 실제 count 표시

```
outputs/visualizations/keypoint_comparison/
├── performance_by_keypoint_*.png     # 기본 성능 비교
├── hierarchical_action_comparison_*.png  # 계층적 action 비교
├── confusion_matrix_grid_*.png       # Confusion matrix 그리드
└── performance_metrics.json          # JSON 메트릭 데이터
```

---

## Stage 3: Cross-Species Action Recognition

여러 종의 공통 행동(walking, running, stationary)과 Body Size를 비교 분석합니다.

### 실행

```bash
# 기본 실행 (mouse + dog)
python run_cross_species.py data.video.max_frames=100

# 3종 비교 (zsh에서는 따옴표 필수!)
python run_cross_species.py "species=[mouse,dog,horse]" data.video.max_frames=50

# 2종만 비교
python run_cross_species.py "species=[mouse,horse]"
```

### 지원 종

| 종 | Sample | Model Type | 설명 |
|----|--------|------------|------|
| mouse | mouse_topview | topviewmouse | 생쥐 (상단 뷰) |
| dog | dog_walking | quadruped | 개 (측면 뷰) |
| horse | horse_running | quadruped | 말 (측면 뷰) |

### 출력

```
outputs/experiments/cross_species_{timestamp}/
├── cross_species_report.html     # HTML 비교 보고서
├── cross_species_comparison.png  # 시각화 비교 (4-panel)
│   ├── Action Distribution       # 행동 분포 비교
│   ├── Velocity Profile          # 정규화 속도 비교
│   ├── Body Size Bar Chart       # Body Size 막대 그래프
│   └── Body Size Box Plot        # Body Size 분포
├── action_comparison.csv         # 행동 분포 CSV
├── mouse_predictions/
├── dog_predictions/
└── horse_predictions/
```

### Body Size 추정

각 종의 body size를 키포인트 간 거리(머리-꼬리)로 추정합니다:

```
=== Body Size Comparison ===
Mouse: 114.0 ± 12.5 px (range: 95.2-132.8)
Dog:   245.3 ± 18.7 px (range: 210.5-278.9)
Horse: 320.1 ± 25.2 px (range: 285.0-365.4)
```

속도는 body-length/sec로 정규화되어 종 간 공정한 비교가 가능합니다.

---

## Action GIF 시각화

각 행동 유형별로 RGB + Keypoint overlay 애니메이션 GIF를 생성하여 detection 정확도를 시각적으로 확인할 수 있습니다.

### 활성화

```bash
# HTML 보고서 + Action GIF 생성
python run.py report.html=true report.gifs=true

# GIF 설정 조정
python run.py report.gifs=true report.gifs_per_action=3 report.gif_fps=10
```

### GIF 출력

```
outputs/experiments/{timestamp}/gifs/
├── video_resting_1.gif     # 정지 상태 샘플 1
├── video_resting_2.gif     # 정지 상태 샘플 2
├── video_walking_1.gif     # 걷기 샘플 1
├── video_walking_2.gif     # 걷기 샘플 2
├── video_running_1.gif     # 달리기 샘플 1
└── video_running_2.gif     # 달리기 샘플 2
```

각 GIF에는:
- RGB 영상 프레임
- Keypoint 오버레이 (색상 구분)
- Skeleton 연결선
- Action 라벨 표시
- 프레임 번호

### HTML 보고서 통합

생성된 GIF는 HTML 보고서에 자동으로 삽입되어 각 action별로 갤러리 형태로 확인 가능합니다.

---

## 설정

### config.yaml 주요 설정

```yaml
# 장치 설정
device: auto  # auto, cuda, mps, cpu

# 입력 (선택사항)
input: null  # 비디오 파일 또는 이미지 디렉토리 경로

# Cross-species 종 목록
species:
  - mouse
  - dog

# 보고서 설정
report:
  html: true          # HTML 보고서 생성
  gifs: true          # Action GIF 생성
  gifs_per_action: 2  # 액션당 GIF 수
  gif_fps: 8          # GIF 프레임레이트

# 비디오 처리
data:
  video:
    max_frames: 500   # 최대 프레임 (null = 전체)
```

### 행동 분류 임계값

```yaml
# configs/behavior/default.yaml
analysis:
  classification:
    behaviors:
      - name: resting
        velocity_threshold: 0.3
      - name: walking
        velocity_threshold_min: 0.3
        velocity_threshold_max: 3.0
```

---

## 프로젝트 구조

```
superanimal-behavior-poc/
├── run.py                  # Stage 1: 기본 행동 분석
├── run_comparison.py       # Stage 2: 키포인트 비교
├── run_cross_species.py    # Stage 3: Cross-species 비교
├── configs/
│   ├── config.yaml         # 메인 설정
│   ├── model/              # topviewmouse.yaml, quadruped.yaml
│   ├── data/
│   └── behavior/
├── src/
│   ├── models/
│   │   ├── predictor.py        # SuperAnimal 래퍼 (비디오/이미지)
│   │   └── action_classifier.py # Cross-species 분류기
│   ├── analysis/
│   │   ├── behavior.py         # 행동 분석 + Body Size 추정
│   │   ├── visualizer.py       # 시각화
│   │   └── report_generator.py # HTML 보고서 + GIF 생성
│   ├── data/
│   │   ├── downloader.py       # 샘플 다운로드
│   │   └── human_pose.py       # MediaPipe 래퍼
│   ├── scripts/
│   │   ├── setup.sh            # 환경 설정
│   │   ├── download_samples.py # 샘플 비디오 다운로드
│   │   └── download_datasets.py # 벤치마크 데이터셋 다운로드
│   └── utils/
├── docs/
│   ├── DATASETS.md             # 데이터셋 가이드
│   ├── keypoint_paradigms.md
│   └── custom_dataset_guide.md
├── data/                   # 입력 데이터 (git ignored)
│   ├── raw/
│   ├── processed/
│   └── external/
├── outputs/                # 출력 결과 (git ignored)
│   ├── experiments/        # 실험 결과
│   └── logs/               # Hydra 로그
└── environment.yml         # Conda 환경
```

---

## Python API

### 비디오/이미지 추론

```python
from src.models.predictor import SuperAnimalPredictor

predictor = SuperAnimalPredictor(
    model_type="topviewmouse",
    device="auto",  # auto, cuda, mps, cpu
)

# 비디오 추론
results = predictor.predict_video(
    video_path="video.mp4",
    output_dir="output",
    max_frames=100,
)

# 이미지 추론
results = predictor.predict_images(
    image_paths=["frame1.jpg", "frame2.jpg"],
    output_dir="output",
)

keypoints = results["keypoints"]  # Shape: (frames, keypoints, 3)
```

### Body Size 추정

```python
from src.analysis.behavior import estimate_body_size

body_stats = estimate_body_size(
    keypoints=results["keypoints"],
    keypoint_names=predictor.get_keypoint_names(),
    model_type="topviewmouse",
)

print(f"Body size: {body_stats['mean']:.1f} ± {body_stats['std']:.1f} px")
```

### HTML 보고서 생성

```python
from src.analysis.report_generator import HTMLReportGenerator, ActionGifGenerator

# GIF 생성
gif_gen = ActionGifGenerator(output_dir="gifs")
action_gifs = gif_gen.generate_all_action_gifs(
    video_path="video.mp4",
    keypoints=keypoints,
    keypoint_names=names,
    action_labels=metrics.behavior_labels,
    action_names={0: "resting", 1: "walking", 2: "running"},
)

# HTML 보고서
html_gen = HTMLReportGenerator(output_dir="reports")
html_gen.generate_behavior_report(
    video_name="my_video",
    species="mouse",
    metrics={"behavior_summary": metrics.behavior_summary},
    action_gifs=action_gifs,
    plot_paths={"trajectory": "plots/trajectory.png"},
    body_size_stats=body_stats,
)
```

### Cross-Species 비교

```python
from src.models.action_classifier import UnifiedActionClassifier, CrossSpeciesComparator

comparator = CrossSpeciesComparator(fps=30.0)

# 각 종 분석
for species in ["mouse", "dog"]:
    classifier = UnifiedActionClassifier(species=species, fps=30.0)
    metrics = classifier.analyze(keypoints, keypoint_names)
    comparator.add_result(species.capitalize(), metrics)

# 비교 결과 저장
comparator.save_comparison_csv("comparison.csv")
```

### Keypoint Preset 성능 비교 (Accuracy/F1)

```python
from src.analysis.keypoint_visualizer import (
    KeypointVisualizer,
    compare_presets_with_metrics,
)

# Full preset을 reference로 각 프리셋별 accuracy/F1 계산
presets = ["full", "standard", "mars", "locomotion", "minimal"]
results, metrics = compare_presets_with_metrics(
    keypoints=keypoints,           # (frames, keypoints, 3)
    all_keypoint_names=names,      # 전체 키포인트 이름 리스트
    presets=presets,
    fps=30.0,
    reference_preset="full",       # Ground truth로 사용할 프리셋
)

# 결과 확인
for r in results:
    print(f"{r.preset_name}: accuracy={r.accuracy:.3f}, agreement={r.agreement_with_full:.3f}")
    print(f"  F1 scores: {r.f1_scores}")

# 시각화 생성
viz = KeypointVisualizer(output_dir="outputs/keypoint_comparison")
viz.create_performance_by_keypoint_count(results, video_name="my_video")
```

출력 예시:
```
full: accuracy=1.000, agreement=1.000
  F1 scores: {'stationary': 1.0, 'walking': 1.0, 'running': 1.0}
standard: accuracy=0.952, agreement=0.952
  F1 scores: {'stationary': 0.96, 'walking': 0.94, 'running': 0.93}
minimal: accuracy=0.823, agreement=0.823
  F1 scores: {'stationary': 0.85, 'walking': 0.80, 'running': 0.78}
```

---

## 데이터셋 다운로드

### 샘플 비디오 다운로드

```bash
# 모든 샘플 비디오 다운로드 (mouse, dog, horse)
python -m src.scripts.download_samples

# 또는 개별 다운로드
python -m src.scripts.download_datasets --samples
```

### 벤치마크 데이터셋

```bash
# 사용 가능한 데이터셋 목록
python -m src.scripts.download_datasets --list

# 특정 데이터셋 다운로드 안내
python -m src.scripts.download_datasets --dataset mars
python -m src.scripts.download_datasets --dataset ap10k
python -m src.scripts.download_datasets --dataset coco_pose
```

### 지원 데이터셋

| 데이터셋 | 종 | 키포인트 | 용도 |
|----------|-----|----------|------|
| **Sample Videos** | mouse, dog, horse | 27/39 | 기본 파이프라인 테스트 |
| **MARS** | mouse (×2) | 7 | 사회적 상호작용, 행동 분류 |
| **AP-10K** | 60+ 종 | 17 | 다종 동물 포즈 |
| **COCO** | human | 17 | 인간 포즈 추정 |
| **UCLA Mouse** | mouse | - | 행동 인식 데모 |

---

## 문서

- [이론적 배경 및 파이프라인 구조](docs/theory_and_pipeline.md) - **핵심 이론, 모듈 설명**
- [아키텍처 가이드](docs/architecture.md) - 전체 시스템 구조
- [설정 가이드](docs/configuration.md) - 모든 설정 옵션
- [데이터셋 가이드](docs/DATASETS.md) - 인기 pose estimation 데이터셋
- [키포인트 패러다임 가이드](docs/keypoint_paradigms.md)
- [커스텀 데이터셋 & Fine-tuning 가이드](docs/custom_dataset_guide.md)

---

## Batch Experiment Pipeline

### 전체 실험 자동 실행

```bash
# 권장: Debug → Full → Model Comparison → 시각화 → 대시보드
./run_all.sh

# Debug만 (빠른 검증, ~2분)
./run_all.sh --debug-only
./run_all.sh -d

# Full만 (Debug 스킵, ~8분)
./run_all.sh --full-only
./run_all.sh -f

# Comprehensive 모드 (모든 조합 분석, ~30분)
./run_all.sh --comprehensive
./run_all.sh -c

# Full + Comprehensive 조합 (권장: 완전한 보고서용)
./run_all.sh --full-only -c
```

### 실행 모드 비교

| 모드 | 명령어 | 내용 | 예상 시간 |
|------|--------|------|-----------|
| **기본** | `./run_all.sh` | Debug → Full → Model Comparison → 대시보드 | ~10분 |
| **Debug Only** | `./run_all.sh -d` | 빠른 검증 (50 frames), 시각화 스킵 | ~2분 |
| **Full Only** | `./run_all.sh -f` | Debug 스킵, Full + 시각화 + 대시보드 | ~8분 |
| **Comprehensive** | `./run_all.sh -c` | 모든 종/프리셋/모델 + 종합 대시보드 | ~30분 |
| **Full + Comprehensive** | `./run_all.sh -f -c` | 가장 완전한 분석 및 보고서 | ~25분 |

### 파이프라인 실행 단계

```
./run_all.sh -c 실행 시:

Step 1: Debug Mode (선택적)
   └── 빠른 검증 테스트

Step 2: Full Experiments
   ├── Single Video Analysis (300 frames)
   ├── Keypoint Comparison (Full/Standard/Minimal)
   └── Cross-Species (Mouse vs Dog)

Step 3: Comprehensive Analysis (-c 플래그 필요)
   ├── All Species Comparison (Mouse, Dog, Horse)
   └── All Keypoint Presets (5개 프리셋 × 분석)

Step 4: Model Comparison
   ├── SuperAnimal vs YOLO Pose
   └── Baseline 모델 비교

Step 5: Generate Visualizations ⭐ NEW
   ├── Keypoint Preset Comparison Charts
   ├── Species Comparison Charts
   └── Action Performance Charts

Step 6: Generate Dashboard ⭐ NEW
   └── Comprehensive HTML Dashboard (auto-open)
```

### Comprehensive 모드 상세

Comprehensive 모드(`-c`)는 가능한 모든 조합을 분석합니다:

| 분석 항목 | 내용 |
|-----------|------|
| **Species** | Mouse, Dog, Horse (3종) |
| **Keypoint Presets** | Full (27), Standard (11), MARS (7), Locomotion (5), Minimal (3) |
| **Models** | SuperAnimal, YOLO Pose, Baselines (Random, Majority, Threshold) |
| **총 조합** | 15개 (3종 × 5프리셋) + 모델 비교 |

**생성되는 시각화:**

1. **키포인트 프리셋 비교**
   - 키포인트 수별 Action Distribution 변화
   - 키포인트 포함 매트릭스 (어떤 키포인트가 어떤 프리셋에 포함되는지)
   - 프리셋별 GIF 애니메이션 비교

2. **종간 비교**
   - Body Size 비교 (bar chart, box plot, relative size)
   - Action Distribution by Species
   - Velocity Profile (body-length 정규화)
   - Comprehensive 4-panel 비교 차트

3. **Action Recognition 성능**
   - 모델별 Accuracy 비교
   - F1 Score by Action Class (stationary/walking/running)
   - Consistency Score (시간적 일관성)

### 포함된 실험

| 실험 | 설명 | Debug | Full |
|------|------|-------|------|
| `single_video_mouse` | SuperAnimal-TopViewMouse 분석 | 50 frames | 300 frames |
| `keypoint_comparison` | Full/Standard/Minimal 비교 GIF | 30 frames | 200 frames |
| `cross_species` | Mouse vs Dog 행동 비교 | 50 frames | 200 frames |

### 모델 비교 (Model Comparison)

다른 유명 모델들과 비교합니다:

```bash
# 개별 실행
python run_model_comparison.py
python run_model_comparison.py --models superanimal,yolo_pose
```

| 모델 | 설명 | 설치 |
|------|------|------|
| **SuperAnimal** | DeepLabCut 3.0 사전훈련 모델 (기본) | 기본 포함 |
| **YOLO Pose** | Ultralytics YOLOv8-pose | `pip install ultralytics` |
| **MMPose** | OpenMMLab 동물 포즈 추정 (선택) | `pip install mmpose` |

### 베이스라인 비교

| Baseline | 설명 |
|----------|------|
| **Random** | 랜덤 행동 예측 (40% stationary, 40% walking, 20% running) |
| **Majority** | 가장 빈번한 클래스로 모든 프레임 예측 |
| **SimpleThreshold** | Body-size 정규화 없이 픽셀 속도만 사용 |
| **CentroidOnly** | 전체 키포인트 대신 중심점만 사용 |

### 정량 메트릭

**키포인트 메트릭:**
- **PCK@0.1, PCK@0.2**: Percentage of Correct Keypoints (threshold 기준)
- **OKS**: Object Keypoint Similarity (COCO 표준 메트릭)

**행동 인식 메트릭:**
- **Accuracy**: 전체 정확도
- **F1 Score**: 클래스별 F1
- **Agreement Rate**: 모델 간 일치율
- **Consistency Score**: 시간적 일관성 (급격한 변화 페널티)

```
======================================================================
MODEL COMPARISON RESULTS
======================================================================

Reference Model: superanimal

📊 Action Recognition Comparison:
--------------------------------------------------
Model                     Agreement   Accuracy   Consistency
--------------------------------------------------
yolo_pose                     72.3%      68.5%         0.85
random                        33.2%      33.2%         0.45
majority                      45.0%      45.0%         1.00
--------------------------------------------------

📍 Keypoint Detection Comparison:
--------------------------------------------------
Model                          OKS    PCK@0.1    PCK@0.2
--------------------------------------------------
yolo_pose                    0.652      0.485      0.723
--------------------------------------------------
```

### 모듈 구조

```
run_all.sh                          # 전체 파이프라인 실행 (시각화 + 대시보드 포함)
├── run.py                          # Stage 1: 단일 비디오 분석
├── run_keypoint_comparison.py      # Stage 2: 키포인트 프리셋 비교
├── run_cross_species.py            # Stage 3: 종간 비교
├── run_model_comparison.py         # Step 4: 모델 비교 (YOLO, Baselines)
├── generate_report.py              # ⭐ Step 5-6: 종합 대시보드 생성
│
├── src/models/
│   ├── predictor.py           # SuperAnimal 키포인트 추출
│   ├── action_classifier.py   # 행동 분류 (body-length 정규화)
│   ├── baseline.py            # 베이스라인 모델들 (Random, Majority, etc.)
│   └── yolo_pose.py           # YOLO Pose 래퍼
├── src/evaluation/
│   ├── metrics.py             # 정량 평가 (Accuracy, F1, Confusion Matrix)
│   └── model_comparison.py    # PCK, OKS 메트릭
├── src/analysis/
│   ├── behavior.py            # Body size 추정
│   ├── report_generator.py    # HTML 보고서 + GIF 생성
│   ├── visualizer.py          # 시각화
│   ├── dashboard.py           # ⭐ 종합 HTML 대시보드 생성
│   ├── keypoint_visualizer.py # ⭐ 키포인트 프리셋 비교 시각화
│   └── species_visualizer.py  # ⭐ 종간 비교 시각화
└── configs/
    ├── config.yaml            # 메인 설정
    ├── model/                 # 모델별 설정 + 키포인트 프리셋
    └── species/               # 종별 velocity threshold
```

### 시각화 모듈 사용법

```python
# 1. 키포인트 프리셋 비교 시각화
from src.analysis.keypoint_visualizer import KeypointVisualizer, compare_presets

visualizer = KeypointVisualizer(output_dir="outputs/visualizations")

# 프리셋 비교 정적 이미지
visualizer.create_preset_comparison_figure(
    video_path="video.mp4",
    keypoints=keypoints,
    all_keypoint_names=names,
    presets=["full", "standard", "minimal"],
)

# 프리셋 비교 GIF 애니메이션
visualizer.create_comparison_gif(
    video_path="video.mp4",
    keypoints=keypoints,
    all_keypoint_names=names,
    presets=["full", "standard", "minimal"],
    max_frames=100,
    fps=8.0,
)

# 2. 종간 비교 시각화
from src.analysis.species_visualizer import SpeciesVisualizer, create_species_result

visualizer = SpeciesVisualizer(output_dir="outputs/visualizations")

# 종별 결과 생성
mouse_result = create_species_result("mouse", "topviewmouse", mouse_keypoints, mouse_names)
dog_result = create_species_result("dog", "quadruped", dog_keypoints, dog_names)

# 종합 비교 차트
visualizer.create_comprehensive_comparison([mouse_result, dog_result])

# 3. 종합 대시보드 생성
from src.analysis.dashboard import DashboardGenerator, ExperimentSummary

dashboard = DashboardGenerator(output_dir="outputs/report")
summary = ExperimentSummary(
    experiment_name="My Experiment",
    timestamp="2024-11-27",
    total_frames=1000,
    species=["mouse", "dog"],
    presets_tested=["full", "standard", "minimal"],
)
dashboard.generate_full_dashboard(summary, gif_paths, plot_paths)
```

### Python 원클릭 스크립트 (`run_comprehensive.py`)

Shell 스크립트 대신 Python 스크립트로도 전체 파이프라인을 실행할 수 있습니다:

```bash
# Quick test (debug mode, ~2 min)
python run_comprehensive.py --debug
python run_comprehensive.py -d

# Standard analysis (~10 min)
python run_comprehensive.py

# Full analysis with all species/presets (~25 min)
python run_comprehensive.py --all
python run_comprehensive.py -a

# Custom configuration
python run_comprehensive.py --species mouse,dog --presets full,standard,minimal
python run_comprehensive.py -s mouse,dog,horse -p full,standard -m 200

# Only generate visualizations from existing results
python run_comprehensive.py --visualize-only --input outputs/comprehensive/20241127_123456
```

**옵션:**

| 옵션 | 단축 | 설명 |
|------|------|------|
| `--debug` | `-d` | Debug 모드 (50 frames, mouse만, ~2분) |
| `--all` | `-a` | Full 모드 (모든 종/프리셋, 300 frames, ~25분) |
| `--species` | `-s` | 종 목록 (쉼표 구분) |
| `--presets` | `-p` | 프리셋 목록 (쉼표 구분) |
| `--max-frames` | `-m` | 최대 프레임 수 |
| `--output` | `-o` | 출력 디렉토리 |
| `--visualize-only` | | 실험 스킵, 시각화만 생성 |
| `--input` | `-i` | visualize-only용 입력 디렉토리 |

**모드별 설정:**

| 모드 | Frames | Species | Presets | GIFs |
|------|--------|---------|---------|------|
| `debug` | 50 | mouse | full, minimal | No |
| `standard` | 200 | mouse, dog | full, standard, minimal | Yes |
| `full` | 300 | mouse, dog, horse | all 5 presets | Yes |

---

## Stage 4: Action Recognition Model Evaluation

딥러닝 기반 행동 인식 모델들을 평가하고 비교합니다.

### 실행

```bash
# 빠른 테스트 (~1분)
./run_evaluation.sh --debug

# 표준 평가 (~5분)
./run_evaluation.sh

# 전체 분석 (~15분)
./run_evaluation.sh --full

# 통합 실행 (키포인트 추출 + 모델 평가)
./run_complete.sh --debug    # ~3분
./run_complete.sh            # ~15분
./run_complete.sh --full     # ~45분
```

### 지원 데이터셋 (Dual Label System)

| 데이터셋 | Label Type | 클래스 | 용도 |
|----------|------------|--------|------|
| **locomotion_sample** | Locomotion | stationary, walking, running, other | 이동 행동 분석 |
| **mars_sample** | Social | other, attack, mount, investigation | 사회적 상호작용 |

### 지원 모델

| 모델 | 설명 | 특징 |
|------|------|------|
| **Rule-Based** | 속도 기반 규칙 분류 | 베이스라인, 빠름 |
| **MLP** | Multi-Layer Perceptron | 프레임 단위 분류 |
| **LSTM** | Long Short-Term Memory | 시계열 패턴 학습 |
| **Transformer** | Self-Attention 기반 | 장거리 의존성 |

### 키포인트 프리셋별 평가

| 프리셋 | 키포인트 수 | 특징 |
|--------|-------------|------|
| **full** | 27 | 모든 키포인트, 최고 정확도 |
| **minimal** | 3 | nose, tailbase, tailend |
| **locomotion** | 5 | 이동 분석 최적화 |

### 평가 결과 예시

```
Best Model: minimal_lstm
- Accuracy: 96.1%
- F1 Macro: 95.6%

Model Comparison (full keypoints):
┌─────────────┬──────────┬─────────┐
│ Model       │ Accuracy │ F1      │
├─────────────┼──────────┼─────────┤
│ LSTM        │ 95.3%    │ 95.4%   │
│ MLP         │ 92.7%    │ 92.0%   │
│ Transformer │ 82.8%    │ 82.5%   │
│ Rule-Based  │ 32.0%    │ 12.1%   │
└─────────────┴──────────┴─────────┘
```

### 출력 구조

```
outputs/evaluation/
├── evaluation_results.json     # 전체 결과 (JSON)
├── models/                     # 학습된 모델 가중치
│   ├── full_mlp.pt
│   ├── full_lstm.pt
│   └── full_transformer.pt
└── plots/                      # 시각화
    └── confusion_matrices.png
```

---

## GitHub Repository

### 저장소 클론

```bash
git clone https://github.com/kafkapple/superanimal-behavior-poc.git
cd superanimal-behavior-poc
```

### 제외되는 파일 (Git Ignored)

| 디렉토리/파일 | 설명 | 용량 |
|---------------|------|------|
| `data/` | 원본 비디오, 데이터셋 | ~GB |
| `outputs/` | 실험 결과, 모델 | ~GB |
| `*.mp4, *.avi` | 비디오 파일 | Large |
| `*.h5, *.pt, *.pth` | 키포인트, 모델 가중치 | Large |
| `*.npy, *.npz` | NumPy 배열 | Large |

### 재현을 위한 데이터 다운로드

```bash
# 환경 설치 후
conda activate superanimal-poc

# 샘플 비디오 다운로드 (자동)
python run.py --help  # 첫 실행 시 자동 다운로드

# 또는 수동 다운로드
python -m src.scripts.download_samples
```

---

## 참고 자료

- [DeepLabCut 공식 문서](https://deeplabcut.github.io/DeepLabCut/)
- [SuperAnimal Model Zoo](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html)
- [SuperAnimal 논문](https://www.nature.com/articles/s41467-024-48792-2)
- [MARS Dataset](https://neuroethology.github.io/MARS/) - Mouse Social Behavior
- [CalMS21 Dataset](https://data.caltech.edu/records/s0vdx-0k302) - Multi-Agent Behavior
- [AP-10K Dataset](https://github.com/AlexTheBad/AP-10K) - Animal Pose Benchmark
- [COCO Keypoints](https://cocodataset.org/) - Human Pose Benchmark

---

## 라이선스

MIT License
