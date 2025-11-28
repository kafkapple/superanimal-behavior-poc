#!/bin/bash
# =============================================================================
# Behavior Action Recognition - Comprehensive Evaluation Runner
# =============================================================================
#
# 행동 인식 모델 평가를 위한 통합 실행 스크립트
# - 여러 데이터셋 지원 (MARS, CalMS21, Custom)
# - 여러 모델 비교 (Rule-based, MLP, LSTM, Transformer)
# - 여러 키포인트 프리셋 비교 (full, minimal, locomotion)
# - Train/Val/Test 분할로 적절한 평가
#
# Usage:
#   ./run_evaluation.sh                    # 표준 모드 (~5 min)
#   ./run_evaluation.sh --debug            # 빠른 테스트 (~1 min)
#   ./run_evaluation.sh --full             # 전체 분석 (~15 min)
#   ./run_evaluation.sh --help             # 도움말
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${MAGENTA}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

show_help() {
    cat << 'EOF'
행동 인식 모델 평가 - 통합 실행 스크립트

Usage: ./run_evaluation.sh [MODE] [OPTIONS]

모드 (하나 선택):
    --debug, -d          빠른 테스트: rule_based만, 1 epoch (~1 min)
    --quick, -q          빠른 평가: rule_based + mlp, 10 epochs (~3 min)
    (default)            표준 모드: rule_based + mlp + lstm, 20 epochs (~5 min)
    --full, -f           전체 분석: 모든 모델, 50 epochs (~15 min)

옵션:
    --dataset NAME       데이터셋 이름:
                         - locomotion_sample: stationary/walking/running (기본값)
                         - mars_sample: attack/mount/investigation (사회행동)
                         - mars, calms21: 실제 데이터셋
    --dataset-path PATH  데이터셋 경로 (실제 데이터셋용)
    --models MODELS      쉼표로 구분된 모델 (rule_based,mlp,lstm,transformer)
    --presets PRESETS    쉼표로 구분된 키포인트 프리셋 (full,minimal,locomotion)
    --epochs N           학습 에포크 수
    --batch-size N       배치 크기
    --output DIR         출력 디렉토리
    --verbose, -v        상세 출력
    --help, -h           도움말 표시

예시:
    # 빠른 디버그 (~1 min)
    ./run_evaluation.sh --debug

    # 빠른 평가 (~3 min)
    ./run_evaluation.sh --quick

    # 표준 평가 (~5 min)
    ./run_evaluation.sh

    # 전체 평가 (~15 min)
    ./run_evaluation.sh --full

    # 커스텀: 특정 모델만
    ./run_evaluation.sh --models rule_based,lstm --epochs 30

    # 실제 MARS 데이터셋 사용
    ./run_evaluation.sh --full --dataset mars --dataset-path /path/to/mars

모드 비교:
    ┌─────────────┬──────────┬────────────────────────────────┬──────────┐
    │ 모드        │ Epochs   │ 모델                           │ 예상시간 │
    ├─────────────┼──────────┼────────────────────────────────┼──────────┤
    │ --debug     │ 1        │ rule_based                     │ ~1 min   │
    │ --quick     │ 10       │ rule_based, mlp                │ ~3 min   │
    │ (default)   │ 20       │ rule_based, mlp, lstm          │ ~5 min   │
    │ --full      │ 50       │ rule_based, mlp, lstm, transf. │ ~15 min  │
    └─────────────┴──────────┴────────────────────────────────┴──────────┘

출력:
    결과 저장 위치: outputs/evaluation/
    ├── evaluation_results.json    # 전체 평가 결과
    ├── datasets/                  # 생성된 샘플 데이터셋
    └── models/                    # 학습된 모델 (선택적)

평가 지표:
    - Accuracy: 전체 정확도
    - F1 (Macro): 클래스별 F1 점수의 평균
    - F1 per class: 각 클래스별 F1 점수
    - Confusion Matrix: 혼동 행렬

행동 클래스:
  [locomotion_sample - 속도 기반]
    - stationary (0): 정지
    - walking (1): 걷기
    - running (2): 달리기
    - other (3): 기타

  [mars_sample - 사회행동]
    - other (0): 기타
    - attack (1): 공격
    - mount (2): 마운트
    - investigation (3): 탐색

EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "행동 인식 모델 평가 (Behavior Recognition Evaluation)"

    # Default values
    local mode="standard"
    local dataset="locomotion_sample"  # stationary/walking/running
    local dataset_path=""
    local models=""
    local presets="full,minimal"
    local epochs=""
    local batch_size="32"
    local output_dir="outputs/evaluation"
    local verbose=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug|-d)
                mode="debug"
                shift
                ;;
            --quick|-q)
                mode="quick"
                shift
                ;;
            --full|-f)
                mode="full"
                shift
                ;;
            --dataset)
                dataset="$2"
                shift 2
                ;;
            --dataset-path)
                dataset_path="$2"
                shift 2
                ;;
            --models|-m)
                models="$2"
                shift 2
                ;;
            --presets|-p)
                presets="$2"
                shift 2
                ;;
            --epochs|-e)
                epochs="$2"
                shift 2
                ;;
            --batch-size|-b)
                batch_size="$2"
                shift 2
                ;;
            --output|-o)
                output_dir="$2"
                shift 2
                ;;
            --verbose|-v)
                verbose="--verbose"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Set defaults based on mode
    case $mode in
        debug)
            print_info "모드: DEBUG (빠른 테스트, ~1 min)"
            models="${models:-rule_based}"
            epochs="${epochs:-1}"
            presets="full"
            ;;
        quick)
            print_info "모드: QUICK (빠른 평가, ~3 min)"
            models="${models:-rule_based,mlp}"
            epochs="${epochs:-10}"
            ;;
        standard)
            print_info "모드: STANDARD (표준 평가, ~5 min)"
            models="${models:-rule_based,mlp,lstm}"
            epochs="${epochs:-20}"
            ;;
        full)
            print_info "모드: FULL (전체 분석, ~15 min)"
            models="${models:-rule_based,mlp,lstm,transformer}"
            epochs="${epochs:-50}"
            presets="full,minimal,locomotion"
            ;;
    esac

    # Show configuration
    echo ""
    echo -e "${CYAN}설정:${NC}"
    echo -e "  데이터셋: ${YELLOW}$dataset${NC}"
    echo -e "  모델: ${YELLOW}$models${NC}"
    echo -e "  프리셋: ${YELLOW}$presets${NC}"
    echo -e "  에포크: ${YELLOW}$epochs${NC}"
    echo -e "  출력: ${YELLOW}$output_dir${NC}"
    echo ""

    cd "${SCRIPT_DIR}"

    # Build command
    local cmd="python run_evaluation.py"
    cmd="$cmd --mode quick"  # Always use quick mode for Python script
    cmd="$cmd --dataset $dataset"
    cmd="$cmd --models ${models//,/ }"
    cmd="$cmd --presets ${presets//,/ }"
    cmd="$cmd --epochs $epochs"
    cmd="$cmd --batch-size $batch_size"
    cmd="$cmd --output-dir $output_dir"

    if [ -n "$dataset_path" ]; then
        cmd="$cmd --dataset-path $dataset_path"
    fi

    if [ -n "$verbose" ]; then
        cmd="$cmd $verbose"
    fi

    # Run evaluation
    print_info "실행: $cmd"
    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────${NC}"
    echo ""

    if eval $cmd; then
        echo ""
        print_header "평가 완료! (Evaluation Complete)"
        print_success "결과 저장 위치: $output_dir"

        # Show results file
        if [ -f "$output_dir/evaluation_results.json" ]; then
            print_info "결과 파일: $output_dir/evaluation_results.json"
            echo ""
            echo -e "${CYAN}요약 (Summary):${NC}"
            python3 -c "
import json
with open('$output_dir/evaluation_results.json') as f:
    data = json.load(f)
print(f\"  Best Model: {data['best_model']}\")
print(f\"  Best Accuracy: {data['best_accuracy']:.4f}\")
print(f\"  Best F1: {data['best_f1']:.4f}\")
print()
print('  모델별 결과:')
for row in data['comparison_table']:
    print(f\"    {row['keypoint_preset']}/{row['model']}: Acc={row['accuracy']:.3f}, F1={row['f1_macro']:.3f}\")
" 2>/dev/null || true
        fi
    else
        echo ""
        print_error "평가 실패!"
        exit 1
    fi
}

# Run main with all arguments
main "$@"
