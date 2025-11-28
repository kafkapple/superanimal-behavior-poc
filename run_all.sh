#!/bin/bash
# =============================================================================
# SuperAnimal Behavior PoC - 통합 실행 스크립트
# =============================================================================
#
# 전체 파이프라인을 한 번에 실행:
#   1. 키포인트 추출 (run.py, run_keypoint_comparison.py, run_cross_species.py)
#   2. 행동 인식 모델 평가 (run_evaluation.py)
#
# Usage:
#   ./run_all.sh                    # 표준 모드 (~10 min)
#   ./run_all.sh --debug            # 빠른 테스트 (~3 min)
#   ./run_all.sh --full             # 전체 분석 (~30 min)
#   ./run_all.sh --help             # 도움말
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="outputs/full_pipeline/${TIMESTAMP}"

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   SuperAnimal Behavior PoC - 통합 파이프라인                       ║${NC}"
    echo -e "${CYAN}║                                                                  ║${NC}"
    echo -e "${CYAN}║   Step 1: Keypoint Extraction (Pose Estimation)                  ║${NC}"
    echo -e "${CYAN}║   Step 2: Action Recognition (Behavior Classification)           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${MAGENTA}──────────────────────────────────────────────────────────────────${NC}"
    echo -e "${MAGENTA}  STEP $1: $2${NC}"
    echo -e "${MAGENTA}──────────────────────────────────────────────────────────────────${NC}"
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

show_help() {
    cat << 'EOF'
SuperAnimal Behavior PoC - 통합 실행 스크립트

Usage: ./run_all.sh [MODE] [OPTIONS]

모드 (하나 선택):
    --debug, -d          빠른 테스트 (~3 min)
                         - 키포인트: mouse, 50 frames
                         - 평가: rule_based + mlp, 5 epochs

    (default)            표준 모드 (~10 min)
                         - 키포인트: mouse + dog, 200 frames
                         - 평가: all models, 20 epochs

    --full, -f           전체 분석 (~30 min)
                         - 키포인트: 모든 종, 300 frames
                         - 평가: all models, 50 epochs

옵션:
    --keypoint-only      키포인트 추출만 실행
    --eval-only          평가만 실행
    --verbose, -v        상세 출력
    --help, -h           도움말 표시

예시:
    ./run_all.sh --debug           # 빠른 테스트 (~3 min)
    ./run_all.sh                   # 표준 실행 (~10 min)
    ./run_all.sh --full            # 전체 분석 (~30 min)
    ./run_all.sh --eval-only       # 평가만 실행

모드 비교:
    ┌─────────────┬────────────────────────────┬────────────────────────────┬──────────┐
    │ 모드        │ 키포인트 추출              │ 모델 평가                  │ 예상시간 │
    ├─────────────┼────────────────────────────┼────────────────────────────┼──────────┤
    │ --debug     │ mouse, 50 frames           │ rule_based + mlp, 5 epochs │ ~3 min   │
    │ (default)   │ mouse + dog, 200 frames    │ all models, 20 epochs      │ ~10 min  │
    │ --full      │ all species, 300 frames    │ all models, 50 epochs      │ ~30 min  │
    └─────────────┴────────────────────────────┴────────────────────────────┴──────────┘

출력 구조:
    outputs/
    ├── full_pipeline/<timestamp>/   # 키포인트 & 시각화 결과
    │   ├── single_video/
    │   ├── keypoint_comparison/
    │   └── cross_species/
    │
    └── evaluation/                  # 모델 평가 결과
        ├── evaluation_results.json
        └── models/

EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_banner

    # Default values
    local mode="standard"
    local run_keypoint=true
    local run_eval=true
    local verbose=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug|-d)
                mode="debug"
                shift
                ;;
            --full|-f)
                mode="full"
                shift
                ;;
            --keypoint-only)
                run_eval=false
                shift
                ;;
            --eval-only)
                run_keypoint=false
                shift
                ;;
            --verbose|-v)
                verbose="-v"
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

    # Set mode-specific parameters
    local max_frames=200
    local eval_mode="quick"

    case $mode in
        debug)
            max_frames=50
            eval_mode="demo"
            print_info "모드: DEBUG (빠른 테스트, ~3 min)"
            ;;
        standard)
            max_frames=200
            eval_mode="quick"
            print_info "모드: STANDARD (표준 평가, ~10 min)"
            ;;
        full)
            max_frames=300
            eval_mode="full"
            print_info "모드: FULL (전체 분석, ~30 min)"
            ;;
    esac

    echo ""
    cd "${SCRIPT_DIR}"
    mkdir -p "${OUTPUT_BASE}"

    local step=1
    local total_steps=0
    [[ "$run_keypoint" == true ]] && ((total_steps+=3))
    [[ "$run_eval" == true ]] && ((total_steps++))

    # =========================================================================
    # STEP 1: 단일 비디오 분석
    # =========================================================================
    if [[ "$run_keypoint" == true ]]; then
        print_step "$step/$total_steps" "단일 비디오 분석 (run.py)"

        print_info "실행: python run.py data.video.max_frames=$max_frames"

        if python run.py data.video.max_frames=$max_frames; then
            print_success "단일 비디오 분석 완료!"
        else
            print_error "단일 비디오 분석 실패!"
            exit 1
        fi
        ((step++))

        # =====================================================================
        # STEP 2: 키포인트 프리셋 비교
        # =====================================================================
        print_step "$step/$total_steps" "키포인트 프리셋 비교 (run_keypoint_comparison.py)"

        print_info "실행: python run_keypoint_comparison.py data.video.max_frames=$max_frames"

        if python run_keypoint_comparison.py data.video.max_frames=$max_frames; then
            print_success "키포인트 프리셋 비교 완료!"
        else
            print_error "키포인트 프리셋 비교 실패!"
            exit 1
        fi
        ((step++))

        # =====================================================================
        # STEP 3: Cross-Species 비교
        # =====================================================================
        print_step "$step/$total_steps" "Cross-Species 비교 (run_cross_species.py)"

        print_info "실행: python run_cross_species.py data.video.max_frames=$max_frames"

        if python run_cross_species.py data.video.max_frames=$max_frames; then
            print_success "Cross-Species 비교 완료!"
        else
            print_error "Cross-Species 비교 실패!"
            exit 1
        fi
        ((step++))
    fi

    # =========================================================================
    # STEP 4: 행동 인식 모델 평가
    # =========================================================================
    if [[ "$run_eval" == true ]]; then
        print_step "$step/$total_steps" "행동 인식 모델 평가 (run_evaluation.py)"

        print_info "실행: python run_evaluation.py --mode $eval_mode"

        if python run_evaluation.py --mode $eval_mode; then
            print_success "모델 평가 완료!"
        else
            print_error "모델 평가 실패!"
            exit 1
        fi
        ((step++))
    fi

    # =========================================================================
    # 최종 요약
    # =========================================================================
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   🎉 전체 파이프라인 완료!                                         ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${CYAN}결과 위치:${NC}"
    if [[ "$run_keypoint" == true ]]; then
        echo -e "  - 키포인트 결과: outputs/ (각 스크립트별 출력)"
    fi
    if [[ "$run_eval" == true ]]; then
        echo -e "  - 평가 결과: outputs/evaluation/evaluation_results.json"
    fi
    echo ""

    # Show evaluation summary
    if [[ "$run_eval" == true ]] && [ -f "outputs/evaluation/evaluation_results.json" ]; then
        echo -e "${CYAN}모델 평가 요약:${NC}"
        python3 -c "
import json
with open('outputs/evaluation/evaluation_results.json') as f:
    data = json.load(f)
print(f\"  Best Model: {data['best_model']}\")
print(f\"  Best Accuracy: {data['best_accuracy']:.1%}\")
print(f\"  Best F1: {data['best_f1']:.4f}\")
" 2>/dev/null || true
        echo ""
    fi

    print_success "완료!"
}

# Run main with all arguments
main "$@"
