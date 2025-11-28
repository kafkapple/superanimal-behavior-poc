#!/bin/bash
# =============================================================================
# SuperAnimal Behavior PoC - 종합 실행 스크립트
# =============================================================================
#
# 모든 실험을 한 번에 실행:
# 1. 키포인트 추출 + 행동 분류 (run_all.sh)
# 2. 행동 인식 모델 평가 (run_evaluation.sh)
#
# Usage:
#   ./run_complete.sh                    # 표준 모드 (~15 min)
#   ./run_complete.sh --debug            # 빠른 테스트 (~3 min)
#   ./run_complete.sh --full             # 전체 분석 (~45 min)
#   ./run_complete.sh --help             # 도움말
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
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo ""
    echo -e "${WHITE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${WHITE}║                                                                  ║${NC}"
    echo -e "${WHITE}║   ${CYAN}SuperAnimal Behavior PoC - 종합 파이프라인${WHITE}                    ║${NC}"
    echo -e "${WHITE}║                                                                  ║${NC}"
    echo -e "${WHITE}║   ${YELLOW}1. 키포인트 추출 & 행동 분류${WHITE}                                 ║${NC}"
    echo -e "${WHITE}║   ${YELLOW}2. 행동 인식 모델 학습 & 평가${WHITE}                                ║${NC}"
    echo -e "${WHITE}║                                                                  ║${NC}"
    echo -e "${WHITE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_header() {
    echo ""
    echo -e "${MAGENTA}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}──────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}  STEP $1: $2${NC}"
    echo -e "${BLUE}──────────────────────────────────────────────────────────────────${NC}"
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
SuperAnimal Behavior PoC - 종합 실행 스크립트

모든 실험을 한 번에 실행합니다:
  1. 키포인트 추출 & 행동 분류 (run_all.sh)
  2. 행동 인식 모델 학습 & 평가 (run_evaluation.sh)

Usage: ./run_complete.sh [MODE] [OPTIONS]

모드 (하나 선택):
    --debug, -d          빠른 테스트 (~3 min)
                         - 키포인트: mouse, 50 frames, 2 presets
                         - 평가: rule_based, 1 epoch

    --quick, -q          빠른 평가 (~8 min)
                         - 키포인트: mouse, 100 frames, 3 presets
                         - 평가: rule_based + mlp, 10 epochs

    (default)            표준 모드 (~15 min)
                         - 키포인트: mouse+dog, 200 frames, 3 presets
                         - 평가: rule_based + mlp + lstm, 20 epochs

    --full, -f           전체 분석 (~45 min)
                         - 키포인트: 모든 종, 300 frames, 모든 presets
                         - 평가: 모든 모델, 50 epochs

옵션:
    --keypoint-only      키포인트 추출만 실행 (평가 스킵)
    --eval-only          평가만 실행 (키포인트 스킵)
    --output DIR         출력 디렉토리
    --verbose, -v        상세 출력
    --help, -h           도움말 표시

예시:
    # 빠른 디버그 (~3 min)
    ./run_complete.sh --debug

    # 표준 실행 (~15 min)
    ./run_complete.sh

    # 전체 분석 (~45 min)
    ./run_complete.sh --full

    # 키포인트만 실행
    ./run_complete.sh --keypoint-only

    # 평가만 실행
    ./run_complete.sh --eval-only --full

모드 비교:
    ┌─────────────┬────────────────────────────────┬────────────────────────────┬──────────┐
    │ 모드        │ 키포인트 추출                  │ 모델 평가                  │ 예상시간 │
    ├─────────────┼────────────────────────────────┼────────────────────────────┼──────────┤
    │ --debug     │ mouse, 50fr, 2 presets         │ rule_based, 1 epoch        │ ~3 min   │
    │ --quick     │ mouse, 100fr, 3 presets        │ rule_based+mlp, 10 epochs  │ ~8 min   │
    │ (default)   │ mouse+dog, 200fr, 3 presets    │ +lstm, 20 epochs           │ ~15 min  │
    │ --full      │ all species, 300fr, 5 presets  │ all models, 50 epochs      │ ~45 min  │
    └─────────────┴────────────────────────────────┴────────────────────────────┴──────────┘

출력 구조:
    outputs/
    ├── comprehensive/<timestamp>/     # 키포인트 & 행동 분류 결과
    │   ├── single_video/
    │   ├── keypoint_comparison/
    │   ├── visualizations/
    │   └── final_dashboard.html
    │
    └── evaluation/                    # 모델 평가 결과
        ├── evaluation_results.json
        └── datasets/

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
    local output_dir=""
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
            --keypoint-only)
                run_eval=false
                shift
                ;;
            --eval-only)
                run_keypoint=false
                shift
                ;;
            --output|-o)
                output_dir="$2"
                shift 2
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

    # Show mode info
    case $mode in
        debug)
            print_info "모드: DEBUG (빠른 테스트, ~3 min)"
            echo ""
            echo -e "  ${CYAN}키포인트:${NC} mouse, 50 frames, 2 presets"
            echo -e "  ${CYAN}평가:${NC} rule_based, 1 epoch"
            ;;
        quick)
            print_info "모드: QUICK (빠른 평가, ~8 min)"
            echo ""
            echo -e "  ${CYAN}키포인트:${NC} mouse, 100 frames, 3 presets"
            echo -e "  ${CYAN}평가:${NC} rule_based + mlp, 10 epochs"
            ;;
        standard)
            print_info "모드: STANDARD (표준 평가, ~15 min)"
            echo ""
            echo -e "  ${CYAN}키포인트:${NC} mouse + dog, 200 frames, 3 presets"
            echo -e "  ${CYAN}평가:${NC} rule_based + mlp + lstm, 20 epochs"
            ;;
        full)
            print_info "모드: FULL (전체 분석, ~45 min)"
            echo ""
            echo -e "  ${CYAN}키포인트:${NC} 모든 종, 300 frames, 모든 presets"
            echo -e "  ${CYAN}평가:${NC} 모든 모델, 50 epochs"
            ;;
    esac

    echo ""

    cd "${SCRIPT_DIR}"

    local step=1
    local total_steps=0
    [[ "$run_keypoint" == true ]] && ((total_steps++))
    [[ "$run_eval" == true ]] && ((total_steps++))

    # =========================================================================
    # STEP 1: 키포인트 추출 & 행동 분류
    # =========================================================================
    if [[ "$run_keypoint" == true ]]; then
        print_step "$step/$total_steps" "키포인트 추출 & 행동 분류"

        local kp_args=""
        case $mode in
            debug)
                kp_args="--debug"
                ;;
            quick)
                kp_args="--debug"  # Use debug for quick keypoint
                ;;
            standard)
                kp_args=""  # Default
                ;;
            full)
                kp_args="--all"
                ;;
        esac

        if [ -n "$verbose" ]; then
            kp_args="$kp_args $verbose"
        fi

        print_info "실행: ./run_all.sh $kp_args"
        echo ""

        if ./run_all.sh $kp_args; then
            print_success "키포인트 추출 완료!"
        else
            print_error "키포인트 추출 실패!"
            exit 1
        fi

        ((step++))
    fi

    # =========================================================================
    # STEP 2: 행동 인식 모델 평가
    # =========================================================================
    if [[ "$run_eval" == true ]]; then
        print_step "$step/$total_steps" "행동 인식 모델 평가"

        local eval_args=""
        case $mode in
            debug)
                eval_args="--debug"
                ;;
            quick)
                eval_args="--quick"
                ;;
            standard)
                eval_args=""  # Default
                ;;
            full)
                eval_args="--full"
                ;;
        esac

        if [ -n "$verbose" ]; then
            eval_args="$eval_args $verbose"
        fi

        print_info "실행: ./run_evaluation.sh $eval_args"
        echo ""

        if ./run_evaluation.sh $eval_args; then
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
    print_header "🎉 종합 파이프라인 완료!"

    echo -e "${GREEN}결과 위치:${NC}"
    echo ""

    if [[ "$run_keypoint" == true ]]; then
        local latest_kp=$(ls -td outputs/comprehensive/*/ 2>/dev/null | head -1)
        if [ -n "$latest_kp" ]; then
            echo -e "  ${CYAN}키포인트 결과:${NC} $latest_kp"
            if [ -f "${latest_kp}final_dashboard.html" ]; then
                echo -e "  ${CYAN}대시보드:${NC} ${latest_kp}final_dashboard.html"
            fi
        fi
    fi

    if [[ "$run_eval" == true ]]; then
        echo -e "  ${CYAN}평가 결과:${NC} outputs/evaluation/evaluation_results.json"
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
print(f\"  Best Accuracy: {data['best_accuracy']:.4f}\")
print(f\"  Best F1: {data['best_f1']:.4f}\")
" 2>/dev/null || true
    fi

    echo ""
    print_success "모든 실험 완료!"

    # Open dashboard if available
    if [[ "$run_keypoint" == true ]]; then
        local latest_kp=$(ls -td outputs/comprehensive/*/ 2>/dev/null | head -1)
        if [ -n "$latest_kp" ] && [ -f "${latest_kp}final_dashboard.html" ]; then
            if command -v open &> /dev/null; then
                open "${latest_kp}final_dashboard.html" 2>/dev/null || true
            elif command -v xdg-open &> /dev/null; then
                xdg-open "${latest_kp}final_dashboard.html" 2>/dev/null || true
            fi
        fi
    fi
}

# Run main with all arguments
main "$@"
