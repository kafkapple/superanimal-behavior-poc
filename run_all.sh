#!/bin/bash
# =============================================================================
# SuperAnimal Behavior PoC - Unified Pipeline Runner
# =============================================================================
#
# All experiments are run through run_comprehensive.py for consistency.
# This script provides a simple bash interface with the same options.
#
# Usage:
#   ./run_all.sh                    # Standard mode (~10 min)
#   ./run_all.sh --debug            # Quick test (~2 min)
#   ./run_all.sh --debug-full       # All combinations, minimal frames (~5 min)
#   ./run_all.sh --all              # Full analysis (~30 min)
#   ./run_all.sh --help             # Show help
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
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
SuperAnimal Behavior PoC - Unified Pipeline Runner

Usage: ./run_all.sh [MODE] [OPTIONS]

MODES (choose one):
    --debug, -d          Quick test: mouse only, 50 frames (~2 min)
    --debug-full, -df    All combinations with minimal 20 frames (~5 min) ⭐
    (default)            Standard: mouse+dog, 200 frames (~10 min)
    --all, -a            Full analysis: all species/presets/models (~30 min)

OPTIONS:
    --species SPECIES    Comma-separated species (e.g., mouse,dog,horse)
    --presets PRESETS    Comma-separated presets (e.g., full,standard,minimal)
    --max-frames N       Maximum frames per video
    --output DIR         Output directory
    --labels FILE        Ground truth labels file (CSV, JSON, TXT, or NPY)
    --visualize-only     Skip experiments, regenerate visualizations only
    --input DIR          Input directory for --visualize-only
    --verbose, -v        Verbose output
    --help, -h           Show this help message

EXAMPLES:
    # Quick validation (~2 min)
    ./run_all.sh --debug

    # All combinations with minimal data - great for testing (~5 min)
    ./run_all.sh --debug-full

    # Standard analysis (~10 min)
    ./run_all.sh

    # Full comprehensive analysis (~30 min)
    ./run_all.sh --all

    # Custom: specific species and presets
    ./run_all.sh --species mouse,dog --presets full,standard,minimal

    # Custom: limit frames
    ./run_all.sh --all --max-frames 100

    # With ground truth labels for F1/Accuracy evaluation
    ./run_all.sh --labels data/labels.csv

    # Regenerate visualizations from existing results
    ./run_all.sh --visualize-only --input outputs/comprehensive/20241127_123456

MODE COMPARISON:
    ┌─────────────┬──────────┬────────┬─────────┬──────────┬────────┐
    │ Mode        │ Frames   │ Species│ Presets │ Models   │ Time   │
    ├─────────────┼──────────┼────────┼─────────┼──────────┼────────┤
    │ --debug     │ 50       │ 1      │ 2       │ 1        │ ~2 min │
    │ --debug-full│ 20       │ 3      │ 5       │ ALL      │ ~5 min │
    │ (default)   │ 200      │ 2      │ 3       │ 1        │ ~10min │
    │ --all       │ 300      │ 3      │ 5       │ ALL      │ ~30min │
    └─────────────┴──────────┴────────┴─────────┴──────────┴────────┘

OUTPUT:
    Results saved to: outputs/comprehensive/<timestamp>/
    ├── single_video/           # Per-species analysis
    ├── keypoint_comparison/    # Preset comparison results
    ├── cross_species/          # Cross-species comparison
    ├── visualizations/         # Charts and plots
    │   ├── keypoint_comparison/
    │   │   ├── hierarchical_action_comparison_*.png
    │   │   ├── confusion_matrix_grid_*.png
    │   │   └── performance_metrics.json
    │   ├── ground_truth_evaluation/  # When --labels provided
    │   │   ├── gt_metrics_report.json  # F1/Accuracy per preset
    │   │   └── gifs/
    │   │       ├── gt_comparison_*.gif     # Pred vs GT overlay
    │   │       └── per_class_acc_*.gif     # Per-class accuracy
    │   └── species_comparison/
    ├── report/
    │   └── dashboard.html
    └── final_dashboard.html    # Main dashboard (auto-opens)

EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "SuperAnimal Behavior PoC - Pipeline Runner"

    # Build command arguments
    local cmd_args=""
    local mode_set=false

    # Parse arguments and pass through to Python
    while [[ $# -gt 0 ]]; do
        case $1 in
            --debug|-d)
                if [ "$mode_set" = false ]; then
                    cmd_args="$cmd_args --debug"
                    mode_set=true
                fi
                shift
                ;;
            --debug-full|-df)
                if [ "$mode_set" = false ]; then
                    cmd_args="$cmd_args --debug-full"
                    mode_set=true
                fi
                shift
                ;;
            --all|-a|--comprehensive|-c)
                if [ "$mode_set" = false ]; then
                    cmd_args="$cmd_args --all"
                    mode_set=true
                fi
                shift
                ;;
            --species|-s)
                cmd_args="$cmd_args --species $2"
                shift 2
                ;;
            --presets|-p)
                cmd_args="$cmd_args --presets $2"
                shift 2
                ;;
            --max-frames|-m)
                cmd_args="$cmd_args --max-frames $2"
                shift 2
                ;;
            --output|-o)
                cmd_args="$cmd_args --output $2"
                shift 2
                ;;
            --visualize-only)
                cmd_args="$cmd_args --visualize-only"
                shift
                ;;
            --input|-i)
                cmd_args="$cmd_args --input $2"
                shift 2
                ;;
            --labels|-l)
                cmd_args="$cmd_args --labels $2"
                shift 2
                ;;
            --verbose|-v)
                cmd_args="$cmd_args --verbose"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --debug-only)
                # Legacy support
                cmd_args="$cmd_args --debug"
                mode_set=true
                shift
                ;;
            --full-only)
                # Legacy support - now same as default
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Show configuration
    if [[ "$cmd_args" == *"--debug-full"* ]]; then
        print_info "Mode: DEBUG-FULL (all combinations, 20 frames, ~5 min)"
    elif [[ "$cmd_args" == *"--debug"* ]]; then
        print_info "Mode: DEBUG (quick test, 50 frames, ~2 min)"
    elif [[ "$cmd_args" == *"--all"* ]]; then
        print_info "Mode: FULL (all species/presets/models, 300 frames, ~30 min)"
    else
        print_info "Mode: STANDARD (mouse+dog, 200 frames, ~10 min)"
    fi

    cd "${SCRIPT_DIR}"

    # Run the Python script
    echo ""
    print_info "Running: python run_comprehensive.py $cmd_args"
    echo ""

    if python run_comprehensive.py $cmd_args; then
        echo ""
        print_header "Pipeline Complete!"
        print_success "Check the output directory for results"

        # Find and open the latest dashboard
        local latest_output=$(ls -td outputs/comprehensive/*/ 2>/dev/null | head -1)
        if [ -n "$latest_output" ] && [ -f "${latest_output}final_dashboard.html" ]; then
            print_info "Dashboard: ${latest_output}final_dashboard.html"
            # Try to open in browser (macOS or Linux)
            if command -v open &> /dev/null; then
                open "${latest_output}final_dashboard.html" 2>/dev/null || true
            elif command -v xdg-open &> /dev/null; then
                xdg-open "${latest_output}final_dashboard.html" 2>/dev/null || true
            fi
        fi
    else
        echo ""
        print_error "Pipeline failed!"
        exit 1
    fi
}

# Run main with all arguments
main "$@"
