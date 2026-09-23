#!/usr/bin/env bash
# task2/run_task2.sh — One-command pipeline for Task 2 (Unsupervised Domain Adaptation)
#
# Trains all methods in the correct order, then runs final evaluation.
# Methods trained: source_only → DAN(λ=1) → DAN(λ=0.1) → DAN(λ=10) → DANN → CDAN
# Then evaluates: all checkpoints → all required PA deliverables
#
# Usage:
#   bash task2/run_task2.sh /path/to/pacs
#   PACS_ROOT=/path/to/pacs bash task2/run_task2.sh
#   bash task2/run_task2.sh /path/to/pacs --skip-source-only   # if checkpoint exists
#   bash task2/run_task2.sh /path/to/pacs --eval-only          # only run evaluation
#
# Environment:
#   PACS_ROOT   — override path to PACS dataset root (or pass as $1)
#   OUTPUT_DIR  — override output directory (default: task2/results)
#
# Hardware notes:
#   Recommended: NVIDIA A100 40GB (Jarvis).
#   The spec batch size (8/24 = 48 total) uses ~600 MB VRAM for ResNet-18.
#   All 6 training runs + evaluation complete in ~35-45 min on A100 40GB.
#   TF32 is automatically enabled when an Ampere GPU is detected in train.py.

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
PACS_ROOT="${1:-${PACS_ROOT:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-task2/results}"
SKIP_SOURCE_ONLY=0
EVAL_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --skip-source-only) SKIP_SOURCE_ONLY=1 ;;
        --eval-only)        EVAL_ONLY=1 ;;
    esac
done

if [[ -z "$PACS_ROOT" ]]; then
    echo "ERROR: PACS root not specified."
    echo "Usage: bash task2/run_task2.sh /path/to/pacs"
    exit 1
fi

if [[ ! -d "$PACS_ROOT" ]]; then
    echo "ERROR: PACS root does not exist: $PACS_ROOT"
    exit 1
fi

echo "================================================================"
echo "  Task 2 — Unsupervised Domain Adaptation Pipeline"
echo "  PACS root:  $PACS_ROOT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Start:      $(date)"
echo "================================================================"

mkdir -p "$OUTPUT_DIR/figures"

LOG_FILE="$OUTPUT_DIR/run_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Helpers ───────────────────────────────────────────────────────────────────
run_step() {
    local desc="$1"
    shift
    echo ""
    echo "── $desc ─────────────────────────────────────────────────────────"
    echo "   CMD: $*"
    echo "   Time: $(date)"
    "$@"
    local ec=$?
    if [[ $ec -ne 0 ]]; then
        echo "   FAILED (exit code $ec): $desc"
        exit $ec
    fi
    echo "   ✓ Done: $desc"
}

# Navigate to repo root (script lives in task2/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $EVAL_ONLY -eq 0 ]]; then
    # ── Step 1: Source-only ERM ───────────────────────────────────────────────
    if [[ $SKIP_SOURCE_ONLY -eq 0 ]]; then
        run_step "Source-only ERM (Task 3 checkpoint)" \
            python task2/train.py \
                --config task2/configs/source_only.yaml \
                --pacs_root "$PACS_ROOT"
    else
        echo ""
        echo "── Source-only: SKIPPED (--skip-source-only flag) ──────────────────"
        if [[ ! -f "$OUTPUT_DIR/source_only_checkpoint.pth" ]]; then
            echo "   WARNING: source_only_checkpoint.pth not found. Task 3 ERM may break."
        fi
    fi

    # ── Step 2: DAN main comparison (λ=1) ────────────────────────────────────
    run_step "DAN — λ_MMD=1.0 (main comparison)" \
        python task2/train.py \
            --config task2/configs/dan.yaml \
            --pacs_root "$PACS_ROOT"

    # ── Step 3: DAN controlled study (λ=0.1) ─────────────────────────────────
    run_step "DAN — λ_MMD=0.1 (controlled study)" \
        python task2/train.py \
            --config task2/configs/dan.yaml \
            --pacs_root "$PACS_ROOT" \
            --lambda_mmd 0.1

    # ── Step 4: DAN controlled study (λ=10) ──────────────────────────────────
    run_step "DAN — λ_MMD=10.0 (controlled study)" \
        python task2/train.py \
            --config task2/configs/dan.yaml \
            --pacs_root "$PACS_ROOT" \
            --lambda_mmd 10.0

    # ── Step 5: DANN main comparison (α_max=1.0) ─────────────────────────────
    run_step "DANN — Adversarial alignment (λ_adv=1.0, α_max=1.0, single-pass GRL)" \
        python task2/train.py \
            --config task2/configs/dann.yaml \
            --pacs_root "$PACS_ROOT"

    # ── Step 5b: DANN controlled study (α_max=0.25, 0.5) ─────────────────────
    # Spec option: "vary the maximum gradient-reversal strength over {0.25, 0.5, 1}"
    # These are OPTIONAL — only needed if you choose DANN alpha as your controlled study
    # (vs the DAN lambda study above). Uncomment if needed:
    #
    # run_step "DANN — α_max=0.25 (controlled study)" \
    #     python task2/train.py \
    #         --config task2/configs/dann.yaml \
    #         --pacs_root "$PACS_ROOT" \
    #         --max_alpha 0.25
    #
    # run_step "DANN — α_max=0.5 (controlled study)" \
    #     python task2/train.py \
    #         --config task2/configs/dann.yaml \
    #         --pacs_root "$PACS_ROOT" \
    #         --max_alpha 0.5

    # ── Step 6: CDAN ──────────────────────────────────────────────────────────
    run_step "CDAN — Class-conditional adversarial alignment (λ_adv=1.0)" \
        python task2/train.py \
            --config task2/configs/cdan.yaml \
            --pacs_root "$PACS_ROOT"
fi

# ── Step 7: Final evaluation ──────────────────────────────────────────────────
run_step "Final evaluation — all methods, all PA deliverables" \
    python task2/evaluate_final.py \
        --pacs_root "$PACS_ROOT" \
        --output_dir "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "  Task 2 COMPLETE"
echo "  End: $(date)"
echo ""
echo "  Key outputs:"
echo "    $OUTPUT_DIR/final_results.csv                     — main table"
echo "    $OUTPUT_DIR/per_class_results.csv                 — per-class Sketch acc"
echo "    $OUTPUT_DIR/top_confusions.csv                    — failure analysis"
echo "    $OUTPUT_DIR/figures/per_class_delta.png"
echo "    $OUTPUT_DIR/figures/controlled_study_lambda_mmd.png  — DAN study"
echo "    $OUTPUT_DIR/figures/controlled_study_dann_alpha.png  — DANN study (if run)"
echo "    $OUTPUT_DIR/figures/{method}_confusion.png"
echo "    $OUTPUT_DIR/figures/{method}_curves.png"
echo "================================================================"
