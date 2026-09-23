#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Open-Set Recognition: End-to-End Pipeline Runner
# Usage:
#   bash task4/run_task4.sh [DATA_ROOT] [OUTPUT_DIR]
# Example:
#   bash task4/run_task4.sh ./data/cifar task4/results
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT="${1:-./data/cifar}"
OUTPUT_DIR="${2:-task4/results}"
BATCH_SIZE="${3:-}"

EXTRA_ARGS=""
if [ -n "$BATCH_SIZE" ]; then
    EXTRA_ARGS="--batch_size $BATCH_SIZE"
    echo " Batch size override: ${BATCH_SIZE}"
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_ROOT"

echo "======================================================================"
echo " Starting Task 4 Pipeline (A100 80GB Optimized)"
echo " Data root:   ${DATA_ROOT}"
echo " Output dir:  ${OUTPUT_DIR}"
echo " Date:        $(date)"
echo "======================================================================"

# 1. Download & verify CIFAR-10 & CIFAR-100 datasets (fast CDN mirror)
echo -e "\n[Step 1/5] Checking CIFAR datasets..."
python utilities/download_cifar.py --dest "${DATA_ROOT}"

# 2. Train Vanilla ResNet-18
echo -e "\n[Step 2/5] Training Vanilla ResNet-18..."
python task4/train.py --config task4/configs/vanilla.yaml --data_root "${DATA_ROOT}" ${EXTRA_ARGS}

# 3. Train GCSC (Gaussian Center-based Spherical Classifier)
echo -e "\n[Step 3/5] Training GCSC..."
python task4/train.py --config task4/configs/gcsc.yaml --data_root "${DATA_ROOT}" ${EXTRA_ARGS}

# 4. Train PROSER (Learning Placeholders for Open-Set Recognition)
echo -e "\n[Step 4/5] Training PROSER..."
python task4/train.py --config task4/configs/proser.yaml --data_root "${DATA_ROOT}" ${EXTRA_ARGS}

# 5. Extract penultimate features & logits
echo -e "\n[Step 5/5 Part A] Extracting penultimate features and logits..."
python task4/extract_outputs.py --data_root "${DATA_ROOT}"

# 6. Evaluate all post-hoc scores & methods, generate tables, CSVs, and figures
echo -e "\n[Step 5/5 Part B] Evaluating OSR methods and exporting tables/figures..."
python task4/evaluate_osr.py --data_root "${DATA_ROOT}" --output_dir "${OUTPUT_DIR}"

echo -e "\n======================================================================"
echo " Task 4 Complete! Results saved to ${OUTPUT_DIR}/"
echo " Summary CSVs generated:"
echo "   - ${OUTPUT_DIR}/table1.csv (Post-hoc AUROC & FPR@95)"
echo "   - ${OUTPUT_DIR}/table2.csv (Method Comparisons: Vanilla vs GCSC vs PROSER)"
echo "   - ${OUTPUT_DIR}/final_results.csv (Comprehensive summary table)"
echo "   - ${OUTPUT_DIR}/per_class_unknown_rejection.csv (Per-class near/far rejection)"
echo "   - ${OUTPUT_DIR}/failure_analysis.csv (Top false-positive failure analysis)"
echo "======================================================================"
