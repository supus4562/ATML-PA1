# PA1 — Beyond IID: Inductive Biases, Domain Adaptation, Domain Generalization, Open-Set Recognition

[GitHub link — placeholder]

## Overview

This repository implements Programming Assignment 1 for CS-6304 Advanced Topics in Machine
Learning. The four tasks explore what happens when standard machine-learning assumptions
break down:

| Task | Topic | Key question |
|------|-------|--------------|
| 1 | Inductive biases & feature representations | How do frozen ResNet-50, ViT-B/16, and CLIP differ in color, shape, texture, and spatial sensitivity? |
| 2 | Unsupervised domain adaptation | Can MMD (DAN), gradient-reversal (DANN), or conditional-domain (CDAN) alignment bridge the PACS Photo→Sketch gap? |
| 3 | Domain generalization | Does cross-source MMD or SAM sharpness regularization generalize to an unseen Sketch domain? |
| 4 | Open-set recognition | How well do post-hoc scores (MSP, MLS, Energy, Mahalanobis) and training-time methods (GCSC, PROSER) reject CIFAR-100 unknowns? |

---

## Installation

```bash
pip install -r requirements.txt
```

> **Python 3.10+** is required.  CUDA is strongly recommended (CPU training will be slow).

---

## Dataset Setup

### STL-10 (Task 1)
Downloaded automatically by `torchvision` on first run. No manual steps needed.

### CIFAR-10 / CIFAR-100 (Task 4)
Downloaded automatically by `torchvision` on first run.

### PACS (Tasks 2 & 3)
PACS must be downloaded manually.

1. Download from the [official PACS page](https://dali-dl.github.io/project_iccv2019.html)
   or [Kaggle](https://www.kaggle.com/datasets/nickfratto/pacs-dataset).
2. Unzip so the folder layout is:
   ```
   <pacs_root>/
     art_painting/
       dog/  elephant/  giraffe/ ...
     cartoon/
     photo/
     sketch/
   ```
3. Pass `--pacs_root <pacs_root>` to Task 2 and Task 3 commands below.

---

## Running Each Task

### Task 1 — Inductive Biases and Feature Representations
```bash
python task1/scripts/run_task1.py \
  --data_root ./data/stl10 \
  --output_dir task1/results
```
All figures saved to `task1/results/figures/`.

### Task 2 — Unsupervised Domain Adaptation
```bash
python task2/train.py --config task2/configs/source_only.yaml --pacs_root <path>
python task2/train.py --config task2/configs/dan.yaml        --pacs_root <path>
python task2/train.py --config task2/configs/dann.yaml       --pacs_root <path>
python task2/train.py --config task2/configs/cdan.yaml       --pacs_root <path>
python task2/evaluate_final.py --pacs_root <path> --output_dir task2/results
```

### Task 3 — Domain Generalization
```bash
# ERM baseline is reused from Task 2 (task2/results/source_only_checkpoint.pth)
python task3/train.py --config task3/configs/dan_dg.yaml --pacs_root <path>
python task3/train.py --config task3/configs/sam.yaml    --pacs_root <path>
python task3/evaluate_sketch.py --pacs_root <path> --output_dir task3/results
```

### Task 4 — Open-Set Recognition
# Run full pipeline in one command:
bash task4/run_task4.sh ./data/cifar task4/results

# Or step-by-step:
# 1. Optional: Pre-download CIFAR-10 & CIFAR-100 via high-speed mirror
python utilities/download_cifar.py --dest ./data/cifar

# Training (A100 80GB optimized batch_size=512; use --batch_size 128 for strict PDF baseline)
python task4/train.py --config task4/configs/vanilla.yaml --data_root ./data/cifar
python task4/train.py --config task4/configs/gcsc.yaml    --data_root ./data/cifar
python task4/train.py --config task4/configs/proser.yaml  --data_root ./data/cifar

# Extract penultimate features & logits (batch_size=1024, TF32 accelerated)
python task4/extract_outputs.py --data_root ./data/cifar

# Evaluate post-hoc scores & methods, export CSVs, failure analysis, and figures
python task4/evaluate_osr.py    --data_root ./data/cifar --output_dir task4/results
```

---

## Attribution

- **AdaIN style transfer** (Task 1 cue conflicts): Self-contained implementation of
  Adaptive Instance Normalization following Huang & Belongie, ICCV 2017.
  Inspired by [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) (MIT license).
- **PROSER** (Task 4): Placeholder-class OpenSet Recognition, following
  Zhou et al., "Learning Placeholders for Open-Set Recognition", CVPR 2021.
- **SAM** (Task 3): Sharpness-Aware Minimization, following
  Foret et al., ICLR 2021.
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language
  Supervision", ICML 2021. Via `open_clip_torch`.
