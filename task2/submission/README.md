# Project Activity 2 — Baseline CNN (VMMRdb, 50 make-model classes)

Team: 20230906 Heeseon Yoo, 20230613 Donghyun Shin, 20230673 Seojin Han, 20230497 Hyeonil Cho
Repository: https://github.com/heeseon28/intro_ai_project1 (code in `task2/`)

## Contents

| Path | Description |
|---|---|
| `Activity2_Report.pdf` | Baseline report (2 pages) |
| `task2/train_baseline.py` | Training and evaluation script for the baseline CNN |
| `task2/run_task2.sh` | Wrapper that runs the script from the project root |
| `task2/requirements.txt` | Python dependencies |
| `splits/` | Fixed train / validation / test split (`train.csv`, `val.csv`, `test.csv`, `class_info.csv`) |
| `result_task2/baseline_seed0_DonghyunShin/` | Run A (GPU), the run reported in the PDF |
| `result_task2/baseline_seed0_Hyeonil_Cho/` | Run B (CPU), an independent re-run with the same code, split and seed |
| `task2/report/` | LaTeX source of the report and `make_figures.py`, which rebuilds every figure and quoted number from the result folders |

## Baseline configuration

ResNet-18 (torchvision, `weights=None`), RGB 224×224 (fixed resize), normalization mean = std = 0.5,
Adam (lr 3e-4, no weight decay, no schedule), batch 64, 12 epochs, unweighted cross-entropy, seed 0.
No augmentation, transfer learning, re-weighting or special loss. The checkpoint with the highest validation
macro-F1 is reported. The test split is never loaded.

## Setup

1. Python 3.10+ with PyTorch. Install the remaining packages with:
   ```bash
   python3 -m pip install -r task2/requirements.txt
   ```
   Use your existing CUDA build of PyTorch if you have one.
2. Download VMMRdb from https://github.com/faezetta/VMMRdb and extract it. The data root is the folder that directly
   contains the class folders (for example `ford_mustang_1964/`). Images are not included in this archive.

## Run

Run all commands from this folder (the project root).

```bash
# quick check on a few batches (results go to result_task2/baseline_seed0_smoke/)
bash task2/run_task2.sh --data-root /path/to/VMMRdb --device cuda --smoke

# full baseline (about 40 min on one GPU slice; about 21 h on CPU)
bash task2/run_task2.sh --data-root /path/to/VMMRdb --device cuda --run-name baseline_seed0

# continue an interrupted run from its last completed epoch
bash task2/run_task2.sh --data-root /path/to/VMMRdb --device cuda --run-name baseline_seed0 --resume
```

On Windows, call the script directly: `python task2/train_baseline.py --data-root C:\path\to\VMMRdb --device cuda`.
`--device` accepts `auto`, `cuda`, `mps` or `cpu`. Lower `--workers` (for example `--workers 0`) if the data loader fails.
An existing run folder is never overwritten; use `--resume` or a new `--run-name`.

## Outputs (`result_task2/<run-name>/`)

| File | Content |
|---|---|
| `config.json` | Full training configuration, SHA-256 of the split files and of the script |
| `history.csv`, `learning_curves.png` | Per-epoch train/validation loss, accuracy and macro-F1 |
| `metrics.json` | Train and validation metrics of the selected checkpoint |
| `confusion_matrix_counts.csv`, `confusion_matrix.png` | Validation confusion matrix |
| `per_class_metrics.csv`, `difficult_classes.csv` | Per-class precision, recall and F1 |
| `validation_predictions.csv`, `validation_examples.png` | Per-image predictions and examples |
| `environment.json`, `run.log`, `status.json` | Environment, log and completion state |

Model checkpoints are kept locally in `checkpoints/` and are not included.

## Reported results (validation set)

| Run | Hardware | Best epoch | Accuracy | Macro-F1 | Train accuracy |
|---|---|---:|---:|---:|---:|
| A | GPU | 12 | 77.0% | 73.5% | 96.4% |
| B | CPU | 10 | 77.1% | 73.2% | 95.5% |

Run B's recorded split and source hashes differ from run A's only because Windows converted the line endings
on checkout. The file contents and the validation images are identical.

## Rebuilding the report

```bash
python3 task2/report/make_figures.py /path/to/VMMRdb   # figures and quoted numbers (the image path is only needed for Fig. 3)
cd task2/report && latexmk -pdf main.tex
```
