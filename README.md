# Object Mapping — Overhead Surveillance Tracking & Counting

A computer vision pipeline for detecting, tracking, and counting vehicles and pedestrians in fixed overhead surveillance video, built on YOLOv8s fine-tuned for this camera domain.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

| Component | Detail |
|---|---|
| Detector | YOLOv8s — 2 classes: `person` (0), `car` (1) |
| Model weights | `models/weights/best_v3_merged.pt` |
| Tracker | ByteTrack (BoxMOT) |
| Training platform | UNH Premise HPC — NVIDIA A100 80 GB via SLURM |
| Input video | 1920×1080 @ 60 fps overhead intersection camera |

---

## Repository Structure

```
├── counting_experiment/        # Vehicle & person counting pipeline
│   ├── select_roi.py           # Interactive ROI polygon selector
│   ├── count_objects.py        # Main counting script
│   ├── check_counts.py         # Output verification replay
│   └── README.md               # Detailed docs for the counting pipeline
│
├── src/object_detection/
│   └── training/               # Training script variants
│
├── finetune_on_camera.py       # Fine-tuning entry point (Mac + HPC)
├── prepare_dataset.py          # Merge & balance multi-source datasets
├── merge_datasets.py           # Dataset builder with class remapping
├── test_old_bytetrack.py       # ID-switch evaluation on a video clip
├── threshold_grid_search.py    # Grid search over per-class conf thresholds
│
├── train_hpc.slurm             # SLURM job: fine-tuning on A100
├── inference_hpc.slurm         # SLURM job: full-video inference
└── grid_search_hpc.slurm       # SLURM job: threshold grid search
```

---

## Setup

```bash
conda create -n objectmapping python=3.9
conda activate objectmapping
pip install ultralytics boxmot opencv-python-headless numpy
```

Pull model weights via DVC:
```bash
dvc pull
```

---

## Counting Pipeline

The main product of this project. Counts moving vehicles and pedestrians in a video with ROI masking, parked-car filtering, and ghost-zone deduplication.

```bash
cd counting_experiment

# 1. Define the zone of interest (click polygon → Enter to save)
python select_roi.py

# 2. Run the counter
python count_objects.py

# 3. Verify output (optional)
python check_counts.py
```

See [counting_experiment/README.md](counting_experiment/README.md) for full documentation, configuration options, and the engineering decisions behind each fix.

---

## Model Training

### Fine-tuning (local Mac)

```bash
python finetune_on_camera.py
```

### Fine-tuning on UNH Premise HPC (A100)

```bash
# SSH into Premise, then:
sbatch train_hpc.slurm
squeue -u $USER        # monitor
```

### Dataset preparation

```bash
python prepare_dataset.py   # merge + remap + balance datasets
```

The training dataset merges 5 vehicle datasets and 2 pedestrian datasets with your labelled camera frames. Car:person ratio is balanced to ~2:1 before training.

---

## Evaluation

### ID-switch test (30-second clip)

```bash
python test_old_bytetrack.py models/weights/best_v3_merged.pt
```

### Full-video inference

```bash
python test_old_bytetrack.py models/weights/best_v3_merged.pt --full \
  --conf-car 0.50 --conf-person 0.45
```

### Threshold grid search (HPC)

```bash
sbatch grid_search_hpc.slurm
# results → outputs/threshold_search.csv
```

---

## Model Performance

Fine-tuned on merged dataset (~14,000 images) — 50 epochs on A100:

| Class | mAP@0.5 |
|---|---|
| Person | 0.985 |
| Car | 0.619 |
| **Overall** | **0.802** |

Car mAP is limited by the overhead viewing angle (domain gap from street-level training data) rather than data quantity. Person detection is near-perfect.

**Tracking stability** (30-second clip, `best_v3_merged.pt`):

| Metric | Old model | Fine-tuned |
|---|---|---|
| ID switches | 30 | 3 |
| Unique track IDs | 37 | 5 |
| Detections/frame | 6.0 | 2.5 |

The 3 remaining switches are genuine occlusion events (cars behind trees).

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — ByteTrack implementation
- UNH Premise HPC — A100 compute
