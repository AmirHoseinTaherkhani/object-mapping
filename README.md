# ObjectMapping — Overhead Surveillance Counter

A computer vision pipeline for detecting, tracking, and counting vehicles and pedestrians in fixed overhead surveillance video. Built on a fine-tuned YOLOv8s model with ByteTrack and a ghost-zone deduplication layer to eliminate double-counts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

| Component | Detail |
|---|---|
| Detector | YOLOv8s fine-tuned on this camera domain — 2 classes: `person` (0), `car` (1) |
| Weights | `models/weights/best_v3_merged.pt` (PyTorch) / `best_v3_merged.mlpackage` (CoreML) |
| Tracker | ByteTrack (BoxMOT) — pure IoU + Kalman, no appearance model |
| Training platform | UNH Premise HPC — NVIDIA A100 80 GB via SLURM |
| Input | 1920 × 1080 @ 60 fps fixed overhead intersection camera |

### Model performance

Fine-tuned on a merged dataset (~14 000 images), 50 epochs on an A100:

| Class | mAP@0.5 |
|---|---|
| Person | 0.985 |
| Car | 0.619 |
| **Overall** | **0.802** |

Tracking stability (30-second clip, `best_v3_merged.pt` + ByteTrack):

| Metric | Baseline | Fine-tuned |
|---|---|---|
| ID switches | 30 | 3 |
| Unique track IDs | 37 | 5 |
| Detections / frame | 6.0 | 2.5 |

---

## Repository Structure

```
├── counting_experiment/        # Reference baseline (PyTorch, every frame)
│   ├── count_objects.py        # Main counting script
│   ├── select_roi.py           # Interactive ROI polygon selector
│   ├── check_counts.py         # Frame-by-frame count verification replay
│   └── README.md
│
├── realtime_inference/         # Optimised pipelines for demo / live use
│   ├── run_coreml.py           # Apple Neural Engine — primary demo pipeline
│   ├── run_tensorrt.py         # TensorRT FP16 — NVIDIA GPU path
│   ├── export_coreml.py        # Export .pt → CoreML .mlpackage
│   ├── export_tensorrt.py      # Export .pt → TensorRT .engine
│   ├── export_tensorrt.slurm   # Build engine on Premise A100
│   └── README.md
│
├── tools/                      # Data preparation, training, and evaluation scripts
│   ├── finetune_on_camera.py   # Fine-tuning entry point (Mac + HPC)
│   ├── merge_datasets.py       # Merge and remap multiple source datasets
│   ├── prepare_dataset.py      # Split merged dataset into train/val/test
│   ├── threshold_grid_search.py # Grid search over per-class conf thresholds
│   ├── test_bytetrack.py       # ID-switch evaluation on a video clip
│   ├── extract_labeling_frames.py  # Sample frames for manual labeling
│   ├── autolabel_with_dino.py  # Auto-label with Grounding DINO
│   ├── compare_models.py       # Side-by-side model performance comparison
│   ├── compare_improvements.py # Chart improvements across training runs
│   ├── sahi_inference.py       # SAHI sliced inference for small objects
│   ├── make_comparison_video.py # Generate comparison videos
│   └── make_visdrone_video.py  # VisDrone dataset visualisation
│
├── hpc/                        # SLURM job scripts for UNH Premise A100
│   ├── train.slurm             # Fine-tune (12 h, 1× A100)
│   ├── inference.slurm         # Full-video inference (1 h)
│   └── grid_search.slurm       # Threshold grid search (2 h)
│
├── src/                        # Webapp and coordinate-mapping library
│   ├── object_detection/       # Python package (inference, tracking, mapping, API)
│   ├── webapp/                 # Streamlit frontend
│   ├── scripts/                # Entry points (run_webapp.py, run_realtime_mapping.py)
│   └── demo.ipynb              # Interactive walkthrough notebook
│
├── configs/                    # YAML configuration files
│   ├── training/               # Dataset configs for YOLO training
│   ├── model/                  # Detection model config
│   ├── ui/                     # Streamlit UI config
│   └── visualization/          # Real-time canvas config
│
├── experiments/
│   └── train_results/          # Training curves and confusion matrices
│
├── models/
│   ├── weights/                # Model weights — DVC-managed, not committed
│   └── exports/                # Exported engines / packages
│
├── requirements.txt            # Minimal local-dev dependencies
├── requirements/
│   ├── base.txt                # Core ML dependencies
│   ├── prod.txt                # + FastAPI and Streamlit
│   ├── docker.txt              # Pinned, headless OpenCV (for Docker image)
│   └── test.txt                # + pytest
├── Dockerfile                  # Container for the Streamlit webapp
├── docker-compose.yml
└── Data.dvc                    # DVC pointer to training data
```

---

## Quick Start

### 1 — Install dependencies

```bash
conda create -n objectmapping python=3.9
conda activate objectmapping
pip install -r requirements.txt
```

Pull model weights via DVC:

```bash
dvc pull
```

### 2 — Define the counting zone

```bash
cd counting_experiment
python select_roi.py       # click polygon on first frame, press Enter to save
```

### 3 — Run the counter (reference baseline)

```bash
python counting_experiment/count_objects.py
python counting_experiment/count_objects.py --no-display   # headless
```

### 4 — Run the real-time demo (Apple Silicon)

```bash
# Export once
python realtime_inference/export_coreml.py

# Run
python realtime_inference/run_coreml.py
python realtime_inference/run_coreml.py --no-display
```

See [realtime_inference/README.md](realtime_inference/README.md) for the full argument reference, NVIDIA/TensorRT instructions, and frame-skip guide.

---

## Counting Pipeline

Full documentation: [counting_experiment/README.md](counting_experiment/README.md)

| Argument | Default | Description |
|---|---|---|
| `--conf-car` | `0.50` | YOLO confidence threshold for cars |
| `--conf-person` | `0.45` | YOLO confidence threshold for persons |
| `--no-display` | off | Skip `cv2.imshow`, still save output video |

Output video: `counting_experiment/output_counted.mp4`

---

## Real-time Inference

Two hardware-accelerated paths:

| Scenario | Script | Expected FPS |
|---|---|---|
| Mac (Apple Silicon) | `realtime_inference/run_coreml.py` | 30–45 fps |
| NVIDIA GPU | `realtime_inference/run_tensorrt.py` | 60+ fps |

Both apply hardware-accelerated model format, async capture, and frame skipping. Full docs: [realtime_inference/README.md](realtime_inference/README.md)

---

## Training

### Fine-tune locally

```bash
python tools/finetune_on_camera.py
python tools/finetune_on_camera.py --base best_v3_merged.pt
python tools/finetune_on_camera.py --source original    # original labelled frames only
```

| Argument | Default | Description |
|---|---|---|
| `--source` | `merged` | Dataset pool (`merged` or `original`) |
| `--base` | `last_epoch3.pt` | Starting weights (filename in `models/weights/`) |
| `--hpc` | off | CUDA/HPC mode: `batch=32`, `workers=4`, `amp=True` |

### Fine-tune on HPC (UNH Premise)

```bash
sbatch hpc/train.slurm       # 12 h, 1× A100 80 GB
```

Logs: `logs/train_<job_id>.out`

---

## Dataset Preparation

```bash
# Merge multiple source datasets (remaps all classes to person=0, car=1)
python tools/merge_datasets.py

# Split into train/val/test
python tools/prepare_dataset.py --source merged     # recommended
python tools/prepare_dataset.py --source original   # original frames only
```

---

## Evaluation

### ID-switch test

```bash
python tools/test_bytetrack.py models/weights/best_v3_merged.pt
python tools/test_bytetrack.py models/weights/best_v3_merged.pt --full
```

### Threshold grid search

```bash
python tools/threshold_grid_search.py             # local (30-second clip)
sbatch hpc/grid_search.slurm                      # HPC (2 h)
```

Output: `outputs/threshold_search.csv`

### Full-video inference on HPC

```bash
sbatch hpc/inference.slurm
```

---

## Webapp

A Streamlit interface for ground-truth annotation and real-time coordinate mapping:

```bash
pip install -r requirements/prod.txt

# Run directly
python src/scripts/run_webapp.py

# Or via Docker
docker-compose up
```

The webapp provides:
- Interactive point selection for homography calibration
- Side-by-side video + 2D map visualisation with object trails
- CSV export of trajectories

---

## Syncing to HPC

```bash
rsync -av --progress \
  --exclude='newDatasets/' --exclude='outputs/' --exclude='*.mp4' \
  ./ at1293@premise.sr.unh.edu:~/objmap/
```

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — ByteTrack implementation
- UNH Premise HPC — A100 compute
