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
| Input video | 1920×1080 @ 60 fps fixed overhead intersection camera |

---

## Repository Structure

```
├── counting_experiment/        # Vehicle & person counting pipeline
│   ├── select_roi.py           # Interactive ROI polygon selector
│   ├── count_objects.py        # Main counting script
│   ├── check_counts.py         # Output verification replay
│   └── README.md               # Detailed docs and problem-solving notes
│
├── src/object_detection/
│   └── training/               # Training script variants
│
├── finetune_on_camera.py       # Fine-tuning entry point (Mac + HPC)
├── prepare_dataset.py          # Split merged dataset into train/val/test
├── merge_datasets.py           # Merge & remap multiple source datasets
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

## 1 — Dataset Preparation

### Merge source datasets

Combines 5 vehicle datasets and 2 pedestrian datasets with your labelled camera frames. Remaps all class IDs to `person=0, car=1`, caps large datasets at 2,000 images to prevent class imbalance, and oversamples pedestrian data to reach a ~2:1 car:person ratio.

```bash
python merge_datasets.py
```

Output: `labeling_data/merged/` (~6,600 images, 2.2:1 car:person ratio)

---

### Prepare train / val / test split

Takes the merged (or original) image pool and splits it into `train/`, `val/`, and `test/` folders with the `data.yaml` config file YOLO expects.

```bash
# Use the merged multi-source dataset (recommended)
python prepare_dataset.py --source merged

# Use only the original hand-labelled camera frames
python prepare_dataset.py --source original
```

| Argument | Values | Default | Description |
|---|---|---|---|
| `--source` | `merged`, `original` | `merged` | Which image pool to split |

Output: `labeling_data/data.yaml` + `train/`, `val/`, `test/` subfolders

---

## 2 — Training

### Fine-tune locally (Mac / CPU / MPS)

```bash
python finetune_on_camera.py
```

Resumes from `models/weights/last_epoch3.pt` by default, trains for 50 epochs on the merged dataset.

### Fine-tune with a different base model

```bash
python finetune_on_camera.py --base best_v3_merged.pt
```

### Use the original labelled frames only

```bash
python finetune_on_camera.py --source original
```

| Argument | Values | Default | Description |
|---|---|---|---|
| `--source` | `merged`, `original` | `merged` | Dataset pool to train on |
| `--base` | any `.pt` filename in `models/weights/` | `last_epoch3.pt` | Starting weights |
| `--hpc` | flag | off | CUDA/HPC mode: `batch=32`, `workers=4`, `amp=True` |

Checkpoints are saved to `runs/detect/yolov8s_merged_v2/weights/` after every epoch.

---

### Fine-tune on HPC (UNH Premise — A100)

```bash
# From login node, after syncing files:
sbatch train_hpc.slurm
```

Resources allocated: 1× A100 80 GB, 4 CPUs, 32 GB RAM, 12-hour wall time.
Logs: `logs/train_<job_id>.out` / `.err`

Monitor the job:
```bash
squeue -u $USER                  # show running jobs
tail -f logs/train_<job_id>.out  # stream live output
scancel <job_id>                  # cancel if needed
```

---

## 3 — Evaluation

### ID-switch test (30-second clip)

Runs the model + ByteTrack on the first 30 seconds of the video and counts track ID switches as a stability metric. Lower = better.

```bash
python test_old_bytetrack.py models/weights/best_v3_merged.pt
```

### ID-switch test on the full video

```bash
python test_old_bytetrack.py models/weights/best_v3_merged.pt --full
```

### Adjust per-class confidence thresholds

```bash
python test_old_bytetrack.py models/weights/best_v3_merged.pt \
  --conf-car 0.50 --conf-person 0.45
```

| Argument | Default | Description |
|---|---|---|
| `weights` | *(positional)* | Path to `.pt` weights file |
| `--full` | off | Run entire video instead of 30-second clip |
| `--conf-car` | `0.40` | Confidence threshold for car class |
| `--conf-person` | `0.50` | Confidence threshold for person class |

Output video saved to `outputs/model_comparison/<weights_stem>_car<cc>_p<cp>_<30s|full>.mp4`

---

### Run full-video inference on HPC

```bash
sbatch inference_hpc.slurm
```

Resources: 1× A100, 2 CPUs, 16 GB RAM, 1-hour wall time.
Runs `best_v3_merged.pt` with `conf-car=0.50`, `conf-person=0.50` on the full video.
Logs: `logs/inference_<job_id>.out`

---

## 4 — Threshold Tuning

### Grid search (local)

Tests every combination of 7 car thresholds × 5 person thresholds on a 30-second clip and scores each by tracking stability. Saves a ranked CSV.

```bash
python threshold_grid_search.py
```

### Grid search on a longer clip

```bash
python threshold_grid_search.py --video-seconds 60
```

| Argument | Default | Description |
|---|---|---|
| `--video-seconds` | `30` | Length of clip used per trial |

Output: `outputs/threshold_search.csv` — columns: `conf_car`, `conf_person`, `total_unique_ids`, `id_switches`, `mean_dets_per_frame`, `persistent_ratio`, `score`

### Grid search on HPC

```bash
sbatch grid_search_hpc.slurm
```

Resources: 1× A100, 2 CPUs, 16 GB RAM, 2-hour wall time.
Logs: `logs/grid_<job_id>.out`

---

## 5 — Counting Pipeline

Full documentation: [counting_experiment/README.md](counting_experiment/README.md)

### Step 1 — Define the counting zone

Opens the first video frame. Click to place polygon points, press **Enter** to save, **Esc** to reset.

```bash
cd counting_experiment
python select_roi.py
```

Output: `counting_experiment/roi.json` — loaded automatically by `count_objects.py`

---

### Step 2 — Run the counter

```bash
python count_objects.py
```

Runs with live display. Press **Q** to stop early.

### Adjust thresholds

```bash
python count_objects.py --conf-car 0.50 --conf-person 0.45
```

### Run headlessly (no window — for HPC or background jobs)

```bash
python count_objects.py --no-display
```

| Argument | Default | Description |
|---|---|---|
| `--conf-car` | `0.50` | YOLO confidence threshold for cars |
| `--conf-person` | `0.45` | YOLO confidence threshold for persons |
| `--no-display` | off | Skip `cv2.imshow`, still save output video |

Output video: `counting_experiment/output_counted.mp4`
Final summary printed to terminal: total cars, total people, frames processed.

---

### Step 3 — Verify counts (optional)

Replays the output video and prints a timestamped log of every moment a count increments, so you can spot double-counts or missed entries.

```bash
python check_counts.py

# Or point at a specific video
python check_counts.py path/to/output_counted.mp4
```

---

## 6 — Syncing Files to HPC

```bash
# Push code + weights to Premise
bash scripts/sync_weights.sh

# Or manually with rsync
rsync -av --progress \
  --exclude='newDatasets/' --exclude='outputs/' --exclude='*.mp4' \
  ./ at1293@premise.sr.unh.edu:~/objmap/

# Pull results back to Mac
rsync -av --progress \
  at1293@premise.sr.unh.edu:~/objmap/outputs/model_comparison/ \
  ./outputs/model_comparison/
```

---

## Model Performance

Fine-tuned on merged dataset (~14,000 images), 50 epochs on A100:

| Class | mAP@0.5 |
|---|---|
| Person | 0.985 |
| Car | 0.619 |
| **Overall** | **0.802** |

**Tracking stability** (30-second clip, `best_v3_merged.pt` + ByteTrack):

| Metric | Baseline model | Fine-tuned |
|---|---|---|
| ID switches | 30 | 3 |
| Unique track IDs | 37 | 5 |
| Detections / frame | 6.0 | 2.5 |

The 3 remaining switches are genuine occlusion events (cars passing behind trees).

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — ByteTrack implementation
- UNH Premise HPC — A100 compute
