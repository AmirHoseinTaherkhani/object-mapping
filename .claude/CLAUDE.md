# ObjectMapping — Project Context

## What this project does

Fixed-camera overhead surveillance pipeline that **detects, tracks, and counts** people and cars
passing through a camera's field of view. The camera is stationary and mounted overhead
(bird's-eye perspective). Two classes only: `person=0`, `car=1`.

The core challenge is **deduplication**: the same physical object must be counted exactly once
even when a tracker assigns it multiple track IDs due to occlusion, confidence jitter, or
re-entry into the frame.

---

## Repo layout

```
ObjectMapping/
├── realtime_inference/
│   ├── run_coreml.py          ← PRIMARY demo pipeline (Apple Neural Engine)
│   └── run_tensorrt.py        ← NVIDIA GPU path
├── counting_experiment/
│   └── count_objects.py       ← Reference baseline (PyTorch, every frame)
├── tools/                     ← Experiment & dataset utilities (see below)
├── experiments/
│   └── yolo11_experiments/
│       ├── finetuned/         ← Fine-tuned weights from HPC (gitignored, large)
│       │   └── yolo11{s,m,l,x}/train/weights/best.pt
│       ├── tracking_inference/ ← Output videos from tracking batch (gitignored)
│       │   ├── pretrained_yolo11s/   ← no-reid tracking videos, all Demo/ videos
│       │   ├── finetuned_yolo11s/    ← (one folder per model)
│       │   └── reid/                 ← with OSNet re-ID, ANMR0006.mp4 only
│       ├── videos/            ← Earlier tracking outputs (gitignored)
│       ├── results/           ← Per-model JSON benchmark counts ✓ in git
│       ├── summary.csv        ← All results in one table ✓ in git
│       └── summary.md         ← Human-readable results ✓ in git
├── hpc/
│   └── finetune_yolo11{s,m,l,x}.slurm  ← SLURM jobs for UNH Premise A100
├── models/weights/
│   ├── best_v3_merged.pt      ← Production YOLOv8s weights (DVC, S3 broken)
│   ├── best_v3_merged.mlpackage ← CoreML export (local only)
│   └── yolo11{s,m}.pt         ← Pretrained COCO weights (downloaded)
├── labeling_data/trainingData/ ← Full merged dataset (gitignored, large)
│   ├── train/  (~32k images)
│   ├── valid/  (~7k images)
│   └── test/   (~3.5k images)
├── Demo/                      ← Videos (gitignored, large)
│   ├── ANMR0006.mp4           ← User-recorded surveillance video (31k frames, 60fps)
│   ├── benchmark_stmarc.avi   ← Benchmark video #1
│   ├── benchmark_traffic.mp4  ← Benchmark video #2
│   └── videos/                ← 10 public benchmark videos (Pexels)
└── docs/
    ├── realtime_pipeline_engineering.md
    ├── model_limitations.md
    ├── slides_engineering.pptx
    └── slides_limitations.pptx
```

---

## Models

### Production model (CoreML pipeline)
- **File**: `models/weights/best_v3_merged.pt` / `best_v3_merged.mlpackage`
- **Architecture**: YOLOv8s, 2-class (person, car)
- **Training data**: ~14k images, UNH Premise A100
- **Critical**: CoreML quantization shifts car confidence to 0.15–0.37 (vs 0.45+ in PyTorch).
  Every threshold and ghost-zone constant in `run_coreml.py` was tuned for this range.

### YOLO11 experiment models
Fine-tuned on a merged 32k-image dataset (see Dataset section) on UNH Premise A100.
Weights live at `experiments/yolo11_experiments/finetuned/<model>/train/weights/best.pt`.

| Model     | mAP50  | mAP50-95 | Precision | Recall | Stopped at epoch |
|-----------|--------|----------|-----------|--------|-----------------|
| yolo11s   | 0.631  | 0.367    | 0.798     | 0.561  | ~100            |
| yolo11m   | 0.671  | 0.399    | 0.816     | 0.606  | ~110            |
| yolo11l   | 0.677  | 0.407    | 0.816     | 0.612  | 144             |
| yolo11x   | 0.689  | 0.416    | 0.818     | 0.626  | ~125            |

All four hit early stopping (patience=30) — more epochs will not help. The moderate mAP is
expected: the test set includes CrowdHuman (dense crowds) and WiderPerson (heavy occlusion)
which are genuinely hard, and the goal is production counting accuracy on the actual camera,
not benchmark mAP.

---

## Dataset

Training data is a merge of 7 sources into `labeling_data/trainingData/`:

| Source | Images (approx) | Classes used |
|---|---|---|
| Original newDatasets (this camera) | ~14k | person, car |
| VisDrone | 5,000 | person→person, all vehicles→car |
| WiderPerson | 5,000 | all→person |
| UA-DETRAC-10K | 5,000 | all→car |
| CrowdHuman | 2,101 | person→person (head skipped) |
| COWC | 665 | Car→car |
| COCO | 2,350 | person(0)→person, car/motorcycle/bus/truck→car |

Class maps are hard-coded in `tools/merge_roboflow_zips.py` and `tools/download_public_datasets.py`.

---

## Three pipeline implementations

### 1. `realtime_inference/run_coreml.py` — Primary demo pipeline

Apple Neural Engine (ANE) inference via CoreML `.mlpackage`. Designed for live demo on Mac.

**Key constants:**
```python
MIN_TRACK_AGE = {CLASS_CAR: 5,  CLASS_PERSON: 20}
GHOST_RADIUS  = {CLASS_CAR: 80, CLASS_PERSON: 80}
GHOST_TIMEOUT = {CLASS_CAR: 90, CLASS_PERSON: 120}
COREML_CONF_FLOOR = 0.10
```

**Ghost zone system** (deduplication):
- When a track dies, a ghost zone is placed at its last centroid. New tracks that start inside
  an active ghost zone are suppressed (not counted).
- **Size-adaptive car ghost**: `radius = max(80, bbox_width * 0.6)`. Close cars have large
  bounding boxes and large centroid jumps → need wider radius. But also shorter timeout
  (`max(45, 90 * 100/bbox_width)`) because fragments appear quickly.
- **Person ghost uses first centroid**: With `MIN_TRACK_AGE=20`, a person drifts ~80px before
  the ghost check fires. Using the *first* centroid (where the track started) rather than the
  current centroid gives a reliable hit against the ghost zone center.
- **Person DIED-YOUNG → no ghost**: Short fragments at crosswalk entries were blocking
  legitimate next persons. Suppressed by not creating a ghost for person tracks that die before
  `MIN_TRACK_AGE=20`.
- **Car DIED-YOUNG → ghost IS created**: Needed to catch confidence-dip re-identifications of
  the same physical car.
- **Live-track proximity check** (arm-opening suppression): When T1 is alive and T2 appears
  nearby, look up T1's centroid from `age_of_T2` frames ago. If T1 was within `GHOST_RADIUS`
  of T2's starting position, suppress T2. Handles the case where opening arms creates a new
  track while the original is still alive (so no ghost zone exists yet).

**Frame-skip rule**: Always pass `last_dets` (previous detections) to ByteTrack on skip frames.
Never pass `np.empty` — that puts all tracks into ByteTrack's "lost" state, stalling
`track_age` so nothing ever reaches `MIN_TRACK_AGE`. This was a critical early bug.

**Tracker init:**
```python
ByteTrack(frame_rate=int(cap.fps), track_buffer=600, track_thresh=0.15)
```
`track_thresh=0.15` is mandatory — CoreML car confidences start at 0.15 and the default
ByteTrack threshold would refuse to create tracks for most car detections.

**Known remaining edge cases:**
- Close car occasionally double-counted: centroid jump > 180px after fragmentation. 0.6× radius
  is the current balance; 0.8× caused adjacent-lane false suppression.
- Person double-counted on ROI re-entry: person exits ROI, ghost placed at exit, re-enters at
  different position outside ghost radius. Requires ROI-boundary-aware ghost zones to fix.
- Stopped car: track may die and restart after ghost expires (~0.75s for close cars). Attempted
  fix was reverted (commit ~53d9119) — all approaches overshot and blocked new cars.

**Diagnose counting decisions:**
```bash
python realtime_inference/run_coreml.py --verbose --no-display 2>/dev/null | \
  grep -E "COUNTED|GHOST-SUPP|LIVE-SUPP|DIED-YOUNG|GHOST-ZONE"
```

### 2. `counting_experiment/count_objects.py` — Reference baseline

PyTorch inference, every frame, flat constants (`MIN_TRACK_AGE=30`, `GHOST_TIMEOUT=300`).
Use this as ground truth when diagnosing bugs in the faster pipelines. Its counts are the
most reliable because stable PyTorch confidence avoids fragmentation and no frames are skipped.

### 3. `realtime_inference/run_tensorrt.py` — NVIDIA GPU path

Similar to run_coreml.py but for NVIDIA GPUs. Simpler ghost zone (no per-class tuning) because
PyTorch confidence range is more stable than CoreML's.

---

## Experiment pipeline (YOLO11 + BoT-SORT)

A parallel research track comparing YOLO11 model sizes against the production YOLOv8s.

### Tools

| Script | Purpose |
|---|---|
| `tools/run_yolo11_experiment.py` | Single-video counting with YOLO11 + BoT-SORT + ghost zones. `--pretrained` flag remaps COCO classes on-the-fly. |
| `tools/run_tracking_inference.py` | **Batch tracking visualization** — no counting, no ghost zones. `--batch` runs all 8 models × all Demo/ videos. `--reid` enables OSNet x0.25 appearance embedding, runs only on ANMR0006.mp4, outputs to `tracking_inference/reid/`. |
| `tools/evaluate_experiments.py` | Orchestrates all 8 experiments: mAP eval + benchmark counting. Writes `results/*.json`, `summary.csv`, `summary.md`. |
| `tools/download_public_datasets.py` | Downloads VisDrone and COCO; merges into trainingData. |
| `tools/merge_roboflow_zips.py` | Extracts and merges manually-downloaded Roboflow zips (CrowdHuman, WiderPerson, UA-DETRAC-10K, COWC). |
| `tools/merge_test_sets.py` | Consolidates test splits across all datasets. |
| `tools/make_slides.py` | Generates engineering and limitations slide decks. |

### Tracker: BoT-SORT (boxmot 18.0.0)

```python
BotSort(
    with_reid=False,          # set True for reid variant
    reid_model=reid.model,    # pass ReID(...).model — NOT the ReID wrapper
    track_high_thresh=0.25,
    track_low_thresh=0.10,
    new_track_thresh=0.25,
    track_buffer=600,
    frame_rate=int(fps),
    cmc_method="sof",         # sparse optical flow for camera motion compensation
)
```

**COCO → 2-class remapping** (pretrained mode):
```python
COCO_TO_OURS = {0: person, 2: car, 3: car, 5: car, 7: car}
# person(0), car(2), motorcycle(3), bus(5), truck(7)
```

**ReID**: OSNet x0.25 MSMT17 — pre-installed with boxmot at
`/opt/anaconda3/envs/objectmapping/lib/python3.9/site-packages/models/osnet_x0_25_msmt17.pt`.
Pass `ReID(weights='osnet_x0_25_msmt17.pt', device='cpu').model` to BotSort (the inner backend,
not the ReID wrapper — BotSort calls `.get_features()` directly).

### HPC training (UNH Premise A100)

```bash
# Sync code to HPC (from repo root)
rsync -av --exclude='newDatasets/' --exclude='outputs/' --exclude='*.mp4' \
  ./ at1293@premise.sr.unh.edu:~/objmap/

# Submit jobs
cd ~/objmap && sbatch hpc/finetune_yolo11s.slurm
sbatch hpc/finetune_yolo11m.slurm
sbatch hpc/finetune_yolo11l.slurm
sbatch hpc/finetune_yolo11x.slurm

# Monitor
squeue -u at1293
tail -f logs/yolo11s_<jobid>.out

# Sync results back
rsync -av at1293@premise.sr.unh.edu:~/objmap/experiments/yolo11_experiments/finetuned/ \
  experiments/yolo11_experiments/finetuned/
```

Python on HPC login node: `~/.conda/envs/objmap/bin/python` (conda not in PATH by default).
Always delete label caches before training — `/mnt/home` is a symlink to `/mnt/gpfs01/home`
and ultralytics caches the hash under the gpfs01 path, causing mismatch errors.

---

## Current state (as of 2026-07-07)

### Done
- Production pipeline: `run_coreml.py` is working and tuned. Known edge cases documented above.
- Dataset: 32k train / 7k valid / 3.5k test images merged from 7 sources.
- YOLO11 training: all 4 model sizes (s/m/l/x) fine-tuned on HPC, weights synced locally.
- Experiment evaluation: mAP scores and benchmark video counts recorded in `summary.md`.
- Tracking inference batch: `run_tracking_inference.py --batch` is running locally in terminal,
  processing all 8 models × 13 videos (no-reid). Output: `tracking_inference/<model>/*.mp4`.
  ETA ~7 hours from start. Skip-existing logic means it can be interrupted and resumed.
- ReID comparison: `run_tracking_inference.py --batch --reid` ready to run on ANMR0006.mp4
  for all 8 models. Not yet started — run after the no-reid batch completes (or in parallel,
  they write to different folders).

### Pending / next logical steps
- **Counting tuning for YOLO11**: The ghost-zone and MIN_TRACK_AGE constants in
  `run_yolo11_experiment.py` are copied from run_coreml.py and not tuned for PyTorch
  confidence ranges. ANMR0006 counts 0 cars / 1 person across all models — clearly wrong.
  Needs the same tuning process that was done for run_coreml.py.
- **Compare tracking quality**: Review the `tracking_inference/` output videos side-by-side
  to see which model + reid configuration produces the most stable tracks on ANMR0006.mp4.
- **Web app integration**: Branch is named `dev-webapp-integration` — a web front-end is
  planned but not yet started.
- **DVC remote**: The S3 bucket (`s3://carperson-model-yolov8/dvc-cache`) no longer exists.
  If model weights need to be shared via DVC, a new remote must be configured.

---

## Environment

```bash
conda activate objectmapping   # Python 3.9
# Key packages: ultralytics==8.4.55, boxmot==18.0.0, opencv-python-headless, numpy
```

**DVC note**: `dvc pull` is broken — the S3 bucket was deleted. Fine-tuned YOLO11 weights
live at `experiments/yolo11_experiments/finetuned/` (gitignored). Original production weights
(`best_v3_merged.pt`) are only on local disk and the HPC.

**MPS**: Apple GPU (Metal) is available on this machine. Use `--device mps` for YOLO11
inference. The reid model (OSNet) should use `--reid-device cpu` — it's tiny and CPU is fine.

---

## Branch and repo

- **Active branch**: `dev-webapp-integration`
- **GitHub**: `git@github.com:AmirHoseinTaherkhani/object-mapping.git`
- **Main branch**: `main` (stable, behind dev-webapp-integration by 14+ commits)
- Large files gitignored: `Demo/`, `labeling_data/`, `experiments/yolo11_experiments/finetuned/`,
  `experiments/yolo11_experiments/tracking_inference/`, `models/weights/*.pt`
