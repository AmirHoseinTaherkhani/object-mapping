# ObjectMapping — Claude Context

## Project overview

Fixed-camera overhead surveillance counting pipeline. YOLOv8s (2 classes: `person=0`, `car=1`) fine-tuned on this camera domain, tracked with ByteTrack, with a ghost-zone deduplication layer on top to prevent the same physical object being counted multiple times.

Three counting implementations exist:
- `counting_experiment/count_objects.py` — **reference baseline**. PyTorch, every-frame inference, known-good counts. Used as ground truth when diagnosing bugs in the faster pipelines.
- `realtime_inference/run_coreml.py` — **primary demo pipeline**. Apple Neural Engine (CoreML `.mlpackage`), frame-skipping, per-class ghost zones. Most of the debugging work in this project lives here.
- `realtime_inference/run_tensorrt.py` — NVIDIA GPU path. Simpler ghost zone (no per-class tuning needed, PyTorch confidence range is more stable).

---

## Model

- Weights: `models/weights/best_v3_merged.pt` (PyTorch), `best_v3_merged.mlpackage` (CoreML)
- Classes: `person=0`, `car=1`
- YOLOv8s, trained on merged dataset (~14k images) on UNH Premise A100
- **CoreML confidence distribution differs from PyTorch**: cars output 0.15–0.37 (vs 0.45+ for PyTorch). This affects every threshold and ghost-zone parameter in `run_coreml.py`.

---

## `run_coreml.py` architecture

### Key constants

```python
MIN_TRACK_AGE = {CLASS_CAR: 5,  CLASS_PERSON: 20}
GHOST_RADIUS  = {CLASS_CAR: 80, CLASS_PERSON: 80}
GHOST_TIMEOUT = {CLASS_CAR: 90, CLASS_PERSON: 120}
COREML_CONF_FLOOR = 0.10   # pre-NMS floor; per-class thresholds applied after
```

**Why per-class?** Cars fragment more (confidence jitter at CoreML's lower output range) → need a shorter `MIN_TRACK_AGE` to be counted before fragmenting. Persons with `MIN_TRACK_AGE=20` absorb arm-opening events that create track fragments lasting <20 frames.

### Size-adaptive car ghost zone

```python
def car_ghost_params(bw: float):
    radius  = max(80,  bw * 0.6)   # 300px car → 180px radius
    timeout = max(45, int(90 * 100.0 / max(bw, 100)))  # 300px car → 45 frames
```

Close cars (large pixel footprint) need wider radius because edge-clipping causes large centroid jumps when transitioning from partially to fully visible. But they need a *shorter* timeout because their fragments appear quickly — a long timeout would block new cars entering the same corridor.

### Person ghost check uses first centroid

With `MIN_TRACK_AGE=20`, a person walks ~80px before the ghost check fires. Using the *current* centroid at that point places the check near the edge of `GHOST_RADIUS=80`, making the comparison unreliable. Using the *first* (oldest) centroid in the track history correctly identifies that the track started on top of a ghost zone.

Cars use the current centroid — `MIN_TRACK_AGE=5` means minimal drift.

### Ghost zone creation rules

- **Car DIED-YOUNG** (died before `MIN_TRACK_AGE=5`): ghost zone IS created. Needed to prevent the same physical close-car from being re-identified after a brief confidence dip.
- **Person DIED-YOUNG** (died before `MIN_TRACK_AGE=20`): ghost zone NOT created. Short person fragments at crosswalk entries were blocking the next legitimate person starting at the same position.
- **Any counted track dies**: ghost zone created at last centroid.

### Arm-opening suppression (live-track proximity check)

When a person opens their arms, ByteTrack may create a new track ID (T2) while the original (T1) is still alive. Since T1 hasn't died, no ghost zone exists yet. The counting code runs a second check: look up T1's historical centroid from `track_age[T2]` frames ago and suppress T2 if T1 was within `GHOST_RADIUS` of T2's starting position.

```python
ago = min(age - 1, len(c) - 1)
hx, hy = c[-ago - 1]   # T1's position when T2 first appeared
```

### Frame-skip behavior

**Always pass `last_dets` to the tracker on skip frames — never `np.empty`.** Passing `np.empty` sends all tracks to ByteTrack's "lost" state every other frame. Lost tracks are not output, `track_age` stalls, and nothing reaches `MIN_TRACK_AGE`. This was a critical bug early in the project.

### ByteTrack init

```python
tracker = ByteTrack(frame_rate=int(cap.fps), track_buffer=600, track_thresh=0.15)
```

`track_thresh=0.15` is required because CoreML outputs car confidences in the 0.15–0.37 range. The default ByteTrack `track_thresh` is higher and would refuse to create new tracks for most car detections.

`track_buffer=600` = 10 seconds at 60 fps. Keeps tracks alive through long occlusions.

---

## Known remaining edge cases

- **One close car occasionally double-counted**: When a very large (close-camera) car fragments and the new track's centroid has jumped further than 180px from the ghost zone center, the ghost doesn't catch it. Widening the radius to 0.8× caused adjacent-lane false suppression, so 0.6× is the current balance.
- **One person double-counted when re-entering ROI**: Person walks out of the ROI boundary and back in. The original track dies, ghost is placed at exit. When the person re-enters in a different position (perspective shift), the ghost doesn't cover the new entry → counted again. Low-frequency edge case; fixing it would require ROI-boundary-aware ghost zones.
- **Stopped car at pedestrian crossing**: If a car stops mid-journey, the track may die and restart after the ghost zone expires (~0.75s for close cars). This was explicitly attempted and reverted (see commits around `53d9119`) because all fixes overshot and blocked new cars.

---

## Reference baseline (`count_objects.py`)

Uses flat constants (no per-class): `MIN_TRACK_AGE=30`, `GHOST_RADIUS=80`, `GHOST_TIMEOUT=300`. Works because:
- PyTorch confidence is stable (no CoreML jitter) → less fragmentation → flat age=30 is enough
- Every frame is processed (no skip) → `track_age` increments every frame, not every 2 frames
- `GHOST_TIMEOUT=300` is 5 seconds — safe because track fragmentation for pedestrians is rare with stable detections

When diagnosing a bug in `run_coreml.py`, compare its output against `count_objects.py` on the same video.

---

## Environment

```bash
conda activate objectmapping   # Python 3.9
# key packages: ultralytics==8.4.55, boxmot, opencv-python-headless, numpy
```

Model weights via DVC: `dvc pull`

---

## How to run and diagnose

```bash
# Normal run
python realtime_inference/run_coreml.py

# Headless
python realtime_inference/run_coreml.py --no-display

# Diagnose counting decisions (pipe through grep to filter noise)
python realtime_inference/run_coreml.py --verbose --no-display 2>/dev/null | \
  grep -E "COUNTED|GHOST-SUPP|LIVE-SUPP|DIED-YOUNG|GHOST-ZONE"
```

**Verbose output tags:**
- `COUNTED` — track reached MIN_TRACK_AGE, no ghost hit, counter incremented
- `GHOST-SUPP` — suppressed by a dead-track ghost zone
- `LIVE-SUPP` — suppressed by the live-track proximity check (arm-opening)
- `DIED-YOUNG` — track died before MIN_TRACK_AGE; person → no ghost, car → ghost created
- `GHOST-ZONE` — a counted track died and its ghost zone was registered (with position, radius, timeout)

---

## HPC (UNH Premise A100)

```bash
sbatch train_hpc.slurm          # fine-tune
sbatch inference_hpc.slurm      # full-video inference
sbatch grid_search_hpc.slurm    # threshold search
```

Sync: `rsync -av --exclude='newDatasets/' --exclude='outputs/' --exclude='*.mp4' ./ at1293@premise.sr.unh.edu:~/objmap/`

---

## Active branch

`dev-webapp-integration` — all counting pipeline work is on this branch. Main branch is `main`.
