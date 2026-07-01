# Model Limitations and Proposed Solutions

Current model: `best_v3_merged.pt` / `best_v3_merged.mlpackage`  
Pipeline: `realtime_inference/run_coreml.py`

> **Goal shift (updated):** The target is no longer a model fine-tuned for one
> specific camera. The goal is a **general overhead surveillance model** —
> strong enough to deploy on any fixed overhead camera with near-perfect
> tracking, without per-camera retraining.  This changes the priority and
> approach for most limitations below.

Each limitation includes its root cause, a solution aligned with the general-model
goal, and a difficulty rating for a student contributor.

---

## 1. Model Is Too Small for General Use

**What you will see:** High validation mAP on the training camera but immediate
accuracy drop on any other footage.  The model has learned one narrow
distribution and lacks the parameter capacity to hold a broader one.

**Why:** YOLOv8s has ~11 M parameters.  It can specialise well on one domain
but cannot simultaneously represent the variation in lighting, camera height,
vehicle type, and pedestrian density found across many cameras.

**What to do:** Upgrade the base model to **YOLOv8m** (~25 M parameters) for
all future training runs.  Change one line in `tools/finetune_on_camera.py`:

```python
model = YOLO("yolov8m.pt")   # was yolov8s.pt
```

YOLOv8m still runs within the real-time budget on the Apple Neural Engine.
Re-export to CoreML after training.

**Student difficulty:** Low (one-line change + one HPC training job).

---

## 2. Training Data Is Too Narrow (Root Cause of Most Other Limitations)

**What you will see:** Nearly every other limitation in this document — poor
person detection, car false positives, fragile tracking — is downstream of
this single problem.  The ~14 000-image dataset comes overwhelmingly from one
camera at one intersection.  The model has converged on that narrow
distribution.

**Why:** Camera-specific fine-tuning was the original design goal.  The new
goal is the opposite: a model that generalises across all overhead cameras
without retraining.

**What to do:** Replace the camera-specific dataset with a combination of
large, diverse, publicly available overhead datasets.  The student should
download, convert to YOLO format (Roboflow can do this automatically), and
merge using `tools/merge_datasets.py`:

| Dataset | Content | Size | Why |
|---|---|---|---|
| **VisDrone 2019 (full)** | Drone overhead, 10 classes | ~10 k images | Diverse cities, angles, altitudes. Already partially used. |
| **Stanford Drone Dataset** | Fixed overhead campus cameras | ~20 k frames | Closest geometry to our use case — overhead fixed camera, persons + vehicles. |
| **UA-DETRAC** | Fixed overhead traffic cameras | ~140 k frames | 100+ real intersections; excellent car diversity across weather and lighting. |
| **COWC** | Satellite/aerial overhead cars | ~32 k patches | Six cities; strong car appearance diversity. |
| **CrowdHuman** | Dense pedestrian scenes | ~15 k images | Directly addresses person detection weakness; heavily occluded examples. |
| **WiderPerson** | Diverse pedestrian contexts | ~13 k images | Includes overhead and elevated perspectives. |

Combined: ~230 k labeled images across many cameras, cities, lighting, and
seasons — without recording a single new frame.

> **Important:** do NOT include `Demo/ANMR0006.mp4`-derived frames in this
> general training run.  That footage should be reserved as a held-out test
> set to measure real-world performance.

**Student difficulty:** Medium (mostly dataset download, format conversion, and
merge; no new code required beyond using existing tools).

---

## 3. Tracker Is IoU-Only (ByteTrack)

**What you will see:** ID switches when two people or vehicles cross paths.
After occlusion (one object passing behind another), the tracker re-assigns the
wrong ID to each object when they re-emerge.

**Why:** ByteTrack matches tracks purely by IoU (bounding-box overlap).  It has
no appearance model, so when two objects occupy overlapping boxes, it has no
way to distinguish them.

**What to do:** Switch to **BoT-SORT**, which is already available in BoxMOT
(the tracker library already installed).  BoT-SORT adds a lightweight
re-identification step using appearance features — it can tell two people apart
even after they cross, because they look different.  For a fixed overhead
camera (no camera motion), the camera-motion-compensation part of BoT-SORT is
not needed and can be disabled.

Change the tracker initialisation in `run_coreml.py`:

```python
# from
tracker = ByteTrack(frame_rate=int(cap.fps), track_buffer=600, track_thresh=0.15)

# to
from boxmot.trackers import BoTrack
tracker = BoTrack(frame_rate=int(cap.fps), track_buffer=600, track_thresh=0.15)
```

After switching, measure ID-switch counts using `tools/test_bytetrack.py`
on a benchmark clip to confirm improvement.

**Student difficulty:** Low–Medium (tracker swap is a few lines; evaluating the
improvement requires running test scripts and comparing output videos).

---

## 4. No Automated Accuracy Metric

**What you will see:** There is no automated way to know whether a model change
improved or degraded performance.  Every evaluation currently requires watching
output video manually.

**Why:** No ground-truth count annotation exists for any test clip.

**What to do:**
1. Student manually counts cars and persons in a 60-second clip from at least
   **two different videos** — one being ANMR0006 (the original camera) and one
   from the benchmark set — producing ground-truth JSONs:
   ```json
   {"source": "ANMR0006.mp4", "clip_frames": [0, 3600],
    "cars": 12, "people": 7}
   ```
2. Write `tools/evaluate.py`: runs `count_objects.py` on each clip, compares
   output to the JSON, reports count error and percentage.
3. Any future model change that moves a count by more than ±1 from ground truth
   should be flagged before merging.

Using two different videos is important: it tests both the model's accuracy on
the original camera and its generalization to a second one.

**Student difficulty:** Low (most work is manual counting; the script is ~30
lines).

---

## 5. Person Detection Is Too Low

**What you will see:** Pedestrians are rarely detected, especially when small
(distant) or partially occluded.

**Why:** The current training data is car-biased.  Overhead person examples are
underrepresented, and most person instances in the data are large and
unoccluded — the model never learned to handle the hard cases.

**What to do:** This is solved almost entirely by limitation #2 (diverse
training data).  CrowdHuman and WiderPerson both provide dense, heavily
occluded pedestrian scenes at scale.  Once those datasets are merged in,
re-training should recover person detection without any additional labeling.

If person recall is still low after the dataset expansion, apply class weighting
(`cls_pw` in the Ultralytics config) to up-weight person loss during training.

**Student difficulty:** Low — no extra work beyond #2 if the datasets are
included.

---

## 6. Car False Positives

**What you will see:** The car count overshoots, especially on dense traffic or
close-camera vehicles.

**Why:** Two compounding causes: (a) CoreML shifts confidence to a lower range
(0.15–0.37), forcing a low detection threshold that admits noise; (b) the model
was trained on limited car appearance diversity, so it generalises poorly to
unusual sizes or angles.

**What to do:**
- **Short-term:** Run `tools/threshold_grid_search.py` against a ground-truth
  clip (produced in #4) to find the confidence value that maximises F1.
- **Long-term:** Solved by #2 (diverse training data from UA-DETRAC and COWC
  provides extensive car diversity).  A better-trained model has more stable
  confidence outputs and requires less threshold tuning.

**Student difficulty:** Low (grid search is an existing script).

---

## 7. Ghost Zone Edge Cases

Three known counting errors caused by limitations of the current ghost zone
system.  All three are less likely to occur once the model from #1 and #2 is in
place — a stronger general detector fragments less, reducing pressure on the
ghost zone entirely.  They are documented here in case they persist after
retraining.

### 7a. One close car occasionally double-counted
A large (close-camera) car's track can fragment and re-appear more than 180 px
from the ghost zone centre, slipping past suppression.  **Fix:** replace the
circular ghost with an elliptical zone oriented in the car's direction of
motion.  ~20 lines of code.  **Difficulty:** Low–Medium.

### 7b. Person double-counted on ROI re-entry
A pedestrian who exits the ROI and re-enters from a slightly different position
arrives outside the ghost radius placed at the exit point.  **Fix:** implement
a boundary-segment ghost that covers a stretch of the ROI edge rather than a
single point.  **Difficulty:** Medium.

### 7c. Stopped car double-counted
A car stopped at a red light can have its track die (confidence dip) and
restart after the ghost expires (~0.75 s for close cars), triggering a second
count.  **Fix:** a stationary-object memory dict that keeps a position
suppressed for as long as the object stays still, independent of the ghost
timeout.  ~50 lines.  **Difficulty:** Medium.

---

## 8. No Multi-Camera Infrastructure

**What you will see:** Deploying to a second camera requires editing constants
inside `run_coreml.py` directly (ROI coordinates, confidence thresholds, ghost
zone parameters).

**Why:** The original system was designed for one installation.  With the goal
now being a general multi-camera deployment, this is a structural gap.

**What to do:** Abstract all camera-specific values into a YAML config:

```yaml
# configs/cameras/intersection_A.yaml
roi: [[120, 80], [1800, 80], [1800, 950], [120, 950]]
conf_car: 0.15
conf_person: 0.45
min_track_age: {car: 5, person: 20}
ghost_radius: {car: 80, person: 80}
ghost_timeout: {car: 90, person: 120}
```

`run_coreml.py` loads this via `--camera-config`.  Adding a new camera is a
new YAML file — no code changes.

**Student difficulty:** Low (~40 lines; config loader + argparse addition).

---

## Priority Order

| Priority | Limitation | Why now | Effort |
|---|---|---|---|
| **1** | #4 Accuracy metric | Cannot measure any improvement without it | Low |
| **2** | #2 Diverse training data | Root cause of most other problems; single highest-leverage action | Medium |
| **3** | #1 Upgrade to YOLOv8m | More capacity to hold the larger dataset; one-line change | Low |
| **4** | #8 Multi-camera YAML config | Required infrastructure for the general-model goal | Low |
| **5** | #3 Tracker → BoT-SORT | Better ID retention across occlusion; quick swap | Low–Medium |
| **6** | #5 Person detection | Resolved by #2; only needs extra work if still weak after retraining | Low |
| **7** | #6 Car false positives | Reduced by #2; threshold search as quick fix | Low |
| **8** | #7 Ghost zone edge cases | Lower priority — will largely self-resolve with a stronger detector | Medium |
