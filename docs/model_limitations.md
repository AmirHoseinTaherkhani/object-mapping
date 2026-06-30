# Model Limitations and Proposed Solutions

Current model: `best_v3_merged.pt` / `best_v3_merged.mlpackage`  
Pipeline: `realtime_inference/run_coreml.py`

Each limitation is described, its root cause explained, and a concrete solution
proposed.  Difficulty is rated for a student contributor.

---

## 1. Person Detection Is Too Low

**What you will see:** People walking through the intersection are rarely
detected, especially at a distance or when partially occluded.  The person
counter stays near zero on most test videos.

**Why:** The training dataset (~14 000 images) is heavily car-biased.  Overhead
person examples are underrepresented, and many of the person instances in the
data are small, blurry, or occluded — exactly the kind the model learns to
ignore.

**What to do:**
1. Use `tools/extract_labeling_frames.py` to sample 500–1000 frames from
   `Demo/ANMR0006.mp4` that contain pedestrians (pick busy crossing moments).
2. Label them in a tool like CVAT or Label Studio.  Each person needs a tight
   bounding box.
3. Re-run training with the new person-heavy split added to the merged dataset.
   Use `cls_pw` (class weight in Ultralytics YOLO config) to up-weight the
   person class during loss computation.
4. Re-export to CoreML.

**Student difficulty:** Medium (labeling is time-consuming but straightforward;
re-training on HPC is one `sbatch` command once the data is ready).

---

## 2. Weak Generalization to Other Camera Locations

**What you will see:** On any video other than `ANMR0006.mp4` the model
detects very few objects or detects the wrong class.  Even slightly different
lighting, angle, or resolution causes a large drop in accuracy.

**Why:** The model was fine-tuned on footage from a single fixed camera.  The
training distribution is extremely narrow.

**What to do:**
1. Collect overhead footage from at least 3–5 additional intersection cameras
   (different cities or buildings are fine).
2. Label ~200 frames per location (cars only is fine if person labeling is
   resource-constrained).
3. Include all locations in the merged training dataset.
4. Re-train.

**Student difficulty:** Medium–High (mostly a data collection and labeling
project; no new code required).

---

## 3. Car False Positives (Noisy Detection, Especially Close Cars)

**What you will see:** The car counter overshoots on dense traffic scenes.
Close-camera cars (large bounding boxes) are particularly prone to multiple
detections per physical vehicle per frame.

**Why:** CoreML shifts car confidence scores to a lower range (0.15–0.37 vs
PyTorch's 0.45+).  To avoid missing cars entirely, the confidence threshold was
lowered to `0.15`, which also admits more noise.  The ghost zone system absorbs
most duplicate counts but not all.

**What to do (short-term):** Run `tools/threshold_grid_search.py` on a labelled
clip of ANMR0006 to find the confidence threshold that maximises F1 for cars.
Then re-export the CoreML model with Ultralytics' post-processing calibrated to
that threshold.

**What to do (long-term):** Re-train with more close-camera car examples where
the ground truth contains exactly one box per physical car.  This teaches the
model not to fire multiple times on a single vehicle.

**Student difficulty:** Low (threshold search is an existing script; re-training
is the longer step).

---

## 4. One Close Car Occasionally Double-Counted

**What you will see:** On the left/near side of the frame, one car is
sometimes counted twice.  The second count typically appears within 1–2 seconds
of the first.

**Why:** When a close car's track fragments, the new fragment's centroid can
jump more than 180px from the ghost zone centre (the ghost radius for a
300px-wide car is `300 × 0.6 = 180px`).  The ghost does not reach the new
fragment's position, so it is counted.  Widening the radius to `0.8×` caught
these cases but caused adjacent-lane false suppressions.

**What to do:**
1. The cleanest code fix is to replace the circular ghost zone with an
   **elliptical** one oriented in the direction of motion.  A car moving
   horizontally needs a wide x-radius but a narrow y-radius; this avoids
   extending into the adjacent lane.
2. Alternatively, more close-camera training examples will reduce fragmentation
   frequency, reducing pressure on the ghost zone.

**Student difficulty:** Low–Medium (elliptical ghost zone is ~20 lines of code;
testing requires a few hours of video review).

---

## 5. Person Double-Counted When Re-Entering the ROI

**What you will see:** A pedestrian who walks out of the counting zone and
comes back in a slightly different position is counted twice.

**Why:** When the person exits, a ghost zone is placed at the exit centroid.
When they re-enter from a different angle or position, they arrive at a point
outside the ghost radius — the system treats them as a new person.

**What to do:** At ROI boundary crossings, create a **boundary-segment ghost**
that covers a stretch of the ROI edge rather than a single point.  Any new
track appearing on the same edge segment within the timeout window is
suppressed.  The ROI polygon already exists as a set of line segments —
the student needs to implement a point-to-segment distance check instead of
the current point-to-point check.

**Student difficulty:** Medium (geometry work, requires careful testing at the
boundary).

---

## 6. Stopped Car at Pedestrian Crossing Can Be Double-Counted

**What you will see:** A car that stops and waits at a red light for more than
~0.75 s is sometimes counted a second time when it starts moving again.

**Why:** The ghost timeout for close cars is short (45 frames, ~0.75 s) to
avoid blocking new cars in the same lane.  If the stopped car's track dies
during the wait (common due to confidence fluctuation) and the ghost expires
before the car moves, the resuming track has no ghost to suppress it.

**What to do:** Add a **stationary-object memory** layer: a separate dict that
records the position of any object classified as stationary (by the existing
`is_stationary()` logic) for as long as it stays still, regardless of ghost
timeout.  When the car moves again, keep the memory active for one extra ghost
timeout.  This is distinct from ghost zones and does not risk blocking the
adjacent lane.

**Student difficulty:** Medium (new data structure alongside existing code;
~50 lines).

---

## 7. No Automated Accuracy Metric

**What you will see:** Counting accuracy is verified by watching the output
video and counting manually.  There is no automated test that says "expected
7 cars, got 7."

**Why:** No ground-truth annotation file exists for ANMR0006.

**What to do:**
1. Student manually counts objects in a single 60-second clip of ANMR0006,
   frame by frame, producing a JSON:
   `{"cars": 12, "people": 7, "clip_start_frame": 0, "clip_end_frame": 3600}`
2. Write a script in `tools/` that runs `count_objects.py` on that clip and
   compares the output to the JSON, reporting precision/recall on counts.
3. Add this as a CI step: any model change that moves the count by more than
   ±1 from ground truth flags for review.

**Student difficulty:** Low (the labeling is the main work; the script is ~30
lines).

---

## 8. No Multi-Camera Support

**What you will see:** Deploying to a second intersection requires manually
editing constants inside `run_coreml.py` (ROI coordinates, confidence
thresholds, ghost zone parameters).

**Why:** The system was built for one specific camera installation.

**What to do:** Abstract all camera-specific values into a YAML config file:

```yaml
# configs/cameras/intersection_A.yaml
roi: [[120, 80], [1800, 80], [1800, 950], [120, 950]]
conf_car: 0.15
conf_person: 0.45
min_track_age: {car: 5, person: 20}
ghost_radius: {car: 80, person: 80}
ghost_timeout: {car: 90, person: 120}
```

`run_coreml.py` loads this via `--camera-config`.  A new camera is a new YAML
file — no code changes.

**Student difficulty:** Low (config loader + argparse change, ~40 lines; the
student can also do the per-camera tuning using the existing `--verbose` output
and grid search script).

---

## Priority Order

| # | Limitation | Impact | Effort |
|---|---|---|---|
| 7 | No accuracy metric | Blocks everything else — you cannot measure progress | Low |
| 1 | Person detection too low | Core feature barely works | Medium |
| 3 | Car false positives | Degrades count accuracy | Low (threshold) → Medium (retrain) |
| 8 | No multi-camera support | Limits deployment | Low |
| 4 | Close-car double-count | Known, infrequent | Low–Medium |
| 5 | ROI re-entry double-count | Infrequent | Medium |
| 6 | Stopped-car double-count | Infrequent | Medium |
| 2 | No generalization | Long-term research concern | High |
