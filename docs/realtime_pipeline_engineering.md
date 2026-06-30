# Real-Time Counting Pipeline: Engineering Challenges and Solutions

This document records every significant engineering problem encountered when
adapting the reference PyTorch counter (`counting_experiment/count_objects.py`)
into the real-time CoreML pipeline (`realtime_inference/run_coreml.py`).  Each
section describes the problem, why it occurred, and the solution.

---

## 1. Why a Separate Real-Time Pipeline Was Needed

The reference pipeline (`count_objects.py`) runs YOLOv8s inference in PyTorch
on every frame of a 1920×1080 @ 60 fps video.  On an Apple Silicon Mac that
yields roughly 12–18 fps — far below the camera's 60 fps.  The result is a
progressive lag: the processing queue grows until frames are dropped by the OS,
and eventually the live view falls seconds behind the real world.

The goal was to reach ≥30 fps so that the system could serve as a real-time
demo at the intersection.

---

## 2. CoreML Export: Getting the Model onto the Neural Engine

### 2.1 Why CoreML
Apple's Neural Engine (ANE) is a dedicated matrix-multiply accelerator built
into every M-series chip.  Routing a YOLOv8 model through CoreML's `.mlpackage`
format makes the ANE handle the heavy convolutions, freeing the CPU and GPU for
video decode and display.  In practice this gives a 3–4× throughput increase
over running the same `.pt` weights through PyTorch.

### 2.2 The NMS-inside-model bug
The first CoreML export used `nms=True`, which bakes Non-Maximum Suppression
into the CoreML graph itself.  This broke the Ultralytics output parser in
version 8.x: the baked NMS produced a differently-shaped tensor that the
Python wrapper could not parse, so every inference call returned empty boxes —
the model appeared to detect nothing.

**Fix:** export without `nms=True`.  NMS runs in Python after the ANE returns
raw boxes, adding negligible overhead.

### 2.3 Confidence distribution shift
After the export was fixed, car detections reappeared but were almost
immediately lost again.  Investigation with `--verbose` revealed that ByteTrack
was creating no new car tracks.

**Root cause:** CoreML's quantisation changes the model's output scale.
PyTorch produces car confidence scores in the 0.45–0.85 range.  The same model
exported to CoreML produces car scores in the 0.15–0.37 range.  Person scores
are less affected (they stay near 0.90).

This caused three cascading failures:

| Layer | What happened |
|---|---|
| Pre-NMS confidence floor (was 0.45) | All cars filtered out before the tracker ever saw them |
| ByteTrack `track_thresh` (was 0.45) | Any car that slipped through went to stage-2 (recovery mode), which only matches existing tracks — new car tracks were never created |
| Detection confidence threshold `--conf-car` (was 0.50) | Same filtering problem at the count stage |

**Fix:**
- `COREML_CONF_FLOOR = 0.10` — a loose pre-NMS floor applied to the model call
- `--conf-car` default lowered to `0.15`
- ByteTrack `track_thresh = 0.15` so CoreML-range cars enter as stage-1 detections and can spawn new tracks

---

## 3. Frame Skipping

### 3.1 Why frame skip
Even with ANE acceleration, running YOLO on every one of 60 frames per second
consumed more ANE budget than available on the demo hardware.  Frame skipping
runs YOLO on every Nth frame and lets the tracker fill the gaps.

### 3.2 The critical np.empty bug
The first implementation passed `np.empty((0, 6))` to `tracker.update()` on
skipped frames — an empty detection array signalling "nothing was seen."

**What this actually does to ByteTrack:** ByteTrack has two track states:
*confirmed* and *lost*.  When it receives zero detections, every confirmed track
transitions to *lost*.  Lost tracks are not output by `tracker.update()`.  The
main loop therefore sees no tracks on skipped frames, `track_age` stalls (it is
only incremented on frames where a track appears in the output), and
`MIN_TRACK_AGE` is never reached — no object is ever counted.

An attempt was made to fix this by passing `last_dets` (the most recent real
detection) instead of empty.  This worked for counting but introduced a
different problem (described in §3.3).  After a revert cycle the final correct
solution was confirmed: **always pass `last_dets` to `tracker.update()` on
skip frames.**  This keeps all tracks confirmed and their Kalman filters
updating at every frame.

### 3.3 The Kalman velocity corruption (attempted fix, reverted)
One intermediate commit tried passing `np.empty` instead of `last_dets`,
believing that ByteTrack's Kalman filter would then predict forward using
velocity rather than being fed a stale position.  In theory this is correct.
In practice, ByteTrack's Kalman prediction uses the stale position as its
starting point regardless, and the "prediction without detections" mode
produced position estimates that diverged from the actual object after 2–3
skip frames, causing IoU mismatches and track fragmentation on the next
detection frame.  This commit was reverted.

---

## 4. Asynchronous Video Capture

### 4.1 Purpose
`AsyncCapture` reads video frames in a background thread into a small ring
buffer (`deque(maxlen=4)`).  The main inference loop pulls from the buffer
without blocking on disk I/O.  For a live camera or RTSP stream this ensures
the system always displays the most recent frame, dropping stale frames rather
than queuing them.

### 4.2 The None-frame bug
`AsyncCapture.read()` returns `(True, None)` when the ring buffer is
momentarily empty but the video is still running (background thread has not
produced a new frame yet).  The original guard was `if not ret: continue`,
which only catches `ret=False` (end of video).  A `None` frame slipped through
to the YOLO model.

**What CoreML does with None input:** rather than raising an exception, it
silently returns empty predictions — exactly mimicking the behaviour of a
frame with no objects.  This caused near-zero detection rates and
artificially inflated throughput numbers whenever the inference loop ran
faster than the decoder.

**Fix:** explicit guard `if frame is None: sleep(0.001); continue`.

### 4.3 Long-run crash (SIGABRT) on recorded files
A hard OS-level process abort (`Abort trap: 6` / SIGABRT) was observed when
running the pipeline on the full 1920×1080 @ 60fps custom video for 1800+
frames.  The same pipeline ran without crashing on short YouTube clips.

**Root cause (hypothesis):** the CoreML ANE maintains internal state across
inference calls.  The background reader thread continuously decodes frames and
pushes them into the ring buffer at the full decode speed of the video (well
above the ANE's inference rate).  Over hundreds of inference calls this
thread–ANE interaction accumulated until the ANE runtime aborted.  Short clips
finished before hitting the threshold.

**Fix:** introduce `SyncCapture` — a simple synchronous `cv2.VideoCapture`
wrapper with no background thread — and auto-select it for all file sources.
`AsyncCapture` is kept only for live/RTSP sources where staying current by
dropping old frames is the correct behaviour.  For recorded files synchronous
reading is both safer and equally fast since disk I/O is not the bottleneck.

---

## 5. ByteTrack Configuration

Two parameters required careful calibration beyond the defaults:

**`track_thresh = 0.15`** (default is ~0.45)  
As described in §2.3, CoreML cars arrive at 0.15–0.37 confidence.
Setting `track_thresh` to match allows ByteTrack to treat them as
high-confidence stage-1 detections capable of spawning new tracks.

**`track_buffer = 600`** (600 frames = 10 s at 60 fps)  
The default buffer is 30 frames.  A vehicle or pedestrian can be
fully occluded (e.g. under a tree canopy, or another vehicle overhead)
for several seconds at this camera angle.  A 30-frame buffer kills the
track during occlusion, and the object re-appears as a new ID — the
primary source of double-counts before the ghost zone system was added.
600 frames ensures tracks survive all observed occlusion durations.

---

## 6. Ghost Zone Deduplication System

### 6.1 The problem ghost zones solve
Even with `track_buffer=600`, brief confidence dips in the CoreML output cause
a track to die and immediately restart with a new ID.  Without a deduplication
layer, every such fragmentation increments the counter.

A *ghost zone* is a circular suppression region placed at the last known
centroid of a track when it dies.  Any new track appearing inside an active
ghost zone is suppressed — it is treated as the same physical object, not a
new arrival.

### 6.2 Per-class parameters
The reference pipeline (`count_objects.py`) uses flat constants for all
classes: `MIN_TRACK_AGE=30`, `GHOST_RADIUS=80px`, `GHOST_TIMEOUT=300 frames`.
These work because PyTorch confidence is stable and every frame is processed.

In the CoreML pipeline, cars and persons need different parameters:

| Parameter | Cars | Persons | Reason |
|---|---|---|---|
| `MIN_TRACK_AGE` | 5 | 20 | Cars fragment at low confidence — count early before the track dies. Persons are more stable; 20 frames filters arm-opening fragments naturally. |
| `GHOST_TIMEOUT` | 90 | 120 | Car fragments reappear within <1 s. Person fragments are rarer; 120 frames is enough without blocking a new pedestrian in the same spot. |
| `GHOST_RADIUS` | size-adaptive | 80px | See §6.3. |

The first tuning attempt used a single set of values for both classes.  Setting
`MIN_TRACK_AGE` low enough for cars (3–5) caused short arm-opening person
fragments to be counted (they lasted longer than 3 frames).  Setting it high
enough for persons (20–30) caused cars to die before being counted at all.
Per-class parameters resolved the conflict.

### 6.3 Size-adaptive ghost radius for cars
Cars on the near (left) side of the frame project to a larger pixel footprint.
When such a car's track fragments, the new fragment's centroid can be more than
80px from the dead track's last centroid because:

- **Edge clipping:** as the car transitions from partially to fully visible, the
  visible portion of the bounding box shifts, moving the centroid.
- **Faster pixel motion:** objects close to the camera cover more pixels per
  frame of real motion.

A fixed 80px radius missed these close-car re-fragments.  The fix: compute the
ghost radius from the car's bounding box width at the time it died:

```
radius  = max(80,  box_width × 0.6)
timeout = max(45,  90 × 100 / max(box_width, 100))
```

A 300px-wide car gets a 180px radius and a 45-frame timeout.  A 100px-wide
distant car keeps the base 80px radius and 90-frame timeout.  The inverse
relationship between size and timeout prevents a close-car ghost from blocking
the next legitimate car entering the same corridor for too long.

The radius multiplier was tuned through several iterations:
- `0.6×` — current value; catches most fragments without overlapping the
  adjacent lane
- `0.8×` — tried and reverted; the wider radius spanned the adjacent incoming
  lane and suppressed legitimate new cars

### 6.4 Person ghost uses first centroid, not current
With `MIN_TRACK_AGE=20`, a person walks approximately 80px before the ghost
check fires at counting time.  If the ghost check uses the *current* centroid
at frame 20, the centroid has drifted close to the edge of `GHOST_RADIUS=80`
and may escape suppression even though the track started directly on top of the
ghost.

**Fix:** for persons, store the entire centroid history in a deque.  At
counting time, retrieve the *first* centroid (the one recorded when the track
appeared) and use that for the ghost check.  The track started there — that is
the position that should be compared to the ghost zone centre.

Cars are unaffected: `MIN_TRACK_AGE=5` means at most 5px of drift, well within
the ghost radius.

### 6.5 Died-young asymmetry
A track that dies before reaching `MIN_TRACK_AGE` ("died young") was never
counted.  The question is whether it should leave a ghost zone:

- **Car died-young → ghost zone IS created.**  A close car's track can dip
  below the confidence threshold for a single frame, creating a fragment that
  dies at age 1–4.  Without a ghost, the same physical car spawns a new track
  immediately at the same location and — if that track survives to
  `MIN_TRACK_AGE` — is counted.  The ghost prevents this.

- **Person died-young → ghost zone is NOT created.**  Short person fragments
  appear at crosswalk entry points (partial visibility at the ROI edge).  If
  these fragments leave ghost zones, they block the next legitimate pedestrian
  who steps into the same position.  Since person fragments that die young are
  noise, they should not suppress anything.

---

## 7. Arm-Opening Double-Count (Live-Track Proximity Check)

### 7.1 The scenario
When a tracked person opens their arms wide, YOLOv8 sometimes loses the
original bounding box and immediately creates a new, slightly wider one.
ByteTrack assigns a new track ID (T2) to the new box while the original
(T1) is still alive.

At the moment T2 appears, T1 has not died — there is no ghost zone yet.
The standard ghost-zone check therefore does not suppress T2.  Once T2
reaches `MIN_TRACK_AGE=20`, it is counted as a second person even though
the same physical person is still being tracked as T1.

### 7.2 The solution
At the moment a new track is eligible for counting (age reaches
`MIN_TRACK_AGE`), a second check runs against all *currently alive* tracks:

1. Retrieve T2's age at this moment.
2. Look up T1's centroid history from `age` frames ago — this is
   approximately where T1 was *when T2 first appeared*.
3. If the distance between T1's historical position and T2's first centroid
   is within `GHOST_RADIUS`, suppress T2.

```python
ago = min(age - 1, len(centroids[other_tid]) - 1)
hx, hy = centroids[other_tid][-ago - 1]   # T1's position when T2 started
```

This catches arm-opening without requiring T1 to die first.

---

## 8. In-Car Person Suppression

At this camera angle, the windshield and rear window of a car are visible.
YOLOv8 sometimes detects the driver or passengers as persons *inside* the
car's bounding box.  These detections are real (there are people there) but
they should not be counted as pedestrians.

**Fix:** after each inference frame, any person detection whose centroid falls
inside a detected car bounding box is removed from the detection list before it
is passed to the tracker.  This runs before ByteTrack sees any detections, so
no in-car person track is ever created.

---

## 9. Stationary Car Filter (Parked Car Suppression)

The ROI includes a small section of a parking area at the edge of the frame.
Parked cars generate persistent tracks that, if counted, inflate the car total.

**Fix:** maintain a motion history of the last N centroids per track.  A track
is classified as *stationary* if the maximum displacement across its history is
below `MOTION_MIN_PX = 8` pixels.  Stationary tracks are displayed in grey and
excluded from the count.  They continue to be tracked normally; if the car
later moves, it becomes eligible for counting.

---

## 10. ROI Masking

Only objects whose centroid falls inside the manually-defined Region of
Interest polygon are tracked and counted.  The ROI is drawn once on the first
frame using `select_roi.py` and saved to `counting_experiment/roi.json`.
Filtering happens before detections reach ByteTrack so the tracker never
creates tracks for out-of-ROI objects.

The ROI addresses two practical problems:
- Vehicles on adjacent roads (visible at the edge of the frame) must not be counted.
- Parked cars at the boundary of the frame are partially excluded by the ROI before the stationary filter handles the rest.

---

## 11. Known Remaining Edge Cases

These issues were investigated but not fully resolved:

**One close car occasionally double-counted.** When a very large (close-camera)
car fragments, the new track's centroid sometimes jumps more than 180px from
the ghost zone centre (the `0.6×` radius limit).  Widening to `0.8×` caused
false suppression of cars in the adjacent lane, so `0.6×` is kept as the
current balance.

**Person double-counted on ROI re-entry.** A person who walks out of the ROI
and back in from a different angle arrives at a position the ghost zone does not
cover (the ghost was placed at the exit point, not the re-entry point).  Fixing
this properly requires boundary-aware ghost zones.

**Stopped car at pedestrian crossing.** A car waiting at a red light can stop
for several seconds.  If its track dies and the ghost expires before the car
moves again, the restart is counted as a new car.  Multiple fixes were attempted
(extending ghost timeout for stationary cars) but all overcorrected, blocking
new cars in the same lane.  The feature was reverted.

---

## Summary: Reference Baseline vs. CoreML Pipeline

| Concern | `count_objects.py` | `run_coreml.py` |
|---|---|---|
| Inference backend | PyTorch (CPU/MPS) | CoreML (ANE) |
| Frame strategy | Every frame | Every N frames (`--skip-n`) |
| Skip-frame tracker feed | N/A | `last_dets` (not empty) |
| `MIN_TRACK_AGE` | 30 (flat) | 5 cars / 20 persons |
| `GHOST_RADIUS` | 80px (flat) | 80px persons / size-adaptive cars |
| `GHOST_TIMEOUT` | 300 frames (flat) | 90 cars / 120 persons |
| Ghost for died-young | Not applicable (stable conf) | Car yes / Person no |
| Person ghost centroid | Current | First (drift compensation) |
| Arm-opening check | Not needed | Live-track proximity check |
| Capture mode | Synchronous | Synchronous (file) / Async (live) |
