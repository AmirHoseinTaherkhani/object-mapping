# Counting Experiment — Vehicle & Person Counter

Counts moving vehicles and pedestrians in a fixed overhead surveillance video using YOLOv8s + ByteTrack.

## Scripts

| Script | Purpose |
|---|---|
| `select_roi.py` | Interactive polygon ROI selector — saves `roi.json` |
| `count_objects.py` | Main counting pipeline |
| `check_counts.py` | Sanity-check replay of the output video |

## Quick Start

```bash
# 1. Define the counting zone (click polygon, press Enter)
python select_roi.py

# 2. Run the counter
python count_objects.py

# 3. (Optional) Verify counts frame-by-frame
python check_counts.py
```

Run headlessly on HPC:
```bash
python count_objects.py --no-display
```

## Configuration

| Argument | Default | Description |
|---|---|---|
| `--conf-car` | `0.50` | YOLO confidence threshold for cars |
| `--conf-person` | `0.45` | YOLO confidence threshold for persons |
| `--no-display` | off | Skip `cv2.imshow` for headless runs |

Constants in `count_objects.py`:

| Constant | Value | Description |
|---|---|---|
| `MOTION_BUFFER` | 45 frames | Window for parked-car displacement check |
| `MOTION_MIN_PX` | 8 px | Max displacement to be considered parked |
| `MIN_TRACK_AGE` | 30 frames | Frames a track must be seen before counting |
| `GHOST_RADIUS` | 80 px | Spatial deduplication search radius |
| `GHOST_TIMEOUT` | 300 frames | How long a ghost zone is held after track death |

## Model

- Detector: `best_v3_merged.pt` (YOLOv8s, 2 classes: `person=0`, `car=1`)
- Tracker: ByteTrack (`frame_rate=60`, `track_buffer=600`)
- No Re-ID model — pure IoU + Kalman matching

## Problems Encountered and Solutions

### 1. Persons inside passing cars counted as pedestrians
**Problem:** YOLO detects faces/torsos visible through car windows and assigns them person tracks.

**Fix:** Before passing detections to the tracker, any person detection whose centroid falls inside a car bounding box is dropped. From an overhead camera, a real pedestrian's centre point should never sit inside a car box.

```python
car_boxes = [(d[0],d[1],d[2],d[3]) for d in dets if cls == CLASS_CAR]
# drop person if its centroid is inside any car box
```

---

### 2. Stationary person counted multiple times (arms opening → new track ID)
**Problem:** When a person extends their arms, the bounding box shape changes significantly. The StrongSORT tracker with appearance Re-ID (OSNet) treated this as a different person and spawned a new track ID, incrementing the counter again.

**Root cause:** OSNet Re-ID was trained on street-level pedestrian datasets (MSMT17). From an overhead surveillance angle the appearance embeddings are unreliable — adjusting `max_cos_dist` just made the tracker accept bad matches.

**Fix:** Switched from StrongSORT to **ByteTrack**. ByteTrack uses a two-stage IoU + Kalman cascade with no appearance model:
- Stage 1: match high-confidence detections
- Stage 2: sweep low-confidence detections to recover tracks before they die

When arms extend, the confidence may dip but the stage-2 sweep keeps the same track ID. No appearance embedding means no overhead-camera blindspot.

---

### 3. Passing pedestrian counted multiple times (detection gaps)
**Problem:** A fast-moving pedestrian who passes briefly through the ROI had intermittent detections. Each detection gap caused the track to die and be reborn with a new ID, each of which incremented the count.

**Fix (three layers):**

**a) Lower person threshold (`conf_person=0.45`):** Detects the person in more frames, shrinking the gaps that kill the track.

**b) `track_buffer=600` (10 s at 60 fps):** ByteTrack holds a lost track alive for 10 seconds. Kalman prediction bridges short gaps without spawning a new ID.

**c) Ghost-zone deduplication:** When a track finally dies, its last centroid is stored as a "ghost" for `GHOST_TIMEOUT` frames. If a new track of the same class appears within `GHOST_RADIUS` pixels of a ghost, it is silently absorbed into `counted_ids` without incrementing the counter.

```
track dies → ghost stored at (cx, cy, cls, frame)
new track appears nearby → check ghost zone → suppress count increment
```

---

### 4. Slow inference (~1 s/frame with StrongSORT)
**Problem:** StrongSORT runs ECC (Enhanced Correlation Coefficient) camera-motion compensation via `cv2.findTransformECC` on the full 1920×1080 frame on every tick — ~300–500 ms per frame on CPU.

**Fix:** Replaced StrongSORT with ByteTrack (no CMC needed, no Re-ID). Throughput went from ~1 fps → **~19 fps** on an Apple M-series Mac.

---

### 5. Parked cars counted as moving vehicles
**Problem:** Cars parked within the ROI were being counted.

**Fix:** 45-frame centroid displacement buffer per track. If the maximum displacement across the buffer is below 8 px the car is classified as parked (shown in grey) and excluded from the count.
