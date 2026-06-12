# Real-time Inference

Two optimised pipelines for running the counting system at real-time speed, one for each hardware scenario you might encounter at a demo.

| Pipeline | File | Hardware | Expected FPS |
|---|---|---|---|
| CoreML + frame skip | `run_coreml.py` | Mac (Apple Silicon) | 30–45 fps |
| TensorRT + frame skip | `run_tensorrt.py` | Any NVIDIA GPU | 60+ fps |

Both pipelines share identical counting logic with `counting_experiment/count_objects.py`: ROI masking, parked-car filter, ghost-zone deduplication, and in-car person suppression.

---

## Optimisations applied

### 1 — Hardware-accelerated model export

**CoreML** (Apple Silicon):
- The `.mlpackage` format routes computation through the Apple Neural Engine rather than the CPU/MPS GPU path. This gives a 3–4× speedup over loading a `.pt` file on the same machine.

**TensorRT** (NVIDIA GPU):
- The `.engine` format is a compiled, GPU-specific execution plan. TensorRT fuses layers, applies FP16 quantisation, and eliminates Python overhead. Typical speedup: 5–10× over PyTorch.

### 2 — Frame skipping (`--skip-n N`)

YOLO runs on every Nth frame only. On skipped frames, the ByteTrack tracker receives empty detections and uses its Kalman filter to predict where each object moved. The result on screen is smooth: tracks glide forward between detection hits.

- `--skip-n 2` (default): YOLO load halved, near-zero accuracy loss at 60 fps input
- `--skip-n 3`: YOLO load at 33%, suitable for very slow hardware or high frame-rate cameras

### 3 — Async video capture

A background thread continuously reads frames from disk or camera into a small ring buffer. The main thread pulls from the buffer without ever blocking on I/O. On fast hardware this is a minor gain; on systems where disk I/O competes with GPU, it prevents frame-rate dips.

---

## Step-by-step: CoreML (Mac)

### Step 1 — Export the model (once)

```bash
python export_coreml.py
# Produces: models/weights/best_v3_merged.mlpackage
```

### Step 2 — Run

```bash
cd realtime_inference

# Recorded video (default)
python run_coreml.py

# Live webcam
python run_coreml.py --source 0

# Headless (no window)
python run_coreml.py --no-display

# Tune aggressiveness
python run_coreml.py --skip-n 3 --conf-car 0.50 --conf-person 0.45
```

---

## Step-by-step: TensorRT (NVIDIA GPU)

### Step 1 — Export the engine

> **Important:** TensorRT engines are compiled for a specific GPU architecture. Build the engine on the same machine (or same GPU model) that will run the demo.

**Option A — build on the demo machine directly:**
```bash
python export_tensorrt.py
# Produces: models/weights/best_v3_merged.engine
```

**Option B — build on the Premise A100, then transfer:**
```bash
# On Premise login node:
sbatch export_tensorrt.slurm
# Check logs/trt_export_<job_id>.out for completion

# Pull the engine back to Mac:
rsync -av at1293@premise.sr.unh.edu:~/objmap/models/weights/best_v3_merged.engine \
          ../models/weights/
```

### Step 2 — Run

```bash
cd realtime_inference

# Recorded video (default)
python run_tensorrt.py

# Live webcam
python run_tensorrt.py --source 0

# IP / RTSP camera
python run_tensorrt.py --source rtsp://192.168.1.10:554/stream

# Headless
python run_tensorrt.py --no-display

# Second GPU
python run_tensorrt.py --device 1
```

---

## All arguments

### `run_coreml.py`

| Argument | Default | Description |
|---|---|---|
| `--weights` | `models/weights/best_v3_merged.mlpackage` | Path to CoreML package |
| `--source` | `Demo/ANMR0006.mp4` | Video file, `0` for webcam, `rtsp://` URL |
| `--skip-n` | `2` | Run YOLO every N frames |
| `--conf-car` | `0.50` | Confidence threshold for cars |
| `--conf-person` | `0.45` | Confidence threshold for persons |
| `--no-display` | off | Headless mode — skip `imshow` |

### `run_tensorrt.py`

Same as above, plus:

| Argument | Default | Description |
|---|---|---|
| `--weights` | `models/weights/best_v3_merged.engine` | Path to TensorRT engine |
| `--device` | `0` | CUDA device index |

### `export_tensorrt.py`

| Argument | Default | Description |
|---|---|---|
| `--fp32` | off | Use FP32 instead of FP16 (slower, use if accuracy degrades) |
| `--device` | `0` | CUDA device to build on |

---

## Choosing `--skip-n`

| Input FPS | `--skip-n` | Effective detection rate | Notes |
|---|---|---|---|
| 60 | 2 | 30 fps | Recommended — smooth tracking |
| 60 | 3 | 20 fps | Good if hardware is marginal |
| 30 | 2 | 15 fps | Minimum for comfortable tracking |
| 30 | 1 | 30 fps | No skipping — maximum accuracy |

ByteTrack's Kalman filter keeps tracks smooth between detection frames, so increasing `--skip-n` does not cause jitter on screen.

---

## Output

Both scripts write an annotated video to `realtime_inference/output_coreml.mp4` or `output_tensorrt.mp4` respectively, and print final counts to the terminal.
