#!/bin/bash
#SBATCH --job-name=yolo11m_finetune
#SBATCH --output=logs/yolo11m_%j.out
#SBATCH --error=logs/yolo11m_%j.err
#SBATCH --partition=shared
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

MODEL_SIZE="m"
MODEL_NAME="yolo11${MODEL_SIZE}"
PROJ_DIR="$HOME/objmap"
EXP_DIR="$PROJ_DIR/experiments/yolo11_experiments/finetuned/${MODEL_NAME}"
DATA_YAML="$PROJ_DIR/labeling_data/trainingData/data_hpc.yaml"
PYTHON="$HOME/.conda/envs/objmap/bin/python"

cd "$PROJ_DIR"
mkdir -p logs "$EXP_DIR"

echo "================================================================"
echo "Job  : $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Model: $MODEL_NAME"
echo "Start: $(date)"
echo "================================================================"

# ── Step 1: write HPC-specific data.yaml ────────────────────────────────────
cat > "$DATA_YAML" << ENDYAML
path: $PROJ_DIR/labeling_data/trainingData
train: train/images
val:   valid/images
test:  test/images
nc: 2
names:
  - person
  - car
ENDYAML
echo "Written: $DATA_YAML"

# ── Step 2: download public datasets (if not already present) ────────────────
echo ""
echo "── Dataset download ────────────────────────────────────────────"

ROBOFLOW_API_KEY="${ROBOFLOW_API_KEY:-}"   # inherit from environment if set

if [ -n "$ROBOFLOW_API_KEY" ]; then
    echo "Roboflow key found — downloading all 6 datasets…"
    $PYTHON tools/download_public_datasets.py --merge --max-per-dataset 5000
else
    echo "No ROBOFLOW_API_KEY set — downloading VisDrone only (no registration required)."
    $PYTHON tools/download_public_datasets.py --dataset visdrone --merge --max-per-dataset 5000
    echo ""
    echo "  Other datasets (CrowdHuman, WiderPerson, UA-DETRAC, COWC, SDD) skipped."
    echo "  Export ROBOFLOW_API_KEY and re-submit to include them."
fi

# Report final dataset sizes
echo ""
echo "── Dataset sizes after merge ───────────────────────────────────"
echo "  Train  : $(ls "$PROJ_DIR/labeling_data/trainingData/train/images" | wc -l) images"
echo "  Valid  : $(ls "$PROJ_DIR/labeling_data/trainingData/valid/images" | wc -l) images"
echo "  Test   : $(ls "$PROJ_DIR/labeling_data/trainingData/test/images"  | wc -l) images"

# ── Step 3: fine-tune YOLO11s ───────────────────────────────────────────────
rm -f "$PROJ_DIR/labeling_data/trainingData/train/labels.cache"
rm -f "$PROJ_DIR/labeling_data/trainingData/valid/labels.cache"
rm -f "$PROJ_DIR/labeling_data/trainingData/test/labels.cache"
echo ""
echo "── Fine-tuning $MODEL_NAME ────────────────────────────────────────"
$PYTHON -u -c "
from ultralytics import YOLO
import torch

print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')

model = YOLO('$PROJ_DIR/models/weights/${MODEL_NAME}.pt')   # pre-downloaded on login node
results = model.train(
    data='$DATA_YAML',
    epochs=150,
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    project='$EXP_DIR',
    name='train',
    exist_ok=True,
    # Augmentation
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    # Regularisation
    weight_decay=0.0005,
    warmup_epochs=3,
    close_mosaic=10,
    # Class weights to counteract car:person imbalance
    cls=1.0,
    # Patience for early stopping
    patience=30,
    save_period=10,   # checkpoint every 10 epochs
    verbose=True,
)
print('Training complete.')
print('Best weights:', results.save_dir)
"

# ── Step 4: evaluate on test set (mAP) ──────────────────────────────────────
echo ""
echo "── Test set evaluation ─────────────────────────────────────────"
BEST_PT="$EXP_DIR/train/weights/best.pt"
if [ -f "$BEST_PT" ]; then
    $PYTHON -u -c "
from ultralytics import YOLO
model = YOLO('$BEST_PT')
metrics = model.val(data='$DATA_YAML', split='test', device=0, verbose=True)
print('mAP50     :', metrics.box.map50)
print('mAP50-95  :', metrics.box.map)
print('Precision :', metrics.box.mp)
print('Recall    :', metrics.box.mr)
" | tee "$EXP_DIR/test_metrics.txt"
else
    echo "best.pt not found at $BEST_PT — training may have failed."
fi

# ── Step 5: run counting on benchmark videos ─────────────────────────────────
echo ""
echo "── Benchmark video counting ────────────────────────────────────"
BENCH_OUT="$EXP_DIR/benchmark_results"
mkdir -p "$BENCH_OUT"

for VID in "$PROJ_DIR/Demo/videos/"*.mp4 "$PROJ_DIR/Demo/videos/"*.avi \
           "$PROJ_DIR/Demo/benchmark_stmarc.avi"; do
    [ -f "$VID" ] || continue
    STEM=$(basename "$VID" | sed 's/\.[^.]*$//')
    echo "  Processing: $STEM"
    $PYTHON tools/run_yolo11_experiment.py \
        --weights "$BEST_PT" \
        --source "$VID" \
        --roi none \
        --no-display \
        --output "$BENCH_OUT/${STEM}.mp4" \
        2>&1 | tail -6 | tee -a "$BENCH_OUT/counts.txt"
    echo "---" >> "$BENCH_OUT/counts.txt"
done

# ANMR0006 uses real ROI
if [ -f "$PROJ_DIR/Demo/ANMR0006.mp4" ]; then
    echo "  Processing: ANMR0006"
    $PYTHON tools/run_yolo11_experiment.py \
        --weights "$BEST_PT" \
        --source "$PROJ_DIR/Demo/ANMR0006.mp4" \
        --no-display \
        --output "$BENCH_OUT/ANMR0006.mp4" \
        2>&1 | tail -6 | tee -a "$BENCH_OUT/counts.txt"
fi

echo ""
echo "================================================================"
echo "DONE — $MODEL_NAME"
echo "Weights  : $BEST_PT"
echo "Test mAP : $EXP_DIR/test_metrics.txt"
echo "Bench    : $BENCH_OUT/counts.txt"
echo "Finished : $(date)"
echo "================================================================"
