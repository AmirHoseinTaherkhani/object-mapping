#!/bin/bash
#SBATCH --job-name=objmap_trt_export
#SBATCH --output=logs/trt_export_%j.out
#SBATCH --error=logs/trt_export_%j.err
#SBATCH --partition=shared
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

PYTHON=$HOME/.conda/envs/objmap/bin/python

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Node : $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PYTHON -u realtime_inference/export_tensorrt.py

# After the job completes, pull the engine back to your Mac:
#   rsync -av at1293@premise.sr.unh.edu:~/objmap/models/weights/best_v3_merged.engine \
#             ./models/weights/
