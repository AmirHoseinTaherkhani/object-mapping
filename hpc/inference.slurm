#!/bin/bash
#SBATCH --job-name=objmap_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=shared
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

PYTHON=$HOME/.conda/envs/objmap/bin/python

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/model_comparison

echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

$PYTHON -u tools/test_bytetrack.py models/weights/best_v3_merged.pt --full \
  --conf-car 0.50 --conf-person 0.50
