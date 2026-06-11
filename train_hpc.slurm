#!/bin/bash
#SBATCH --job-name=objmap_finetune
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=shared
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00             # 12 h is plenty for 50 epochs on a GPU

# ── environment ──────────────────────────────────────────────────────────────
PYTHON=$HOME/.conda/envs/objmap/bin/python

# ── run ──────────────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Python : $($PYTHON --version)"

$PYTHON -u finetune_on_camera.py --hpc
