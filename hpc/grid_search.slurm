#!/bin/bash
#SBATCH --job-name=objmap_grid
#SBATCH --output=logs/grid_%j.out
#SBATCH --error=logs/grid_%j.err
#SBATCH --partition=shared
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

PYTHON=$HOME/.conda/envs/objmap/bin/python

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs

echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

$PYTHON -u tools/threshold_grid_search.py
