#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -J tf_tinyin_a1new
#SBATCH -o tf_tinyin_a1new_%j.out
#SBATCH -e tf_tinyin_a1new_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=0
#SBATCH --time=0-06:00:00
#SBATCH --signal=SIGUSR1@90

## User python environment
HOME2=/gpfs/u/home/BNSS/BNSSsgwh
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/scratch/miniconda3

## Activate WMLCE virtual environment 
source init_env.sh
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# wandb offline flag
export WANDB_MODE=offline

srun python3 \
      scripts/run_meta_transfer.py \
      config/image/transfer_viewmaker_tinyin_a1new_simclr.json \
      --dataset tinyin

echo "Run completed at:- "
date



