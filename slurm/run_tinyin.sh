#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -J vm_tinyin
#SBATCH -o vm_tinyin_%j.out
#SBATCH -e vm_tinyin_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
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
export WANDB_MODE=OFFLINE

srun python3 \
      scripts/run_image.py \
      config/image/pretrain_viewmaker_tinyin_simclr.json \

echo "Run completed at:- "
date

