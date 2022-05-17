#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -J tf_cifar10_a1
#SBATCH -o tf_cifar10_a1_%j.out
#SBATCH -e tf_cifar10_a1_%j.err
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
      config/image/transfer_viewmaker_cifar10_a1_simclr.json \
      --dataset cifar10

echo "Run completed at:- "
date

