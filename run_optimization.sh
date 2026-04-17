#!/bin/bash
#SBATCH --job-name=optuna_hpo
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G

# USO: sbatch run_optimization.sh [MODELLO] [DATASET] [PRETRAIN] [TRIALS]
# Esempio: sbatch run_optimization.sh Hybrid IBD False 20

MODEL=${1:-Hybrid}
DATASET=${2:-IBD}
PRETRAIN=${3:-False}
TRIALS=${4:-50}

mkdir -p logs

module purge
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# Installazione di optuna nel caso non ci fosse
python3 -m pip install --user optuna

echo "=========================================="
echo "Avvio Optuna Search"
echo "Modello: $MODEL"
echo "Dataset: $DATASET"
echo "Pretrain: $PRETRAIN"
echo "Trials: $TRIALS"
echo "=========================================="

srun python3 src/tuning/optuna_study.py \
    --model $MODEL \
    --dataset $DATASET \
    --pretrain $PRETRAIN \
    --n-trials $TRIALS \
    --epochs-pretrain 30 \
    --epochs-finetune 40

echo "Ricerca Ottimizzata Completata!"
