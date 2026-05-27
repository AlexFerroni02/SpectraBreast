#!/bin/bash
#SBATCH --job-name=optuna_hpo
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G

# USO:
#   CNN (arch + training params):
#     sbatch run_final_optuna.sh CNN IBD 100
#
#   Transformer (solo training params, architettura fissa dal config):
#     sbatch run_final_optuna.sh Transformer IBD 50 configs/classification/IBD/Transformer/fine-tune/exp_02_smae.yaml
#
#   Hybrid (solo training params, architettura fissa dal config):
#     sbatch run_final_optuna.sh Hybrid IBD 50 configs/classification/IBD/Hybrid/exp_01_hybrid_finetune.yaml

MODEL=${1:-CNN}
DATASET=${2:-IBD}
TRIALS=${3:-100}
CONFIG=${4:-""}

mkdir -p logs

module purge
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

python3 -m pip install --user optuna tqdm timm einops

echo "=========================================="
echo "Avvio Optuna HPO Search"
echo "  Modello: $MODEL"
echo "  Dataset: $DATASET"
echo "  Trials:  $TRIALS"
if [ -n "$CONFIG" ]; then
    echo "  Config:  $CONFIG"
fi
echo "=========================================="

# Costruisci il comando
CMD="srun python3 -m src.tuning.optuna_final_search --model $MODEL --dataset $DATASET --n-trials $TRIALS"

# Aggiungi --config solo se fornito (obbligatorio per Transformer e Hybrid)
if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi

eval $CMD

echo "Ricerca Ottimizzata Completata!"
