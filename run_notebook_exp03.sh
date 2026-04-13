#!/bin/bash
#SBATCH --job-name=exp03_cnn
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8G

# --- SETUP AMBIENTE ---
module purge
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# Passiamo il config del terzo esperimento al notebook
export EXP_CONFIG_FILE="configs/classification/IBD/CNN/exp_03_cnn_kfold_test_pipeline.yaml"

# --- ESECUZIONE NOTEBOOK ---
NOTEBOOK_PATH="notebooks/03_classification.ipynb"
OUTPUT_DIR="experiments/classification/IBD/CNN/exp_03_cnn_kfold_test_pipeline"
OUTPUT_NOTEBOOK_NAME="notebook_eseguito.ipynb"

mkdir -p $OUTPUT_DIR

echo "Esecuzione del notebook: $NOTEBOOK_PATH"
echo "Salvataggio dell'output in: $OUTPUT_DIR/$OUTPUT_NOTEBOOK_NAME"

srun python3 -m jupyter nbconvert --execute "$NOTEBOOK_PATH" \
    --to notebook \
    --output "$OUTPUT_NOTEBOOK_NAME" \
    --output-dir "$OUTPUT_DIR"

echo "✅ Esperimento 3 completato!."
