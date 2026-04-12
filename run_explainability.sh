#!/bin/bash
#SBATCH --job-name=spectrabreast_xai
#SBATCH --output=job_xai_%j.out
#SBATCH --error=job_xai_%j.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8G

# --- AMBIENTE HPC ---
module purge
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# Verifica argomenti input. 
# USO: sbatch run_explainability.sh configs/classification/IBD/CNN/exp_01_cnn_baseline.yaml
CONFIG_FILE=${1:-configs/classification/IBD/CNN/exp_01_cnn_baseline.yaml}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERRORE: File YAML non trovato: $CONFIG_FILE"
    exit 1
fi

export EXP_CONFIG_FILE="$CONFIG_FILE"

# Genera cartella output speculare (gemella dell'esperimento)
OUTPUT_DIR="${CONFIG_FILE/configs/experiments}" 
OUTPUT_DIR="${OUTPUT_DIR%.yaml}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERRORE: Non trovo la cartella esperimento in $OUTPUT_DIR. Fai prima il training!"
    exit 1
fi

NOTEBOOK_IN="notebooks/04_explainability_xai.ipynb"
NOTEBOOK_OUT="notebook_xai_eseguito.ipynb"

echo "=========================================="
echo "Avvio Explainability XAI su : $CONFIG_FILE"
echo "Output Directory esp        : $OUTPUT_DIR"
echo "=========================================="

python3 -m pip install --user --upgrade captum typing-extensions nbconvert

srun python3 -m jupyter nbconvert --execute "$NOTEBOOK_IN" \
    --to notebook \
    --output "$NOTEBOOK_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --ExecutePreprocessor.timeout=-1

echo "✅ Explainability completata."
