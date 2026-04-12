#!/bin/bash
#SBATCH --job-name=spectrabreast_run
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=04:00:00
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
# USO: sbatch run_classification.sh configs/classification/IBD/CNN/exp_02_cnn_kfold.yaml
CONFIG_FILE=${1:-configs/classification/IBD/CNN/exp_01_cnn_baseline.yaml}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERRORE: File YAML non trovato: $CONFIG_FILE"
    exit 1
fi

export EXP_CONFIG_FILE="$CONFIG_FILE"

# Genera cartella output speculare
OUTPUT_DIR="${CONFIG_FILE/configs/experiments}" 
OUTPUT_DIR="${OUTPUT_DIR%.yaml}"
mkdir -p "$OUTPUT_DIR"

NOTEBOOK_IN="notebooks/03_classification.ipynb"
NOTEBOOK_OUT="notebook_eseguito.ipynb"

echo "=========================================="
echo "Avvio Esperimento: $CONFIG_FILE"
echo "Output Directory:  $OUTPUT_DIR"
echo "=========================================="

python3 -m pip install --user --upgrade typing-extensions referencing jsonschema nbconvert scikit-learn seaborn h5py pyyaml jupyter

srun python3 -m jupyter nbconvert --execute "$NOTEBOOK_IN" \
    --to notebook \
    --output "$NOTEBOOK_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --ExecutePreprocessor.timeout=-1

echo "✅ Esperimento classificativo completato."
