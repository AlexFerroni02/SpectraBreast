#!/bin/bash
#SBATCH --job-name=smae_pretrain
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G

# ===========================================================
#  SMAE Pre-training su HPC (PyTorch)
#  USO:  sbatch run_smae_pretrain.sh [EPOCHS]
#  Es.:  sbatch run_smae_pretrain.sh 1     # test
#        sbatch run_smae_pretrain.sh 200   # full
# ===========================================================

# --- AMBIENTE HPC ---
module purge
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

# Aggiungi pacchetti utente (~/.local) al PYTHONPATH per nbconvert
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.local/bin" | tr '\n' ':' | sed 's/:$//')
PYTHON_BIN=$(dirname $(which python3))
export PATH="$PYTHON_BIN:$PATH"

# --- PARAMETRI ---
EPOCHS=${1:-1}
OUTPUT_DIR="experiments/smae_pretrain/1000_pt/FingerPrint/Min-Max/exp_5"
DATA_PATH="data/PreTrain/1000_pt/FingerPrint/Min-Max/SUPER_PRETRAIN_1000pt.npz"

# --- SETUP ---
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  SMAE Pre-training"
echo "=========================================="
echo "  Epochs:           $EPOCHS"
echo "  Data Path:        $DATA_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Python:           $(python3 --version 2>&1)"
echo "=========================================="

# Verifica GPU (PyTorch)
python3 -c "import torch; print(f'GPU rilevate: {torch.cuda.device_count()}'); [print(f'  {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null

# --- ESECUZIONE DEL NOTEBOOK ---
NOTEBOOK_IN="pretrain_03_SMAE.ipynb"

export EPOCHS="$EPOCHS"
export OUTPUT_DIR="$OUTPUT_DIR"
export DATA_PATH="$DATA_PATH"

echo ""
echo ">>> Esecuzione notebook..."

srun python3 -m nbconvert --execute "$NOTEBOOK_IN" \
    --to notebook \
    --output "notebook_eseguito.ipynb" \
    --output-dir "$OUTPUT_DIR" \
    --ExecutePreprocessor.timeout=-1

if [ $? -ne 0 ]; then
    echo "ERRORE: Esecuzione notebook fallita!"
    exit 1
fi


echo ""
echo "=========================================="
echo "  Pre-training completato!"
echo "  Output in: $OUTPUT_DIR"
echo "=========================================="
