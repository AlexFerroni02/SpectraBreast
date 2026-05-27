#!/bin/bash
#SBATCH --job-name=ramanfound_pretrain
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G

# ===========================================================
#  RamanFoundation Pre-training su HPC
#  USO:  sbatch run_ramanfoundation_pretrain.sh [EPOCHS]
#  Es.:  sbatch run_ramanfoundation_pretrain.sh 1     # test
#        sbatch run_ramanfoundation_pretrain.sh 200   # full
# ===========================================================

# --- AMBIENTE HPC ---
module purge
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

# Rimuovi ~/.local/bin da PATH per evitare conflitti con Python 3.9
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.local/bin" | tr '\n' ':' | sed 's/:$//')
# Assicurati che il python3 del modulo sia nel PATH
PYTHON_BIN=$(dirname $(which python3))
export PATH="$PYTHON_BIN:$PATH"

# --- PARAMETRI ---
EPOCHS=${1:-1}
OUTPUT_DIR="experiments/ramanfoundation_pretrain/1000pt/FullRange/Min-Max/exp_4"
DATA_PATH="data/PreTrain/1000_pt/FullRange/Min-Max/SUPER_PRETRAIN_1000pt.npz"

# --- SETUP ---
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  RamanFoundation Pre-training"
echo "=========================================="
echo "  Epochs:           $EPOCHS"
echo "  Data Path:        $DATA_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Python:           $(python3 --version 2>&1)"
echo "=========================================="

# Verifica GPU
python3 -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'GPU rilevate: {len(gpus)}'); [print(f'  {g}') for g in gpus]" 2>/dev/null

# --- STEP 1: Prepara il notebook per HPC ---
echo ""
echo ">>> Step 1: Preparazione notebook..."

export PRETRAIN_EPOCHS="$EPOCHS"
export OUTPUT_DIR="$OUTPUT_DIR"
export DATA_PATH="$DATA_PATH"



# --- STEP 2: Esegui il notebook ---
NOTEBOOK_IN="pretraining_ramanfoundation_hpc.ipynb"

echo ""
echo ">>> Step 2: Esecuzione notebook..."

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
