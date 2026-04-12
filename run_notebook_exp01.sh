#!/bin/bash
#SBATCH --job-name=exp01_cnn
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

# --- FORZIAMO PYTHON A USARE I PACCHETTI CORRETTI ---
# Il cluster bloccava l'aggiornamento di "typing_extensions" imponendo la sua versione vecchia.
# Inseriamo la tua cartella utente in CIMA ai percorsi di Python, così vince su tutto.
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

echo "Installazione dipendenze essenziali..."
python3 -m pip install --user --upgrade typing-extensions referencing jsonschema nbconvert
python3 -m pip install --user --no-cache-dir scikit-learn seaborn h5py pyyaml jupyter

# --- ESECUZIONE NOTEBOOK ---
# Definisce i percorsi in modo robusto
NOTEBOOK_PATH="notebooks/03_classification.ipynb"
OUTPUT_DIR="experiments/classification/IBD/CNN/exp_01_cnn_baseline"
OUTPUT_NOTEBOOK_NAME="notebook_eseguito.ipynb"

# Crea la cartella di output se non esiste
mkdir -p $OUTPUT_DIR

echo "Esecuzione del notebook: $NOTEBOOK_PATH"
echo "Salvataggio dell'output in: $OUTPUT_DIR/$OUTPUT_NOTEBOOK_NAME"

# Esegue il notebook e salva l'output direttamente nella cartella dell'esperimento
# Questo comando è più affidabile e non richiede modifiche al notebook.
# Non usiamo più 'cd notebooks' per rendere i percorsi più chiari.
srun python3 -m jupyter nbconvert --execute "$NOTEBOOK_PATH" \
    --to notebook \
    --output "$OUTPUT_NOTEBOOK_NAME" \
    --output-dir "$OUTPUT_DIR"

echo "✅ Esperimento completato. Notebook salvato in $OUTPUT_DIR."

