#!/bin/bash

# Abilita il fail on error
set -e

echo "=============================================="
echo " Avvio Ricerca Optuna: Transformer (LP + FT)  "
echo " Dataset: IBD                                 "
echo "=============================================="

# Numero di trial che vuoi provare (50 è un buon compromesso per iniziare)
N_TRIALS=50

# Lancia lo script Python
python src/tuning/optuna_final_search.py \
    --model Transformer \
    --dataset IBD \
    --pretrain False \
    --n-trials $N_TRIALS

echo "=============================================="
echo " Ricerca completata con successo!             "
echo " Copia i parametri finali nel tuo file .yaml  "
echo "=============================================="