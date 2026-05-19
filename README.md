# 🔬 SpectraBreast: Spectral Analysis & Classification con Deep Learning

Questo progetto esplora l'uso di architetture Deep Learning (CNN e Masked Autoencoders/Transformers) per il pre-training, la classificazione e l'analisi dell'explainability (XAI) su dataset di spettri 1D (es. spettroscopia Raman).

Il progetto implementa funzionalità **State of the Art (SOTA)** per l'analisi di dati spettroscopici scarsi (come l'augmentation dinamica) e supporta la ricerca automatizzata degli iperparametri tramite Optuna.

---

## 🏗️ Struttura Simmetrica del Progetto

L'intera repository è pensata per essere eseguita su cluster HPC (via Slurm) o su Google Colab tramite VS Code. 
Il cuore organizzativo si basa sulla **simmetria esatta** tra la cartella `configs/` (le intenzioni) e la cartella `experiments/` (i risultati). Questo garantisce un tracciamento perfetto per ogni combinazione di Task e Dataset.

```text
SpectraBreast/
├── configs/                   # ⚙️ RICETTARIO: Parametri per ogni singolo esperimento
├── data/                      # 📊 DATI: Dataset processati
├── src/                       # 🧠 CODICE SORGENTE: Moduli riutilizzabili (Mai duplicati)
│   ├── models/                # Architetture (CNN, Transformers, HybridMAE)
│   ├── data/                  # Dataloader e SOTA Augmentation (RamanDataset)
│   ├── xai/                   # Explainability (GradCAM, Attention Maps)
│   └── tuning/                # Pipeline di ricerca iperparametri (Optuna)
├── notebooks/                 # 📓 ESECUTORI: Notebook generici
└── experiments/               # 💾 SCONTRINI: Output salvati in automatico
```

---

## 🧬 Tecniche SOTA Implementate

Per combattere l'overfitting su dataset biomedici piccoli (es. 200-500 campioni), il framework utilizza:
1. **Data Augmentation Dinamica (`src/data/augmentation.py`)**: 
   - *Gaussian Noise*: Simula lo shot-noise del detector.
   - *Amplitude Scaling*: Simula variazioni di potenza laser.
   - *Baseline Shifting*: Simula fluttuazioni del background di fluorescenza.
2. **Masked Autoencoders (MAE)**: Pre-training self-supervised ad alta capacità per estrarre le leggi fisiche dei segnali (SpectraMAE e HybridMAE).
3. **Linear Probing & Fine-Tuning**: Approccio in due stadi con congelamento progressivo dei pesi pre-addestrati e learning rates microscopici per prevenire il Catastrophic Forgetting.

---

## 🚀 Guida all'Uso e Lancio degli Esperimenti (Walkthrough)

La pipeline supporta 3 modelli principali: **CNN (da zero)**, **Transformer Puro (Pre-Train + Finetune)** e **Hybrid (Pre-Train + Finetune)**. Abbiamo semplificato la gestione nascondendo parametri complessi nel codice e lasciando nei file YAML solo l'essenziale.

### 1. Modello CNN (Train from Scratch)
La CNN non usa il pre-training. Viene addestrata da zero direttamente sul dataset. È l'unica in cui Optuna ottimizza anche l'architettura.

**Template YAML (es. `configs/classification/IBD/CNN/exp_03_cnn_final.yaml`):**
```yaml
experiment_name: "CNN_Final_Optimized"
dataset:
  path: "data/processed/IBD/SUPER_IBD_500pt_15G.mat"
  name: "IBD"
  batch_size: 64

model:
  architecture: "RamanCNN"
  input_length: 500
  n_classes: 2
  # Parametri architettura (fissi se trovati da Optuna)
  num_layers: 4
  base_filters: 64
  kernel_size: 7
  dropout: 0.42

training:
  mode: "scratch"
  epochs: 100
  learning_rate: 7.68e-05
  patience: 20
  pos_weight_multiplier: 2.58
```

**Comandi di lancio:**
1. *Ricerca Optuna:* `sbatch run_final_optuna.sh CNN IBD 50`
2. *Addestramento Finale:* `sbatch run_classification.sh configs/classification/IBD/CNN/exp_03_cnn_final.yaml`

---

### 2. Modello Transformer Puro (Pre-Train + Fine-Tune)
Il Transformer impara la rappresentazione dal dataset senza etichette, per poi specializzarsi.

**Step 1: Pre-Training**
```bash
sbatch run_pretraining.sh configs/pretraining/exp_01_smae_baseline.yaml
```

**Step 2: Fine-Tuning (Template YAML pulito)**
La struttura `training` ha i Learning Rate divisi per Linear Probe (`lr_lp`) e Fine-Tune (`lr_ft`).

```yaml
experiment_name: "Transformer_FineTune"
dataset:
  path: "data/processed/IBD/SUPER_IBD_500pt_15G.mat"
  name: "IBD"
  batch_size: 64

pretrain:
  pretrained_path: "experiments/pretraining/exp_01_smae_baseline/best_pretrained_model.pth"

model:
  architecture: "ViT_1D"
  # DEVONO essere identici al file di Pre-Training!
  patch_size: 20
  embedding_dim: 256
  depth: 8
  heads: 8

training:
  mode: "finetune"
  lr_lp: 0.005         # Learning rate fase 1 (Testa)
  lr_ft: 1.5e-05       # Learning rate fase 2 (Microscopico per sblocco)
  patience: 16
  pos_weight_multiplier: 2.48
```

**Comandi di lancio Fine-Tuning:**
1. *Ricerca Optuna:* `sbatch run_final_optuna.sh Transformer IBD 50 configs/classification/IBD/Transformer/fine-tune/exp_transformer_finetune.yaml`
2. *Addestramento Finale:* `sbatch run_classification_finetune.sh configs/classification/IBD/Transformer/fine-tune/exp_transformer_finetune.yaml`

---

### 3. Modello Hybrid (Pre-Train + Fine-Tune)
L'Hybrid è identico al Transformer come logica di lancio, ma unisce la potenza delle CNN (estrazione locale) all'Attention (contesto globale).

**Step 1: Pre-Training**
```bash
sbatch run_pretraining.sh configs/pretraining/exp_03_hybrid_mae_optimal.yaml
```

**Step 2: Fine-Tuning (Template YAML pulito)**
```yaml
experiment_name: "Hybrid_FineTune"
dataset:
  path: "data/processed/IBD/SUPER_IBD_500pt_15G.mat"
  name: "IBD"
  batch_size: 64

pretrain:
  pretrained_path: "experiments/pretraining/exp_03_hybrid_mae_optimal/best_pretrained_model.pth"

model:
  architecture: "HybridCNNTransformer"
  # DEVONO essere identici al pre-training!
  d_model: 64
  nhead: 2
  num_layers: 2
  dropout: 0.37

training:
  mode: "finetune"
  lr_lp: 0.00057
  lr_ft: 1.41e-06
  patience: 16
  pos_weight_multiplier: 2.48
```

**Comandi di lancio Fine-Tuning:**
1. *Ricerca Optuna:* `sbatch run_final_optuna.sh Hybrid IBD 50 configs/classification/IBD/Hybrid/exp_04_hybrid_finetune_final.yaml`
2. *Addestramento Finale:* `sbatch run_classification_finetune.sh configs/classification/IBD/Hybrid/exp_04_hybrid_finetune_final.yaml`

---

## 🔍 Explainability (XAI)
Il notebook `04_explainability_xai.ipynb` è dedicato all'estrazione delle interpretazioni visive. Genera e salva mappe di attenzione/attivazione in cartelle di confronto dedicate.
- Se l'esperimento era **LOMO / K-Fold**, aggregherà globalmente i gradienti tra i fold per generare un profilo medio robusto e di validità clinica superiore.
