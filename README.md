# 🔬 Spectral Analysis & Classification con Deep Learning

Questo progetto esplora l'uso di architetture Deep Learning (CNN e Masked Autoencoders/Transformers) per il pre-training, la classificazione e l'analisi dell'explainability (XAI) su dataset di spettri (es. spettroscopia Raman, segnali 1D).

## 🏗️ Struttura Simmetrica del Progetto

L'intera repository è pensata per risiedere in locale sul PC ed essere eseguita su Google Colab tramite VS Code. 
Il cuore organizzativo si basa sulla **simmetria esatta** tra la cartella `configs/` (le intenzioni) e la cartella `experiments/` (i risultati). Questo garantisce un tracciamento perfetto per ogni combinazione di Task e Dataset.

```text
mio_progetto/
├── configs/                   # ⚙️ RICETTARIO: Parametri per ogni singolo esperimento
│   ├── pretraining/
│   │   ├── dataset_A/
│   │   │   ├── exp_01_mae_base.yaml
│   │   │   └── exp_02_mae_large.yaml
│   │   └── dataset_B/
│   └── classification/
│       └── dataset_A/
│           └── exp_01_resnet.yaml
│
├── data/                      # 📊 DATI: (Non versionati su Git se troppo pesanti)
│   ├── 01_raw/                # Dataset originali (sola lettura)
│   └── 02_processed/          # Spettri processati (baseline correction, smoothing)
│
├── src/                       # 🧠 CODICE SORGENTE: Moduli riutilizzabili (Mai duplicati)
│   ├── models/
│   │   ├── cnn/               # Modelli Convoluzionali (es. resnet_1d.py)
│   │   └── transformers/      # Modelli Attention-based (es. mae_spectra.py)
│   ├── data/
│   │   └── dataloaders.py     # Caricamento e augmentation
│   ├── xai/                   
│   │   ├── gradcam_1d.py      # Metodi XAI per feature map (CNN)
│   │   └── attention_maps.py  # Metodi XAI per mappe di attenzione (Transformers)
│   └── utils/
│       └── metrics.py         # Funzioni per calcolo metriche
│
├── notebooks/                 # 📓 ESECUTORI: Notebook generici da lanciare via Colab
│   ├── 01_data_exploration.ipynb
│   ├── 02_pretraining.ipynb
│   ├── 03_classification.ipynb
│   ├── 03b_classification_finetune.ipynb
│   └── 04_explainability_xai.ipynb
│
└── experiments/               # 💾 SCONTRINI: Output salvati in automatico (Lo Storico)
    ├── pretraining/
    │   └── dataset_A/      
    │       └── exp_01_mae_base/   # <-- Creata in automatico leggendo il config!
    │           ├── config_usato.yaml        # Copia di backup esatta del config
    │           ├── weights.pth              # Pesi finali del modello
    │           ├── metrics.json             # Risultati numerici
    │           └── notebook_eseguito.html   # Report visivo fisso (HTML del notebook)
    └── classification/
```

## 🚀 Flusso di Lavoro Ibrido (VS Code + Colab GPU)

1. **Sviluppo Locale**: Le architetture e le funzioni (XAI, DataLoader) vengono scritte nei file `.py` all'interno di `src/`.
2. **Configurazione (YAML)**: Per avviare un nuovo test, si crea o si modifica un file in `configs/task_name/dataset_name/nome_esperimento.yaml`.
3. **Training su Cloud**: 
   - Si apre il notebook (es. `03_classification.ipynb`) su VS Code locale.
   - Si connette il kernel a una GPU di Google Colab.
   - Si esegue il notebook passandogli il percorso del file `.yaml`.
4. **Automazione degli Artefatti**: 
   Il notebook, leggendo la stringa `"configs/classification/dataset_A/exp_01"`, genererà automaticamente la cartella gemella sotto `experiments/`. Lì salverà:
   - Una **copia esatta del file YAML** (`config_usato.yaml`) a garanzia di riproducibilità.
   - I pesi del modello addestrato.
   - Una **copia HTML del notebook eseguito** (`jupyter nbconvert`), utile come report visivo permanente per tesi e pubblicazioni.

## � Modalità di Validazione (Split Schemes)

Il progetto utilizza un singolo notebook unificato (`03_classification.ipynb`) capace di adattarsi dinamicamente alla strategia di validazione richiesta. È sufficiente specificare la modalità nel file YAML di configurazione sotto la chiave `split.scheme`:

- **Holdout (`scheme: "holdout"`)**:
  - Classico Train / Validation / Test split (es. 70/15/15).
  - Ideale per dataset con pazienti tutti indipendenti (es. IBD).
  - Salva i pesi (`best_weights.pth`) e il resoconto (`metrics.json`) direttamente nella root dell'esperimento.
- **Leave-One-Map-Out (`scheme: "lomo"`)**:
  - Estrae iterativamente una singola mappa intera come Test Set, usando il resto per Train/Val.
  - Ideale quando ci sono più spettri per un singolo paziente/mappa (es. TROPHY) per evitare data leakage.
  - Crea automaticamente delle sotto-cartelle per ogni fold (`experiments/.../folds/LOMO-M01/`) e aggrega i risultati finali globali.
- **K-Fold (`scheme: "kfold"`)**:
  - Stratified K-Fold Cross Validation. Mischia l'intero dataset e lo divide in $K$ fold.
  - Ideale per valutazioni statisticamente robuste su dataset standard senza struttura a mappe.
  - Genera sotto-cartelle (`folds/KFOLD-01/`) e calcola le metriche medie e aggregate esattamente come il LOMO.

**Come lanciare le varie modalità?**
Basta modificare la cella iniziale del notebook `03_classification.ipynb`:
```python
# Per lanciare un esperimento Holdout:
config_file = 'configs/classification/IBD/CNN/exp_01_cnn_baseline.yaml'

# Oppure, per lanciare un K-Fold:
# config_file = 'configs/classification/IBD/CNN/exp_02_cnn_kfold.yaml'

# Oppure, per lanciare un LOMO:
# config_file = 'configs/classification/Trophy/CNN/exp_01_cnn_lomo.yaml'
```
Tutto il resto verrà gestito automaticamente dalla pipeline.

## 🧾 Configurazione YAML (guida rapida)

### 1) Dataset e chiavi dei .mat
Per IBD puoi usare i default. Per TROPHY (o file con nomi variabili diversi) specifica le chiavi:
```yaml
dataset:
   name: "Trophy"
   path: "data/Trophy/TROPHY_500pt_Labeled.mat"
   batch_size: 32
   keys:
      x: "X_trophy_500"
      y: "Ytrophy_binary"
      groups: "MappeID"
```
Se preferisci, puoi usare anche `x_key`, `y_key`, `group_key`, `axis_key` al posto di `keys`.

### 2) Split (holdout / kfold / lomo)
```yaml
split:
   scheme: "kfold"        # holdout | kfold | lomo
   seed: 42
   holdout:
      test_size: 0.30
      val_size_of_temp: 0.50
   kfold:
      n_splits: 5
      val_size: 0.20
   lomo:
      val_size: 0.20
      skip_if_single_class_test: true
```

### 3) Training (scratch)
```yaml
training:
   epochs: 100
   learning_rate: 0.0005
   weight_decay: 0.001
   patience: 50
   scheduler_patience: 7
```

### 4) Training (linear probing + fine-tuning)
Usa il notebook `03b_classification_finetune.ipynb` e lo script dedicato.
```yaml
pretrain:
   pretrained_path: "/path/to/pretrained.pth"

training:
   mode: "finetune"        # scratch | linear_probe | finetune
   lp_epochs: 10
   ft_epochs: 40
   lr_lp: 1e-3
   lr_ft: 1e-4
   weight_decay: 1e-4
   patience: 15
   scheduler_patience: 7
   grad_clip: 1.0
   head_attr: "classifier"
```

### 5) XAI (pesi da usare)
```yaml
xai:
   weights_path: "/path/to/weights.pth"
```

## 🔍 Explainability (XAI)
Per garantire la trasparenza scientifica del progetto, il notebook `04_explainability_xai.ipynb` è dedicato all'estrazione delle interpretazioni visive. Caricando i pesi pre-addestrati da `experiments/` ed eseguendo i moduli in `src/xai/`, genererà e salverà le mappe di attenzione/attivazione in cartelle di confronto dedicate.

- Se l'esperimento originale era **Holdout**, calcolerà lo XAI su quel singolo Test Set.
- Se l'esperimento era **LOMO / K-Fold**, entrerà automaticamente in ogni singolo fold, estrarrà i gradienti, e li **aggregherà globalmente** per generare un profilo medio robusto tra tutti i pazienti/fold.

## 🧩 Architettura del Codice e Modularità (Il principio DRY)

Per evitare duplicazioni (codice "spaghetti") e facilitare la manutenzione, il codice sperimentale è separato dai notebook:

1. **`src/` (La Libreria)**: 
   - Contiene file come `splits.py` (con `make_holdout_split`, `make_kfold_splits`, `make_lomo_folds`) per centralizzare la logica matematica.
   - Contiene file come `trainer.py` e `plotting.py` (facoltativi) per spostare i complessi loop di addestramento e generazione grafici fuori dai notebook.
2. **Notebook Specializzati (`notebooks/`)**:
   - `03a_classification_baseline.ipynb`: Addestra modelli da zero (su IBD o TROPHY, a seconda del config YAML).
   - `03b_classification_finetune.ipynb`: (Opzionale) Addestra modelli caricando pesi pre-addestrati, gestendo logiche come Linear Probing e Fine-Tuning.

In questo modo, se vuoi cambiare dataset o passare da K-Fold a LOMO, **non crei un nuovo notebook**, ma crei semplicemente un nuovo file `.yaml`.
