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

## 🔍 Explainability (XAI)
Per garantire la trasparenza scientifica del progetto, il notebook `04_explainability_xai.ipynb` è dedicato all'estrazione delle interpretazioni visive. Caricando i pesi pre-addestrati da `experiments/` ed eseguendo i moduli in `src/xai/`, genererà e salverà le mappe di attenzione/attivazione in cartelle di confronto dedicate sotto `experiments/explainability/`.