# 🔬 SpectraBreast: Self-Supervised Learning for 1D Raman Spectroscopy Analysis & Classification

[![Python Version](https://img.shields.shields.shields.shields.shields.shields.io/badge/Python-3.9+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.shields.shields.shields.shields.shields.io/badge/PyTorch-1.10+-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.shields.shields.shields.shields.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![HPC Compatible](https://img.shields.shields.shields.shields.shields.shields.io/badge/HPC-Slurm-007A87.svg?style=flat-square)](https://slurm.schedmd.com/)
[![License: MIT](https://img.shields.shields.shields.shields.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**SpectraBreast** is an advanced machine learning framework designed for the preprocessing, self-supervised pretraining (SSL), and downstream classification of **1D Raman Spectroscopy signals**. 

Raman spectroscopy is a powerful technique for biological and clinical diagnostics (such as tissue cancer mapping and bacterial classification). However, model training often suffers from data scarcity and domain shifts. This repository addresses these challenges by using **Self-Supervised Learning (SSL)** to learn robust spectral representations from large, unlabeled, multi-source datasets before fine-tuning them on specific, labeled downstream clinical tasks.

---

## 🚀 Key Framework Features

* **🔬 State-of-the-Art (SOTA) 1D Signal Preprocessing**: Includes vectorized cosmic ray removal, Savitzky-Golay smoothing, Asymmetric Least Squares (ALS) fluorescence baseline correction, and L2 intensity normalization.
* **🌐 Multi-Source Dataset Merging**: Processes and merges fingerprints from 5 different Raman spectroscopy domains (*Trophy, DeepR, Cells, Twist, Immuno*) to build a robust pretraining cohort.
* **🧠 Dual Self-Supervised Learning (SSL) Paradigms**:
  1. **RamanFoundation (Barlow Twins)**: A self-supervised joint-embedding method implemented in **TensorFlow/Keras** that minimizes the cross-correlation redundancy of augmented views.
  2. **SpectraMAE (Masked Autoencoders for 1D)**: A transformer-based reconstructive framework implemented in **PyTorch** that masks up to 75% of spectral patches and reconstructs them using a ViT-1D backbone.
* **📈 Clinical Downstream Fine-Tuning**: Validated on major diagnostics tasks:
  * **Bacteria Classification**
  * **IBD (Inflammatory Bowel Disease) Detection**
  * **SpectraBreast** Breast Tumor Mapping
* **🖥️ HPC Ready**: Built-in Slurm launcher scripts (`.sh`) utilizing `nbconvert` to execute training pipelines on GPU-enabled supercomputing clusters.

---

## 🏗️ Repository Structure

```text
SpectraBreast/
├── src/                       # 🧠 Core Architecture Source Code
│   └── models/                
│       ├── transformer/       # PyTorch ViT-1D and SpectraMAE definitions
│       │   ├── Spectra_MAE.py
│       │   └── ViT_1D.py
│       └── factory.py         # Model Factory to load architectures from configs
│
├── PreProcess/                # 📊 Data Preparation and Merging
│   └── FingerPrint/           # Dataset-specific preprocessing notebooks
│       ├── preprocessing_SpectraBreast.ipynb
│       ├── preprocessing_IBD_FingerPrint.ipynb
│       ├── preprocessing_Cell_FingerPrint.ipynb
│       ├── Merging_dataset_fingerprint.ipynb # Merges datasets for pretraining
│       └── ...
│
├── PreTrain/                  # 🎓 Self-Supervised Pretraining Pipelines
│   ├── MAE/                   # PyTorch Masked Autoencoder experiments (exp_1 to exp_9)
│   └── BarlowTwins/           # TensorFlow Barlow Twins (RamanFoundation) experiments
│
├── FineTuning/                # 🎯 Downstream Target Task Fine-Tuning
│   ├── MAE/                   # Fine-tuning PyTorch SpectraMAE on IBD and Bacteria
│   │   ├── Bacteria/
│   │   └── IBD/
│   └── BarlowTwins/           # Fine-tuning Keras Barlow Twins on IBD and Bacteria
│
├── notebooks/                 # 📓 Template Execution Notebooks
│   ├── pretrain_03_SMAE.ipynb
│   └── pretraining_ramanfoundation_hpc.ipynb
│
├── run_smae_pretrain.sh       # 🖥️ HPC Slurm script for PyTorch SpectraMAE
└── run_ramanfoundation_pretrain.sh # 🖥️ HPC Slurm script for Keras Barlow Twins
```

---

## 📊 Preprocessing Pipeline

Raw Raman spectroscopy signals are heavily contaminated by laser fluorescence and system noise. The preprocessing pipeline implemented in the `PreProcess/` notebooks cleans and normalizes the signals:

1. **Cosmic Ray Removal**: Utilizes a fast vectorized median filter and modified Z-score thresholds to detect and clip high-intensity spikes.
2. **Savitzky-Golay Smoothing**: Filters out high-frequency shot noise while preserving crucial chemical peak shapes.
3. **Fluorescence Baseline Correction**: Employs **Asymmetric Least Squares (ALS)** to estimate and subtract the broad background fluorescence profile.
4. **L2 Normalization**: Scales individual spectra so that their total energy is normalized, minimizing laser power fluctuations.
5. **Spectral Interpolation & Cropping**: Standardizes the Raman shift frequencies to a uniform length (typically 500 or 1000 spectral points).

---

## 🧠 Self-Supervised Pretraining Methods

### 1. SpectraMAE (Masked Autoencoder) - PyTorch
The **SpectraMAE** model divides the 1D spectrum into a sequence of non-overlapping patches (e.g., length 50). It then:
* Randomly masks out a high ratio (typically **75%**) of the patches.
* Passes only the remaining **25%** visible patches through a **1D Vision Transformer (ViT)** encoder.
* Appends learnable mask tokens to the encoded patches and reconstructs the original spectrum using a shallow Transformer decoder.
* **Loss function**: Mean Squared Error (MSE) computed exclusively on the masked patches.

```python
# Model initialization via model factory
from src.models.factory import build_model_from_config

config = {
    "model": {
        "architecture": "Spectra_MAE",
        "parameters": {
            "sequence_length": 1000,
            "patch_size": 50,
            "embedding_dim": 256,
            "encoder_depth": 8,
            "encoder_heads": 8,
            "decoder_dim": 128,
            "decoder_depth": 2,
            "decoder_heads": 8,
            "mask_ratio": 0.75
        }
    }
}
model = build_model_from_config(config, device='cuda')
```

### 2. RamanFoundation (Barlow Twins) - TensorFlow/Keras
The **RamanFoundation** architecture applies a joint-embedding self-supervised learning strategy:
* Generates two distorted/augmented versions of each spectrum (using Gaussian noise, amplitude scaling, and baseline shifting).
* Passes both views through a 1D Convolutional/dense encoder network to extract projection embeddings.
* **Barlow Twins Loss**: Minimizes the difference between the cross-correlation matrix of the twin embeddings and the identity matrix, forcing the network to learn invariant features while preventing representation collapse.

---

## 🖥️ Running on HPC (Slurm)

To run pretraining workloads on an HPC cluster, use the provided Slurm batch scripts. These scripts load the appropriate modules, configure the PyTorch/TensorFlow execution environments, and run the target notebooks using `jupyter nbconvert`.

### Launch PyTorch SpectraMAE Pretraining:
```bash
sbatch run_smae_pretrain.sh <EPOCHS>
```
*Outputs: Saved model checkpoints (`smae_pretrained_model.pth`), metric plots, and the fully-executed notebook will be saved under the designated experiment folder (e.g., `experiments/smae_pretrain/`).*

### Launch TensorFlow RamanFoundation (Barlow Twins) Pretraining:
```bash
sbatch run_ramanfoundation_pretrain.sh <EPOCHS>
```
*Outputs: Saved Keras weights (`.h5` and TensorFlow checkpoint weights), loss logs, and execution notebook saved under `experiments/ramanfoundation_pretrain/`.*

---

## 🎯 Fine-Tuning on Downstream Tasks

After self-supervised pretraining is complete, the learned representations (the encoder backbones) are loaded for downstream classification tasks.

The notebooks in `FineTuning/` support:
* **Linear Probing**: Freezing the encoder backbone and training only the final linear classifier classification head.
* **End-to-End Fine-Tuning**: Training the entire network with differential learning rates (low learning rates on the encoder, higher rates on the classifier) to prevent catastrophic forgetting.

### Results & Performance
Models pretrained with **Barlow Twins (RamanFoundation)** or **SpectraMAE** show significantly faster convergence, higher classification accuracy (F1-score), and increased robustness on small datasets (like the 500-sample IBD and Bacteria classification tests) compared to models trained from scratch.

---

## 🔍 Explainable AI (XAI) & Interpretability

To translate deep learning predictions into actionable clinical findings, the framework implements advanced feature attribution and interpretability algorithms (as demonstrated in downstream notebooks like `FineTuning/MAE/IBD/SMAE__Fine_tuning_IBD_1000pt_exp_9.ipynb`):

* **Integrated Gradients (IG)**: Computes the path integral of the gradients of the model's output with respect to the input spectrum along a straight path from a baseline reference. This maps the model's predictions to specific chemical peaks (Raman shifts) to identify healthy vs. disease biomarkers.
* **SHAP (Shapley Additive exPlanations)**: Utilizes `shap.GradientExplainer` (tailored for differentiable PyTorch models) to compute local and global feature attribution values. This allows visualizing which spectral bands contribute positively or negatively to class predictions (e.g., distinguishing between inflammatory bowel disease and healthy controls).

These explainability techniques build trust in model decisions and facilitate clinical validation.

---

## 🛠️ Requirements & Setup

To replicate and run this framework, ensure you have the following environments configured:

* **Python**: 3.9+
* **Deep Learning Frameworks**: PyTorch 1.10+ (CUDA supported) and/or TensorFlow 2.15+
* **Core Libraries**: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `h5py`, `tqdm`, `nbconvert`, `shap`

If deploying on Slurm HPC clusters, make sure to load the modules matching the script requirements, e.g.:
```bash
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
# or
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
```
