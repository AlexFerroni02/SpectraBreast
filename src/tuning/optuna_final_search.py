"""
Optuna Hyperparameter Search — Unified for CNN, Transformer, and Hybrid.

Strategy:
  - CNN:         Optuna varies BOTH architecture and training hyperparameters (no pre-training).
  - Transformer: Architecture is FIXED from the YAML config (must match pre-training).
                 Optuna varies ONLY fine-tuning hyperparameters.
  - Hybrid:      Same as Transformer.

Usage:
  python -m src.tuning.optuna_final_search --model CNN --dataset IBD --n-trials 100
  python -m src.tuning.optuna_final_search --model Transformer --dataset IBD --config configs/classification/IBD/Transformer/fine-tune/exp_02_smae.yaml --n-trials 50
  python -m src.tuning.optuna_final_search --model Hybrid --dataset IBD --config configs/classification/IBD/Hybrid/exp_01_hybrid_finetune.yaml --n-trials 50
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from src.data.augmentation import RamanDataset
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src.data.mat_loader import load_mat_dataset
from src.data.splits import make_kfold_splits, make_lomo_folds
from src.models.factory import build_model_from_config
from src.engine.finetune import train_finetune, train_linear_probe


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna HPO Search (K-Fold)")
    parser.add_argument("--model", type=str, required=True, choices=["CNN", "Transformer", "Hybrid"],
                        help="Model type to optimize")
    parser.add_argument("--dataset", type=str, required=True, choices=["IBD", "TROPHY"],
                        help="Dataset to use")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (required for Transformer/Hybrid to load fixed architecture + pretrained_path)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of K-Fold splits")
    return parser.parse_args()


def load_yaml_config(config_path):
    """Load and return YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_info(dataset_name):
    """Return dataset path and key names based on dataset name."""
    if dataset_name == "IBD":
        return "data/IBD/Processed/IBD_Dataset_500pt.mat", "X_processed", "labels_binary"
    else:
        return "data/Trophy/TROPHY_500pt_Labeled.mat", "X_trophy_500", "Ytrophy_binary"


def run_final_optuna_search():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate arguments
    if args.model in ("Transformer", "Hybrid") and args.config is None:
        raise ValueError(
            f"--config is required for {args.model}. "
            f"Provide the YAML config that specifies the architecture and pretrained_path."
        )

    # Load YAML config if provided (for Transformer/Hybrid)
    yaml_config = None
    pretrained_path = None
    if args.config:
        yaml_config = load_yaml_config(args.config)
        pretrained_path = yaml_config.get("pretrain", {}).get("pretrained_path", None)

    print(f"\n{'='*60}")
    print(f"🔍 OPTUNA HPO SEARCH (K-Fold)")
    print(f"{'='*60}")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Trials:   {args.n_trials}")
    print(f"  K-Folds:  {args.n_splits}")
    print(f"  Device:   {device}")
    if pretrained_path:
        print(f"  Pretrained: {pretrained_path}")
    print(f"{'='*60}\n")

    # 1. Load Dataset
    dataset_path, x_key, y_key = get_dataset_info(args.dataset)
    dataset = load_mat_dataset(dataset_path, x_key=x_key, y_key=y_key, dataset_name=args.dataset.lower())
    X_full, Y_full = dataset.X, dataset.y
    groups = dataset.groups

    if X_full.shape[0] == 500 and len(Y_full) != 500:
        X_full = X_full.T

    N, L = X_full.shape
    tensor_x = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1)
    tensor_y = torch.tensor(Y_full, dtype=torch.long)
    n_classes = len(np.unique(Y_full))

    # 2. Create Folds
    if groups is not None and len(groups) > 0 and args.dataset == "TROPHY":
        folds = make_lomo_folds(Y_full, groups, seed=42)
    else:
        folds = make_kfold_splits(Y_full, seed=42, n_splits=args.n_splits)

    # 3. Optuna DB
    base_dir = os.path.abspath("experiments/optuna")
    os.makedirs(base_dir, exist_ok=True)
    study_name = f"study_FINAL_{args.model.lower()}_{args.dataset.lower()}"
    if pretrained_path:
        study_name += "_finetune_robust_v3"
    db_path = f"sqlite:///{os.path.join(base_dir, study_name + '.db')}"

    # ------------------------------------------
    # OPTUNA OBJECTIVE
    # ------------------------------------------
    def objective(trial):
        torch.manual_seed(42 + trial.number)
        np.random.seed(42 + trial.number)

        # --- COMMON TRAINING HYPERPARAMETERS ---
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        weight_decay = 0.01 # Default, overridden below
        patience = 20 # Fixed early stopping
        scheduler_patience = trial.suggest_int("scheduler_patience", 3, 8)
        scheduler_factor = 0.5
        pos_weight_multiplier = trial.suggest_float("pos_weight_multiplier", 1.0, 3.0)
        grad_clip = 1.0 # Default, overridden below

        # --- BUILD CONFIG BASED ON MODEL TYPE ---
        config = {
            "model": {"input_length": L, "n_classes": n_classes},
            "training": {
                "weight_decay": weight_decay, 
                "patience": patience,
                "scheduler_patience": scheduler_patience,
                "scheduler_factor": scheduler_factor
            }
        }

        if args.model == "CNN":
            weight_decay = 0.01
            grad_clip = 1.0
            config["training"]["weight_decay"] = weight_decay
            # CNN: Optuna varies ARCHITECTURE + training params
            lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
            epochs = trial.suggest_int("epochs", 30, 100)
            config["model"].update({
                "architecture": "RamanCNN",
                "num_layers": trial.suggest_int("num_layers", 2, 5),
                "base_filters": trial.suggest_categorical("base_filters", [16, 32, 64]),
                "dropout": trial.suggest_float("dropout", 0.2, 0.6),
                "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            })
            head_attr = None
            mode = "scratch"
            epochs_lp = 0
            lr_lp = 0
            lr_ft = lr
            epochs_ft = epochs

        elif args.model == "Transformer":
            weight_decay = 0.01
            grad_clip = 1.0
            config["training"]["weight_decay"] = weight_decay
            # Transformer: Architecture FIXED from YAML, Optuna varies ONLY training params
            model_cfg = yaml_config.get("model", {})
            config["model"].update({
                "architecture": model_cfg.get("architecture", "ViT_1D"),
                "patch_size": int(model_cfg.get("patch_size", 20)),
                "embedding_dim": int(model_cfg.get("embedding_dim", 256)),
                "dim_mlp": int(model_cfg.get("dim_mlp", 512)),
                "depth": int(model_cfg.get("depth", 8)),
                "heads": int(model_cfg.get("heads", 8)),
                "dropout": trial.suggest_float("dropout", 0.05, 0.5),
            })
            head_attr = yaml_config.get("training", {}).get("head_attr", "classifier")
            mode = "finetune"
            lr_lp = trial.suggest_float("lr_lp", 1e-4, 1e-2, log=True)
            epochs_lp = 50
            lr_ft = trial.suggest_float("lr_ft", 1e-6, 5e-5, log=True)
            epochs_ft = 65

        elif args.model == "Hybrid":
            model_cfg = yaml_config.get("model", {})
            config["model"].update({
                "architecture": model_cfg.get("architecture", "HybridCNNTransformer"),
                "input_length": int(model_cfg.get("input_length", 500)),
                "n_classes": n_classes
            })
            head_attr = yaml_config.get("training", {}).get("head_attr", "classifier")
            
            if pretrained_path:
                # Modalità Finetune
                mode = "finetune"
                config["model"].update({
                    "d_model": int(model_cfg.get("d_model", 64)),
                    "nhead": int(model_cfg.get("nhead", 4)),
                    "num_layers": int(model_cfg.get("num_layers", 2)),
                    "dropout": float(model_cfg.get("dropout", 0.1)),
                })
                
                # --- STRATEGIA STABILE PER FINETUNING ---
                # Fissiamo regolarizzatori per ridurre la varianza
                grad_clip = 1.0
                weight_decay = 0.01
                config["training"]["weight_decay"] = weight_decay
                
                # Linear Probe prolungato e con LR alto per scaldare bene la testa
                lr_lp = trial.suggest_float("lr_lp", 5e-4, 5e-3, log=True)
                epochs_lp = 50
                
                # Fine-Tuning delicatissimo per non distruggere le feature fisiche MAE
                lr_ft = trial.suggest_float("lr_ft", 1e-6, 5e-5, log=True)
                epochs_ft = 65
            else:
                # Modalità Scratch: Ottimizziamo anche l'architettura
                weight_decay = 0.01
                grad_clip = 1.0
                config["training"]["weight_decay"] = weight_decay
                mode = "scratch"
                epochs_lp = 0
                lr_lp = 0
                lr_ft = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
                epochs_ft = trial.suggest_int("epochs", 30, 150)
                
                nhead = trial.suggest_categorical("nhead", [2, 4, 8])
                d_model = trial.suggest_categorical("d_model", [32, 64, 128])
                # Evitiamo errori Pytorch: d_model deve essere multiplo di nhead
                if d_model % nhead != 0:
                    d_model = (d_model // nhead) * nhead
                    if d_model == 0: d_model = nhead
                    
                config["model"].update({
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": trial.suggest_int("num_layers", 1, 4),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                })

        # --- K-FOLD LOOP ---
        fold_val_f1_scores = []

        for fold in folds:
            train_loader = DataLoader(
                RamanDataset(tensor_x[fold.idx_train], tensor_y[fold.idx_train], is_train=True),
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                RamanDataset(tensor_x[fold.idx_val], tensor_y[fold.idx_val], is_train=False),
                batch_size=batch_size, shuffle=False
            )

            # Build fresh model for each fold
            target_model = build_model_from_config(config, device)

            if mode == "finetune":
                # Load pretrained weights
                if pretrained_path and os.path.exists(pretrained_path):
                    checkpoint = torch.load(pretrained_path, map_location=device)
                    target_model.load_state_dict(checkpoint, strict=False)
                elif pretrained_path:
                    raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

                # Linear Probe phase
                train_linear_probe(
                    model=target_model,
                    train_loader=train_loader,
                    device=device,
                    y_train=Y_full[fold.idx_train],
                    head_attr=head_attr,
                    epochs=epochs_lp,
                    lr=lr_lp,
                    weight_decay=weight_decay,
                    grad_clip=grad_clip,
                    verbose=False,
                )

            # Full Fine-Tuning (or training from scratch for CNN)
            _, _, val_f1 = train_finetune(
                model=target_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                y_train=Y_full[fold.idx_train],
                epochs=epochs_ft,
                lr=lr_ft,
                weight_decay=weight_decay,
                patience=patience,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
                pos_weight_multiplier=pos_weight_multiplier,
                grad_clip=grad_clip,
                verbose=False,
            )

            fold_val_f1_scores.append(val_f1)

            # Optuna pruning
            trial.report(val_f1, fold.fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_f1 = float(np.mean(fold_val_f1_scores))
        return mean_f1

    # --- RUN STUDY ---
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.n_trials)

    # --- PRINT RESULTS ---
    print(f"\n{'='*60}")
    print(f"🏆 SEARCH COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best Mean K-Fold Macro F1: {study.best_trial.value:.4f}")
    print(f"\n  --- BEST HYPERPARAMETERS (copy to YAML) ---\n")

    # Print in YAML-friendly format
    print("training:")
    for key, value in sorted(study.best_trial.params.items()):
        if key in ("num_layers", "base_filters", "kernel_size", "d_model", "nhead", "dropout"):
            continue  # These go under model section
        if isinstance(value, float):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # Print model params for CNN and Hybrid scratch
    if args.model in ("CNN", "Hybrid"):
        print("\nmodel:")
        for key in ("num_layers", "base_filters", "kernel_size", "d_model", "nhead", "dropout"):
            if key in study.best_trial.params:
                print(f"  {key}: {study.best_trial.params[key]}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    run_final_optuna_search()