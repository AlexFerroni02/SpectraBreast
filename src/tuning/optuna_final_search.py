import os
import sys
import argparse
import torch
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset

# Aggiungiamo root per gli import
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src.data.mat_loader import load_mat_dataset
from src.data.splits import make_kfold_splits, make_lomo_folds
from src.models.factory import build_model_from_config
from src.engine.pretrain import train_mae
from src.engine.finetune import train_finetune, train_linear_probe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["CNN", "Transformer", "Hybrid"])
    parser.add_argument("--dataset", type=str, required=True, choices=["IBD", "TROPHY"])
    parser.add_argument("--pretrain", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--n-trials", type=int, default=100, help="Numero di trial per la ricerca globale")
    parser.add_argument("--epochs-pretrain", type=int, default=30)
    parser.add_argument("--epochs-finetune", type=int, default=50)
    return parser.parse_args()

def run_final_optuna_search():
    args = parse_args()
    do_pretrain = args.pretrain == "True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== INIZIO RICERCA GLOBALE PARAMETRI DEFINITIVI ===")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Device: {device}")
    
    # 1. Caricamento di TUTTO il Dataset
    if args.dataset == "IBD":
        dataset_path = "data/IBD/Processed/IBD_Dataset_500pt.mat" 
        x_key, y_key = "X_processed", "labels_binary"
    else:
        dataset_path = "data/Trophy/TROPHY_500pt_Labeled.mat"
        x_key, y_key = "X_trophy_500", "Ytrophy_binary"
        
    dataset = load_mat_dataset(dataset_path, x_key=x_key, y_key=y_key, dataset_name=args.dataset.lower())
    X_full, Y_full = dataset.X, dataset.y
    groups = dataset.groups
    
    if X_full.shape[0] == 500 or (X_full.ndim == 2 and X_full.shape[1] > X_full.shape[0] and X_full.shape[0] < 1000):
        X_full = X_full.T
        
    N, L = X_full.shape
    tensor_x = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1)
    tensor_y = torch.tensor(Y_full, dtype=torch.long)
    
    # Usiamo TUTTI i dati per una Cross-Validation a 5 Fold
    if groups is not None and len(groups) > 0 and args.dataset == "TROPHY":
        folds = make_lomo_folds(Y_full, groups, seed=42)
    else:
        folds = make_kfold_splits(Y_full, seed=42, n_splits=5)
    
    # Crea cartella DB
    base_dir = os.path.abspath("experiments/optuna")
    os.makedirs(base_dir, exist_ok=True)
    study_name = f"study_FINAL_{args.model.lower()}_{args.dataset.lower()}_pretrain{do_pretrain}"
    db_path = f"sqlite:///{os.path.join(base_dir, study_name + '.db')}"

    # ------------------------------------------
    # OBIETTIVO OPTUNA (K-Fold CV sul 100% dei dati)
    # ------------------------------------------
    def objective(trial):
        torch.manual_seed(42 + trial.number)
        np.random.seed(42 + trial.number)
        
        # Campionamento Iperparametri (Stesso spazio di ricerca)
        if do_pretrain:
            lr_pretrain     = trial.suggest_float("lr_pretrain", 1e-4, 5e-3, log=True)
            epochs_pretrain = trial.suggest_int("epochs_pretrain", 20, 50)
            lr_probe        = trial.suggest_float("lr_probe", 1e-4, 5e-2, log=True)
            epochs_probe    = trial.suggest_categorical("epochs_probe", [5, 10, 15])
            lr_finetune     = trial.suggest_float("lr_finetune", 1e-5, 5e-3, log=True)
            epochs_finetune = trial.suggest_int("epochs_finetune", 30, 80)
        else:
            lr_finetune     = trial.suggest_float("lr_finetune", 1e-5, 5e-3, log=True)
            epochs_finetune = trial.suggest_int("epochs_finetune", 30, 80)

        weight_decay  = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128])
        patience           = trial.suggest_int("patience", 5, 20)
        scheduler_patience = trial.suggest_int("scheduler_patience", 2, 8)
        scheduler_factor   = trial.suggest_float("scheduler_factor", 0.1, 0.9)
        pos_weight_multiplier = trial.suggest_float("pos_weight_multiplier", 0.5, 3.0)

        config = {
            "model": { "input_length": L, "n_classes": len(np.unique(Y_full)) },
            "training": {
                "learning_rate": lr_pretrain if do_pretrain else lr_finetune,
                "weight_decay": weight_decay,
                "epochs": epochs_pretrain if do_pretrain else epochs_finetune,
                "patience": patience
            }
        }
        
        # Architettura CNN
        config["model"]["architecture"] = "RamanCNN"
        config["model"]["num_layers"]   = trial.suggest_int("num_layers",   2, 4)
        config["model"]["base_filters"] = trial.suggest_categorical("base_filters", [16, 32, 64])
        config["model"]["dropout"]      = trial.suggest_float("dropout", 0.2, 0.6)
        config["model"]["kernel_size"]  = trial.suggest_categorical("kernel_size", [3, 5, 7])

        fold_val_f1_scores = []
        
        # Per ogni configurazione, la valutiamo su tutti i 5 Fold
        for fold in folds:
            train_loader = DataLoader(TensorDataset(tensor_x[fold.idx_train], tensor_y[fold.idx_train]), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(tensor_x[fold.idx_val], tensor_y[fold.idx_val]), batch_size=batch_size, shuffle=False)
            
            target_model = build_model_from_config(config, device)
                
            _, _, val_f1 = train_finetune(
                model=target_model, train_loader=train_loader, val_loader=val_loader,
                device=device, y_train=Y_full[fold.idx_train],
                epochs=epochs_finetune if do_pretrain else config["training"]["epochs"],
                lr=lr_finetune if do_pretrain else config["training"]["learning_rate"],
                weight_decay=weight_decay, patience=patience,
                scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor, 
                pos_weight_multiplier=pos_weight_multiplier,
            )
            fold_val_f1_scores.append(val_f1)
            
            # Pruning veloce se la config fa schifo sui primi fold
            trial.report(val_f1, fold.fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Optuna cerca la configurazione che fa la MEDIA più alta sui 5 fold
        return float(np.mean(fold_val_f1_scores))

    # Lancio dello studio
    study = optuna.create_study(
        study_name=study_name, storage=db_path, load_if_exists=True,
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\n" + "="*50)
    print("RICERCA GLOBALE COMPLETATA!")
    print(f"F1 Medio Ottenuto (K-Fold): {study.best_trial.value:.4f}")
    print("--- QUESTI SONO I PARAMETRI DA COPIARE NEL FILE YAML ---")
    
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    print("="*50)

if __name__ == "__main__":
    run_final_optuna_search()