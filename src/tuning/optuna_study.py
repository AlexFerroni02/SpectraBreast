import os
import sys
import argparse
import yaml
import torch
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# Aggiungiamo root per gli import
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src.data.mat_loader import load_mat_dataset
from src.data.splits import make_kfold_splits, make_lomo_folds
from src.models.factory import build_model_from_config
from src.engine.pretrain import train_mae
from src.engine.finetune import train_finetune, train_linear_probe
from src.engine.trainer import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["CNN", "Transformer", "Hybrid"], help="Modello da ottimizzare")
    parser.add_argument("--dataset", type=str, required=True, choices=["IBD", "TROPHY"], help="Dataset da usare (determina le path)")
    parser.add_argument("--pretrain", type=str, default="False", choices=["True", "False"], help="Se eseguire prima il pre-training")
    parser.add_argument("--n-trials", type=int, default=50, help="Numero di trial da esplorare per singolo FOLD")
    parser.add_argument("--epochs-pretrain", type=int, default=30, help="Epoche per il pretrain MAE")
    parser.add_argument("--epochs-finetune", type=int, default=50, help="Epoche per il finetune classification")
    return parser.parse_args()

def run_optuna_study():
    args = parse_args()
    do_pretrain = args.pretrain == "True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== NESTED OPTUNA SEARCH START ===")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Pretrain: {do_pretrain} | Device: {device}")
    
    # 1. Caricamento Dataset
    if args.dataset == "IBD":
        dataset_path = "data/IBD/Processed/IBD_Dataset_500pt.mat" 
        x_key, y_key = "X_processed", "labels_binary"
    else:
        dataset_path = "data/Trophy/TROPHY_500pt_Labeled.mat"
        x_key, y_key = "X_trophy_500", "Ytrophy_binary"
        
    print(f"Caricamento dati da {dataset_path}...")
    dataset = load_mat_dataset(dataset_path, x_key=x_key, y_key=y_key, dataset_name=args.dataset.lower())
    X_full = dataset.X
    Y_full = dataset.y
    groups = dataset.groups
    
    if X_full.shape[0] == 500 or (X_full.ndim == 2 and X_full.shape[1] > X_full.shape[0] and X_full.shape[0] < 1000):
        X_full = X_full.T
        
    N, L = X_full.shape
    tensor_x = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1)
    tensor_y = torch.tensor(Y_full, dtype=torch.long)
    
    # Creiamo gli split a livello globale (OUTER FOLDS)
    if groups is not None and len(groups) > 0 and args.dataset == "TROPHY":
        print("Uso LOMO (Leave-One-Group-Out) basato sui gruppi (Pazienti)")
        folds = make_lomo_folds(Y_full, groups, seed=42)
    else:
        print("Uso Stratified K-Fold (5 folds)")
        folds = make_kfold_splits(Y_full, seed=42, n_splits=5)
    
    # Crea cartella DB
    base_dir = os.path.abspath("experiments/optuna")
    os.makedirs(base_dir, exist_ok=True)
    
    suffix = "_arch" if args.model == "CNN" else ""
    outer_f1_scores = []
    
    # ==========================================
    # OUTER LOOP: Cicliamo sui Fold Esterni
    # ==========================================
    for fold in folds:
        print(f"\n{'='*40}")
        print(f"=== INIZIO RICERCA PER OUTER FOLD {fold.fold_idx} ===")
        print(f"{'='*40}")
        
        study_name = f"study_{args.model.lower()}_{args.dataset.lower()}_pretrain{do_pretrain}{suffix}_fold{fold.fold_idx}"
        db_file = os.path.join(base_dir, f"{study_name}.db")
        db_path = f"sqlite:///{db_file}"
        
        # Loader del test set (usato SOLO alla fine del fold)
        test_loader = DataLoader(
            TensorDataset(tensor_x[fold.idx_test], tensor_y[fold.idx_test]), 
            batch_size=32, shuffle=False
        )

        # ------------------------------------------
        # INNER LOOP: Obiettivo di Optuna per QUESTO Fold
        # ------------------------------------------
        def objective(trial):
            # Seed riproducibile specifico per trial
            torch.manual_seed(42 + trial.number)
            np.random.seed(42 + trial.number)
            
            # -- Hyperparameter Sampling Generale --
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

            # Struttura base config
            config = {
                "model": { "input_length": L, "n_classes": len(np.unique(Y_full)) },
                "training": {
                    "learning_rate": lr_pretrain if do_pretrain else lr_finetune,
                    "weight_decay": weight_decay,
                    "epochs": epochs_pretrain if do_pretrain else epochs_finetune,
                    "patience": patience
                }
            }
            
            # -- Architettura --
            if args.model == "Hybrid":
                config["model"]["architecture"] = "HybridCNNTransformer" if not do_pretrain else "HybridMAE"
                config["model"]["d_model"] = trial.suggest_categorical("d_model", [32, 64, 128])
                config["model"]["nhead"] = trial.suggest_categorical("nhead", [2, 4, 8])
                config["model"]["num_layers"] = trial.suggest_int("num_layers", 1, 4)
                config["model"]["mask_ratio"] = trial.suggest_float("mask_ratio", 0.5, 0.8)
            elif args.model == "Transformer":
                config["model"]["architecture"] = "ViT_1D" if not do_pretrain else "Spectra_MAE"
                config["model"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [128, 256])
                config["model"]["depth"] = trial.suggest_int("depth", 2, 6)
                config["model"]["heads"] = trial.suggest_categorical("heads", [4, 8])
                config["model"]["patch_size"] = trial.suggest_categorical("patch_size", [10, 20, 50])
                config["model"]["parameters"] = {
                    "sequence_length": L, "patch_size": config["model"]["patch_size"],
                    "embedding_dim": config["model"]["embedding_dim"], "encoder_depth": config["model"]["depth"],
                    "encoder_heads": config["model"]["heads"], "decoder_depth": 2, "decoder_heads": 4, 
                    "mask_ratio": trial.suggest_float("mask_ratio", 0.6, 0.8)
                }
            else:
                config["model"]["architecture"] = "RamanCNN"
                if do_pretrain: raise ValueError("CNN pretrain non supportato. Usa pretrain=False")
                config["model"]["num_layers"]   = trial.suggest_int("num_layers",   2, 4)
                config["model"]["base_filters"] = trial.suggest_categorical("base_filters", [16, 32, 64])
                config["model"]["dropout"]      = trial.suggest_float("dropout", 0.2, 0.6)
                config["model"]["kernel_size"]  = trial.suggest_categorical("kernel_size", [3, 5, 7])

            # Salviamo tutto il dizionario nel trial per poterlo riprendere dopo!
            trial.set_user_attr("config", config)
            trial.set_user_attr("batch_size", batch_size)
            if do_pretrain:
                trial.set_user_attr("lr_probe", lr_probe)
                trial.set_user_attr("epochs_probe", epochs_probe)
                trial.set_user_attr("lr_finetune", lr_finetune)
                trial.set_user_attr("epochs_finetune", epochs_finetune)

            # Preparazione Dati per l'INNER Loop
            train_loader = DataLoader(TensorDataset(tensor_x[fold.idx_train], tensor_y[fold.idx_train]), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(tensor_x[fold.idx_val], tensor_y[fold.idx_val]), batch_size=batch_size, shuffle=False)
            
            base_model = build_model_from_config(config, device)
            
            # -- FASI DI TRAINING --
            if do_pretrain:
                pt_train_loader = DataLoader(TensorDataset(tensor_x[fold.idx_train]), batch_size=batch_size, shuffle=True)
                pt_val_loader = DataLoader(TensorDataset(tensor_x[fold.idx_val]), batch_size=batch_size, shuffle=False)
                pt_weights, _ = train_mae(base_model, pt_train_loader, pt_val_loader, config, device, config["model"]["architecture"])
                
                clf_config = config.copy()
                clf_config["model"]["architecture"] = "HybridCNNTransformer" if args.model == "Hybrid" else "ViT_1D"
                target_model = build_model_from_config(clf_config, device)
                try:
                    if args.model == "Hybrid": target_model.load_state_dict(pt_weights['encoder'], strict=False)
                    else: target_model.load_state_dict(pt_weights, strict=False)
                except Exception as e: print(f"Warning load pesi: {e}")
                
                train_linear_probe(target_model, train_loader, device, Y_full[fold.idx_train], 'classifier', epochs_probe, lr_probe, weight_decay)
            else:
                target_model = base_model
                
            best_state, history, val_f1 = train_finetune(
                model=target_model, train_loader=train_loader, val_loader=val_loader,
                device=device, y_train=Y_full[fold.idx_train],
                epochs=epochs_finetune if do_pretrain else config["training"]["epochs"],
                lr=lr_finetune if do_pretrain else config["training"]["learning_rate"],
                weight_decay=weight_decay, patience=patience,
                scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor, pos_weight_multiplier=pos_weight_multiplier,
            )
            
            # Restituiamo a Optuna le prestazioni sul VALIDATION SET (NON sul test!)
            return float(val_f1)

        # ------------------------------------------
        # ESECUZIONE RICERCA PER IL FOLD CORRENTE
        # ------------------------------------------
        study = optuna.create_study(
            study_name=study_name, storage=db_path, load_if_exists=True,
            direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
        )
        study.optimize(objective, n_trials=args.n_trials)
        
        print(f"Miglior F1 Validazione trovata per Fold {fold.fold_idx}: {study.best_trial.value:.4f}")
        
        # ==========================================
        # VALUTAZIONE FINALE SUL TEST SET ESTERNO
        # ==========================================
        # Estraiamo i parametri vincenti salvati
        best_t = study.best_trial
        best_config = best_t.user_attrs["config"]
        best_bs = best_t.user_attrs["batch_size"]
        
        print(f"\n--- Riaddestramento modello ottimale sul Fold {fold.fold_idx} ---")
        train_loader_fin = DataLoader(TensorDataset(tensor_x[fold.idx_train], tensor_y[fold.idx_train]), batch_size=best_bs, shuffle=True)
        val_loader_fin = DataLoader(TensorDataset(tensor_x[fold.idx_val], tensor_y[fold.idx_val]), batch_size=best_bs, shuffle=False)
        
        final_model = build_model_from_config(best_config, device)
        
        if do_pretrain:
            pt_train_loader = DataLoader(TensorDataset(tensor_x[fold.idx_train]), batch_size=best_bs, shuffle=True)
            pt_val_loader = DataLoader(TensorDataset(tensor_x[fold.idx_val]), batch_size=best_bs, shuffle=False)
            pt_weights, _ = train_mae(final_model, pt_train_loader, pt_val_loader, best_config, device, best_config["model"]["architecture"])
            
            clf_config = best_config.copy()
            clf_config["model"]["architecture"] = "HybridCNNTransformer" if args.model == "Hybrid" else "ViT_1D"
            target_final = build_model_from_config(clf_config, device)
            try:
                if args.model == "Hybrid": target_final.load_state_dict(pt_weights['encoder'], strict=False)
                else: target_final.load_state_dict(pt_weights, strict=False)
            except: pass
            
            train_linear_probe(target_final, train_loader_fin, device, Y_full[fold.idx_train], 'classifier', best_t.user_attrs["epochs_probe"], best_t.user_attrs["lr_probe"], best_t.params["weight_decay"])
            epochs_ft = best_t.user_attrs["epochs_finetune"]
            lr_ft = best_t.user_attrs["lr_finetune"]
        else:
            target_final = final_model
            epochs_ft = best_config["training"]["epochs"]
            lr_ft = best_config["training"]["learning_rate"]

        best_state_fin, _, _ = train_finetune(
            model=target_final, train_loader=train_loader_fin, val_loader=val_loader_fin,
            device=device, y_train=Y_full[fold.idx_train],
            epochs=epochs_ft, lr=lr_ft, weight_decay=best_t.params["weight_decay"],
            patience=best_t.params["patience"], scheduler_patience=best_t.params["scheduler_patience"],
            scheduler_factor=best_t.params["scheduler_factor"], pos_weight_multiplier=best_t.params["pos_weight_multiplier"],
        )
        
        # Test Finale Unico
        target_final.load_state_dict(best_state_fin)
        preds_test, labels_test, _, _ = evaluate_model(target_final, test_loader, device)
        outer_f1 = f1_score(labels_test, preds_test, average="macro")
        
        print(f">>> TEST F1 REALE DEL FOLD {fold.fold_idx}: {outer_f1:.4f} <<<")
        outer_f1_scores.append(outer_f1)

    # Stampa Riassunto Finale
    print("\n" + "="*50)
    print("=== RISULTATI FINALI NESTED CROSS-VALIDATION ===")
    print("="*50)
    for i, f1 in enumerate(outer_f1_scores):
        print(f"Fold {i} Test Macro F1: {f1:.4f}")
    print(f"\nMedia F1 Test Finale: {np.mean(outer_f1_scores):.4f} ± {np.std(outer_f1_scores):.4f}")
    print("="*50)

if __name__ == "__main__":
    run_optuna_study()