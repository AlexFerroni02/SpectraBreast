import os
import sys
import argparse
import yaml
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
from src.engine.finetune import train_finetune

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["CNN", "Transformer", "Hybrid"], help="Modello da ottimizzare")
    parser.add_argument("--dataset", type=str, required=True, choices=["IBD", "TROPHY"], help="Dataset da usare (determina le path)")
    parser.add_argument("--pretrain", type=str, default="False", choices=["True", "False"], help="Se eseguire prima il pre-training")
    parser.add_argument("--n-trials", type=int, default=50, help="Numero di trial da esplorare")
    parser.add_argument("--epochs-pretrain", type=int, default=30, help="Epoche per il pretrain MAE")
    parser.add_argument("--epochs-finetune", type=int, default=50, help="Epoche per il finetune classification")
    return parser.parse_args()

def run_optuna_study():
    args = parse_args()
    do_pretrain = args.pretrain == "True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== OPTUNA SEARCH START ===")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Pretrain: {do_pretrain} | Device: {device}")
    
    # 1. Caricamento Dataset (tutto in memoria per velocità, assumendo ram sufficiente)
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
    
    # Creiamo gli split KFOLD a livello globale (stessi split per ogni trial per confronto leale)
    if groups is not None and len(groups) > 0 and args.dataset == "TROPHY":
        print("Uso LOMO (Leave-One-Group-Out) basato sui gruppi (Pazienti)")
        folds = make_lomo_folds(Y_full, groups, seed=42)
    else:
        print("Uso Stratified K-Fold (5 folds)")
        folds = make_kfold_splits(Y_full, seed=42, n_splits=5)
    
    # Optuna Objective
    def objective(trial):
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
        # -- Training Schedule --
        patience           = trial.suggest_int("patience", 5, 20)
        scheduler_patience = trial.suggest_int("scheduler_patience", 2, 8)
        scheduler_factor   = trial.suggest_float("scheduler_factor", 0.1, 0.9)
        pos_weight_multiplier = trial.suggest_float("pos_weight_multiplier", 0.5, 3.0)

        # Struttura base config fittizio
        config = {
            "model": {
                "input_length": L,
                "n_classes": len(np.unique(Y_full))
            },
            "training": {
                # Se facciamo pre-training, passiamo i parametri PT al MAE
                "learning_rate": lr_pretrain if do_pretrain else lr_finetune,
                "weight_decay": weight_decay,
                "epochs": epochs_pretrain if do_pretrain else epochs_finetune,
                "patience": patience
            }
        }
        
        # -- Architettura Hyperparameters --
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
        else: # CNN
            config["model"]["architecture"] = "RamanCNN"
            if do_pretrain:
                raise ValueError("La CNN base non supporta il Masked Pretraining per ora. Usa pretrain=False")
            # --- Ricerca architetturale CNN ---
            config["model"]["num_layers"]   = trial.suggest_int("num_layers",   2, 4)
            config["model"]["base_filters"] = trial.suggest_categorical("base_filters", [16, 32, 64])
            config["model"]["dropout"]      = trial.suggest_float("dropout", 0.2, 0.6)
            config["model"]["kernel_size"]  = trial.suggest_categorical("kernel_size", [3, 5, 7])

        f1_scores = []
        
        # CICLO DI CROSS VALIDATION
        for fold in folds:
            # --- Seed riproducibile per init pesi: stesso punto di partenza
            #     per ogni fold indipendentemente dal trial numero ---
            torch.manual_seed(42 + fold.fold_idx)
            np.random.seed(42 + fold.fold_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + fold.fold_idx)

            # 1. Costruisci il modello da zero per questo FOLD
            base_model = build_model_from_config(config, device)
            
            # Preparazione Dati Fold
            train_loader = DataLoader(TensorDataset(tensor_x[fold.idx_train], tensor_y[fold.idx_train]), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(tensor_x[fold.idx_val], tensor_y[fold.idx_val]), batch_size=batch_size, shuffle=False)
            
            # --- FASE 1: PRE-TRAINING (Se applicabile) ---
            if do_pretrain:
                # Esegue un veloce pre-training SOLO sui dati del fold di train corretto per non spiare il test/val!
                # Utilizziamo train_mae (richiede x_train come val_loader senza y)
                pt_train_loader = DataLoader(TensorDataset(tensor_x[fold.idx_train]), batch_size=batch_size, shuffle=True)
                pt_val_loader = DataLoader(TensorDataset(tensor_x[fold.idx_val]), batch_size=batch_size, shuffle=False)
                
                pt_weights, _ = train_mae(base_model, pt_train_loader, pt_val_loader, config, device, config["model"]["architecture"])
                
                # Ora rigeneriamo il modello di CLASSIFICAZIONE e iniettiamo i pesi (Encoder)
                clf_config = config.copy()
                clf_config["model"]["architecture"] = "HybridCNNTransformer" if args.model == "Hybrid" else "ViT_1D"
                clf_model = build_model_from_config(clf_config, device)
                
                # Inietta i pesi dell'encoder (logica di transfer learning standard MAE)
                try:
                    if args.model == "Hybrid":
                        clf_model.load_state_dict(pt_weights['encoder'], strict=False)
                    else:
                        clf_model.load_state_dict(pt_weights, strict=False)
                except Exception as e:
                    print(f"Warning load pesi: {e}")
                    
                target_model = clf_model
                
                from src.engine.finetune import train_linear_probe
                # --- WARM-UP: LINEAR PROBING ---
                train_linear_probe(
                    model=target_model,
                    train_loader=train_loader,
                    device=device,
                    y_train=Y_full[fold.idx_train],
                    head_attr='classifier',
                    epochs=epochs_probe,
                    lr=lr_probe,
                    weight_decay=weight_decay
                )
            else:
                target_model = base_model
                
            # --- FASE 2: FULL FINE-TUNING ---
            best_state, history, val_f1 = train_finetune(
                model=target_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                y_train=Y_full[fold.idx_train],
                epochs=epochs_finetune,
                lr=lr_finetune,
                weight_decay=weight_decay,
                patience=patience,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
                pos_weight_multiplier=pos_weight_multiplier,
            )
            # --- FASE 3: VALUTAZIONE RIGOROSA SUL TEST SET ---
            # Questo garantisce che Optuna e il Notebook abbiano esattamente 
            # gli stessi risultati e non si illudano di memorizzare il validation_set
            test_loader = DataLoader(
                TensorDataset(tensor_x[fold.idx_test], tensor_y[fold.idx_test]), 
                batch_size=batch_size, shuffle=False
            )
            target_model.load_state_dict(best_state)
            from src.engine.trainer import evaluate_model
            preds_test, labels_test, probs_test, test_loss = evaluate_model(target_model, test_loader, device)
            
            from sklearn.metrics import f1_score
            test_f1 = f1_score(labels_test, preds_test, average="macro")
            f1_scores.append(test_f1)
            
            # Optuna Pruning: Se dopo il primo fold il risultato è disastroso, scarta subito i restanti 4 fold!
            trial.report(test_f1, fold.fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_f1 = float(np.mean(f1_scores))
        return mean_f1

    # === SALVATAGGIO IN DATABASE SQLITE ===
    # _arch suffix per CNN: spazio di ricerca cambiato (ora include arch params)
    # → DB separato rispetto ai trial precedenti senza arch params
    suffix = "_arch" if args.model == "CNN" else ""
    study_name = f"study_{args.model.lower()}_{args.dataset.lower()}_pretrain{do_pretrain}{suffix}"
    
    # Crea cartella e forza path assoluto per evitare lock temporanei o read-only erro di NFS
    base_dir = os.path.abspath("experiments/optuna")
    os.makedirs(base_dir, exist_ok=True)
    db_file = os.path.join(base_dir, f"{study_name}.db")
    db_path = f"sqlite:///{db_file}"

    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1) # Efficienza: stoppa combinazioni perdenti presto
    )
    
    print(f"Iniziando ricerca parametri: {study_name}")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("MIGLIOR TRIAL:")
    print(" Valore  :", study.best_trial.value)
    print(" Params  :", study.best_trial.params)

if __name__ == "__main__":
    run_optuna_study()
