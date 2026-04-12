import json
import os

with open("notebooks/03_classification.ipynb", "r") as f:
    nb = json.load(f)

# Keep CELL 1 (imports) and CELL 2 (config).

cell_3_source = """# --- CELLA 3: CARICAMENTO DATI E CREAZIONE SPLIT CONGIUNTI ---
import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.data.mat_loader import load_mat_dataset
from src.data.splits import make_holdout_split, make_kfold_splits

ds_cfg = config.get('dataset', {})
dataset_path = ds_cfg.get('path')
print(f'Caricamento dati da: {dataset_path}...')

loaded = load_mat_dataset(dataset_path)
X_data, Y_data, x_axis, groups = loaded.X, loaded.y, loaded.x_axis, loaded.groups
N, L = X_data.shape

BATCH_SIZE = int(ds_cfg.get('batch_size', config.get('training', {}).get('batch_size', 32)))

split_cfg = config.get('split', {})
SPLIT_SEED = int(split_cfg.get('seed', SEED))

# -- 1) SETUP HOLDOUT
test_size = float(split_cfg.get('holdout', {}).get('test_size', 0.30))
val_size_of_temp = float(split_cfg.get('holdout', {}).get('val_size_of_temp', 0.50))
split_ho = make_holdout_split(Y_data, seed=SPLIT_SEED, test_size=test_size, val_size_of_temp=val_size_of_temp)

train_loader_ho = DataLoader(TensorDataset(torch.from_numpy(X_data[split_ho.idx_train]).unsqueeze(1).float(), torch.from_numpy(Y_data[split_ho.idx_train])), batch_size=BATCH_SIZE, shuffle=True)
val_loader_ho = DataLoader(TensorDataset(torch.from_numpy(X_data[split_ho.idx_val]).unsqueeze(1).float(), torch.from_numpy(Y_data[split_ho.idx_val])), batch_size=BATCH_SIZE, shuffle=False)
test_loader_ho = DataLoader(TensorDataset(torch.from_numpy(X_data[split_ho.idx_test]).unsqueeze(1).float(), torch.from_numpy(Y_data[split_ho.idx_test])), batch_size=BATCH_SIZE, shuffle=False)

print(f'Spettri totali: {N}. Holdout | Train: {len(split_ho.idx_train)}, Val: {len(split_ho.idx_val)}, Test: {len(split_ho.idx_test)}')

# -- 2) SETUP K-FOLD
n_splits = int(split_cfg.get('n_splits', 5))
CV_FOLDS = make_kfold_splits(Y_data, seed=SPLIT_SEED, n_splits=n_splits)
print(f'✅ Fold K-Fold validi estratti: {len(CV_FOLDS)}')
"""

cell_4_source = """# --- CELLA 4: COSTRUZIONE MODELLO E FUNZIONE EVALUATE ---
model_arch = config.get('model', {}).get('architecture', config.get('model', {}).get('model_type', 'RamanCNN'))

def build_model():
    if model_arch in ['RamanCNN', 'CNN']:
        try:
            from src.models.cnn.raman_cnn_01 import RamanCNN
        except ImportError:
            from src.models.cnn.raman_cnn import RamanCNN
        return RamanCNN(input_length=int(config['model']['input_length']), n_classes=int(config['model']['n_classes'])).to(device)
    else:
        raise ValueError(f'Architettura sconosciuta: {model_arch}')

def evaluate_model(model, loader):
    model.eval()
    preds, labels, probs, losses = [], [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            losses.append(criterion(logits, y_b).item())
            probs.append(torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu())
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(y_b.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(probs).numpy(), float(np.mean(losses) if len(losses) else np.nan)
"""

cell_5_source = """# --- CELLA 5: DEFINIZIONE FUNZIONE DI ADDESTRAMENTO ---
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

epochs = int(config["training"]["epochs"])
learning_rate = float(config["training"]["learning_rate"])
weight_decay = float(config["training"].get("weight_decay", 0.0))
patience = int(config["training"].get("patience", 50))
scheduler_patience = int(config["training"].get("scheduler_patience", 7))

def train_one(model, train_loader, val_loader, y_train_fold):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    n_sano = (y_train_fold == 0).sum()
    n_ibd = (y_train_fold == 1).sum()
    weights = torch.tensor([1.0/n_sano if n_sano>0 else 0, 1.0/n_ibd if n_ibd>0 else 0], dtype=torch.float, device=device)
    weights = weights / weights.sum() * 2.0
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=scheduler_patience, verbose=True)
    
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_f1, best_state, epochs_no_improve = -1.0, None, 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        preds_val, labels_val, probs_val, val_loss = evaluate_model(model, val_loader)
        val_f1 = f1_score(labels_val, preds_val, average="macro")
        val_acc = accuracy_score(labels_val, preds_val)
        
        train_loss = total_loss / max(len(train_loader), 1)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_f1"].append(float(val_f1))
        history["val_acc"].append(float(val_acc))
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    if best_state is None: best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, history, best_f1
"""

cell_6_source = """# --- CELLA 6: ESECUZIONE HOLDOUT E GRAFICI ---
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

print("="*40)
print("=== FASE 1: ADDESTRAMENTO HOLDOUT ===")
print("="*40)

model_ho = build_model()
best_state_ho, history_ho, best_f1_ho = train_one(model_ho, train_loader_ho, val_loader_ho, Y_data[split_ho.idx_train])
model_ho.load_state_dict(best_state_ho)
torch.save(model_ho.state_dict(), os.path.join(output_dir, "best_weights_holdout.pth"))

preds_test_ho, labels_test_ho, probs_test_ho, test_loss_ho = evaluate_model(model_ho, test_loader_ho)
cm_ho = confusion_matrix(labels_test_ho, preds_test_ho)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(history_ho["train_loss"], label="Train Loss")
ax[0].plot(history_ho["val_loss"], label="Val Loss")
ax[0].set_title("Curve di Loss (Holdout)")
ax[0].legend()

ConfusionMatrixDisplay(cm_ho).plot(ax=ax[1], cmap="Blues")
ax[1].set_title("Confusion Matrix (Holdout)")

plt.savefig(os.path.join(output_dir, "fig_loss_cm_holdout.png"))
plt.show()

ho_acc = accuracy_score(labels_test_ho, preds_test_ho)
ho_f1 = f1_score(labels_test_ho, preds_test_ho, average="macro")
print(f"Holdout Macro F1: {ho_f1:.4f} | Accuracy: {ho_acc:.4f}")
"""

cell_7_source = """# --- CELLA 7: ESECUZIONE K-FOLD E GRAFICI ---
import math
import pandas as pd

print("\\n" + "="*40)
print("=== FASE 2: ADDESTRAMENTO K-FOLD ===")
print("="*40)

folds_dir = os.path.join(output_dir, "kfolds")
os.makedirs(folds_dir, exist_ok=True)
    
CV_RESULTS, all_labels, all_preds, all_probs = [], [], [], []
kf_histories = {}

for i, fold in enumerate(CV_FOLDS, start=1):
    fold_dir = os.path.join(folds_dir, fold.name)
    os.makedirs(fold_dir, exist_ok=True)
    
    y_train, y_val, y_test = Y_data[fold.idx_train], Y_data[fold.idx_val], Y_data[fold.idx_test]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_data[fold.idx_train]).unsqueeze(1).float(), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_data[fold.idx_val]).unsqueeze(1).float(), torch.from_numpy(y_val)), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_data[fold.idx_test]).unsqueeze(1).float(), torch.from_numpy(y_test)), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\\n--- Running {fold.name} ---")
    model_kf = build_model()
    best_state_kf, history_kf, best_val_f1_kf = train_one(model_kf, train_loader, val_loader, y_train)
    model_kf.load_state_dict(best_state_kf)
    kf_histories[fold.name] = history_kf
    
    torch.save(model_kf.state_dict(), os.path.join(fold_dir, "best_weights.pth"))
    preds_test, labels_test, probs_test, test_loss = evaluate_model(model_kf, test_loader)
    
    test_acc = accuracy_score(labels_test, preds_test)
    test_f1 = f1_score(labels_test, preds_test, average="macro")
    test_auc = roc_auc_score(labels_test, probs_test) if len(np.unique(labels_test)) > 1 else None
    
    metrics = {"fold": fold.name, "macro_f1": float(test_f1), "accuracy": float(test_acc), "roc_auc": float(test_auc) if test_auc is not None else "N/A"}
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=4)
    
    CV_RESULTS.append(metrics)
    all_labels.append(labels_test)
    all_preds.append(preds_test)
    all_probs.append(probs_test)

# MATRICE DELLE LOSS K-FOLD
n_folds = len(CV_FOLDS)
cols = min(4, n_folds)
rows = math.ceil(n_folds / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
if n_folds == 1: axes_flat = [axes]
else: axes_flat = np.array(axes).flatten()

for i, fold in enumerate(CV_FOLDS):
    h = kf_histories[fold.name]
    axes_flat[i].plot(h["train_loss"], label="Train")
    axes_flat[i].plot(h["val_loss"], label="Val")
    axes_flat[i].set_title(f"{fold.name} Loss")
    axes_flat[i].legend()
for j in range(i+1, len(axes_flat)): axes_flat[j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_kfold_losses.png"))
plt.show()

# METRICHE FINALI K-FOLD
df = pd.DataFrame(CV_RESULTS)
print("\\n--- RISULTATI PER FOLD ---")
print(df.to_string(index=False))

f1s = [m["macro_f1"] for m in CV_RESULTS]
accs = [m["accuracy"] for m in CV_RESULTS]
aucs = [float(m["roc_auc"]) for m in CV_RESULTS if m["roc_auc"] != "N/A"]

summary = {
    "mean": {"accuracy": np.mean(accs), "macro_f1": np.mean(f1s), "roc_auc": np.mean(aucs) if len(aucs) else None},
    "std": {"accuracy": np.std(accs), "macro_f1": np.std(f1s), "roc_auc": np.std(aucs) if len(aucs) else None}
}

print("\\n--- RISULTATI AGGREGATI K-FOLD (MEDIA ± STD) ---")
print(f"Macro F1: {summary['mean']['macro_f1']:.4f} ± {summary['std']['macro_f1']:.4f}")
print(f"Accuracy: {summary['mean']['accuracy']:.4f} ± {summary['std']['accuracy']:.4f}")
if summary['mean']['roc_auc']: print(f"ROC AUC:  {summary['mean']['roc_auc']:.4f} ± {summary['std']['roc_auc']:.4f}")

# GRAFICI AGGREGATI K-FOLD
labels_all = np.concatenate(all_labels)
preds_all = np.concatenate(all_preds)
probs_all = np.concatenate(all_probs)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_glob = confusion_matrix(labels_all, preds_all)
ConfusionMatrixDisplay(cm_glob).plot(ax=axes[0], cmap="Blues"); axes[0].set_title("Aggregated Confusion Matrix (K-Fold)")
if len(np.unique(labels_all)) > 1:
    RocCurveDisplay.from_predictions(labels_all, probs_all, ax=axes[1])
    axes[1].set_title(f"Aggregated ROC Curve (K-Fold)")
plt.savefig(os.path.join(output_dir, "fig_kfold_global_metrics.png"))
plt.show()

print("✅ Addestramento Congiunto Completato!")
"""


new_cells = nb["cells"][:2]

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

new_cells.append(create_code_cell(cell_3_source))
new_cells.append(create_code_cell(cell_4_source))
new_cells.append(create_code_cell(cell_5_source))
new_cells.append(create_code_cell(cell_6_source))
new_cells.append(create_code_cell(cell_7_source))

nb["cells"] = new_cells

with open("notebooks/03_classification.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("done")
