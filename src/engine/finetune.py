import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from src.engine.trainer import evaluate_model


def compute_class_weights(y_train, device):
    n_class0 = int((y_train == 0).sum())
    n_class1 = int((y_train == 1).sum())
    w0 = 1.0 / n_class0 if n_class0 > 0 else 0.0
    w1 = 1.0 / n_class1 if n_class1 > 0 else 0.0
    weights = torch.tensor([w0, w1], dtype=torch.float, device=device)
    if weights.sum() > 0:
        weights = weights / weights.sum() * 2.0
    return weights


def freeze_all_but_head(model, head_attr):
    for p in model.parameters():
        p.requires_grad = False
    head = getattr(model, head_attr, None)
    if head is None:
        raise ValueError(f"Model has no attribute '{head_attr}' for linear probing.")
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def compute_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else None
    pr_auc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else None
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
    }


def train_linear_probe(model, train_loader, device, y_train, head_attr, epochs, lr, weight_decay=0.0, grad_clip=None, verbose=True):
    """
    Linear Probing: trains ONLY the classification head while keeping the encoder frozen.
    """
    freeze_all_but_head(model, head_attr)
    weights = compute_class_weights(y_train, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    head = getattr(model, head_attr)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    if verbose:
        print(f"\n🔒 Linear Probe: training '{head_attr}' for {epochs} epochs (LR={lr:.1e})")

    model.train()
    epoch_iter = tqdm(range(epochs), desc="Linear Probe", leave=False) if verbose else range(epochs)
    
    for epoch in epoch_iter:
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        if verbose and hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

    if verbose:
        print(f"   ✅ Linear Probe complete. Final loss: {avg_loss:.4f}")


def train_finetune(
    model,
    train_loader,
    val_loader,
    device,
    y_train,
    epochs,
    lr,
    weight_decay=0.0,
    patience=10,
    scheduler_patience=5,
    scheduler_factor=0.5,
    pos_weight_multiplier=1.0,
    grad_clip=None,
    verbose=True,
):
    """
    Full fine-tuning: unfreezes all parameters and trains the complete model.
    Uses early stopping based on validation Macro F1.
    """
    unfreeze_all(model)
    weights = compute_class_weights(y_train, device)
    weights[1] *= pos_weight_multiplier
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    if verbose:
        print(f"\n🔓 Fine-Tuning: {epochs} epochs (LR={lr:.1e}, patience={patience})")

    epoch_iter = tqdm(range(epochs), desc="Fine-Tuning", leave=False) if verbose else range(epochs)

    for epoch_idx in epoch_iter:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        preds_val, labels_val, probs_val, val_loss = evaluate_model(model, val_loader, device, criterion)
        val_f1 = f1_score(labels_val, preds_val, average="macro")
        val_acc = accuracy_score(labels_val, preds_val)

        train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_f1"].append(float(val_f1))
        history["val_acc"].append(float(val_acc))

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)

        improved = ""
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            improved = " ✨"
        else:
            epochs_no_improve += 1

        if verbose and hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(
                tr_loss=f"{train_loss:.4f}",
                val_f1=f"{val_f1:.4f}",
                best=f"{best_f1:.4f}",
                pat=f"{epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"\n   🛑 Early stopping at epoch {epoch_idx + 1}. Best F1: {best_f1:.4f}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    if verbose:
        print(f"   ✅ Fine-Tuning complete. Best Macro F1: {best_f1:.4f}")

    return best_state, history, best_f1
