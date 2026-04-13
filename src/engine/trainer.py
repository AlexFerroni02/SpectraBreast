import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    preds, labels, probs, losses = [], [], [], []
    if criterion is None:
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

def train_one(model, train_loader, val_loader, y_train_fold, config, device):
    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    patience = int(config["training"].get("patience", 50))
    scheduler_patience = int(config["training"].get("scheduler_patience", 7))
    opt_type = config["training"].get("optimizer", "AdamW")

    if opt_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
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
            
        preds_val, labels_val, probs_val, val_loss = evaluate_model(model, val_loader, device, criterion)
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
            
    if best_state is None: 
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, history, best_f1
