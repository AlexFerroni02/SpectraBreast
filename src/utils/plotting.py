import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    RocCurveDisplay, PrecisionRecallDisplay,
    f1_score, accuracy_score, roc_auc_score
)


# ============================================================
# PRE-TRAINING PLOTTING
# ============================================================

def plot_mae_history_sota(history, report_dir):
    """Plots Reconstruction Loss and Masked Cosine Similarity."""
    os.makedirs(report_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    ax1.plot(history['train_loss'], label='Train Loss', color='steelblue', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--', linewidth=2)
    ax1.set_title('Reconstruction Loss (MSE)')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Cosine Similarity Plot
    if 'train_cos_sim' in history and len(history['train_cos_sim']) > 0:
        ax2.plot(history['train_cos_sim'], label='Train Cosine Sim', color='seagreen', linewidth=2)
        ax2.plot(history['val_cos_sim'], label='Val Cosine Sim', color='firebrick', linestyle='--', linewidth=2)
        ax2.set_title('Masked Cosine Similarity (Higher is better)')
        ax2.set_xlabel('Epochs')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
    plt.tight_layout()
    save_path = os.path.join(report_dir, "fig_pretrain_history_sota.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mae_reconstruction_sota(model, val_loader, device, architecture, report_dir, n_examples=4):
    """Plots reconstructions with SOTA khaki background and patched visible regions."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    
    os.makedirs(report_dir, exist_ok=True)
    model.eval()
    
    batch = next(iter(val_loader))[0].to(device)
    if batch.dim() == 2:
        batch = batch.unsqueeze(1)
    elif batch.dim() == 3 and batch.shape[1] != 1:
        batch = batch.transpose(1, 2)
        
    with torch.no_grad():
        if architecture == 'HybridMAE':
            pred, mask = model(batch)
            mask_t = mask.transpose(1, 2)
            mask_up = F.interpolate(mask_t.float(), size=batch.shape[-1], mode='nearest')
            mask_np = mask_up.cpu().numpy()
        else: # Spectra_MAE
            loss, pred, mask = model(batch)
            pred = model.unpatchify(pred)
            mask_det = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_size)
            mask_np = model.unpatchify(mask_det).cpu().numpy()

    original_np = batch.cpu().numpy()
    reconstructed_np = pred.cpu().numpy()
    
    n_plots = min(n_examples, len(original_np))
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
    if n_plots == 1: axes = [axes]
        
    for i in range(n_plots):
        ax = axes[i]
        orig, recon, msk = original_np[i].squeeze(), reconstructed_np[i].squeeze(), mask_np[i].squeeze()
        x_vals = np.arange(len(orig))
        
        # Incolliamo il vero originale sulle parti visibili
        final_reconstruction = np.where(msk == 1, recon, orig)
        
        # Khaki Background for masked regions
        ax.fill_between(x_vals, orig.min() - 0.2, orig.max() + 0.2, 
                        where=(msk == 1), color='khaki', alpha=0.5, label='Masked Region (Predicted)')
        
        ax.plot(x_vals, orig, label='Original Spectrum', color='royalblue', linewidth=1.5, alpha=0.5)
        ax.plot(x_vals, final_reconstruction, label='Reconstructed Spectrum', color='tomato', linestyle='--', linewidth=2)
        
        ax.set_title(f'Validation Spectrum Reconstruction #{i+1}')
        
        # Filter duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        ax.set_ylim(orig.min() - 0.1, orig.max() + 0.1)
        ax.grid(alpha=0.3)
        
    plt.suptitle(f'{architecture} - Qualitative Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(report_dir, "fig_reconstruction_sota.png")
    plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# CLASSIFICATION PLOTTING
# ============================================================

def plot_fold_metrics(history, labels_test, preds_test, probs_test, fold_name, save_path=None):
    """
    Plots a 1x4 figure for a single K-Fold: Loss, Confusion Matrix, ROC, PR.
    """
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))

    # 1. Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", color="steelblue", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="darkorange", linestyle="--", linewidth=2)
    axes[0].set_title(f"Loss ({fold_name})")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Confusion Matrix
    cm = confusion_matrix(labels_test, preds_test)
    ConfusionMatrixDisplay(cm).plot(ax=axes[1], cmap="Blues")
    axes[1].set_title(f"Confusion Matrix ({fold_name})")

    # 3. ROC Curve
    if len(np.unique(labels_test)) > 1:
        RocCurveDisplay.from_predictions(labels_test, probs_test, ax=axes[2])
        axes[2].set_title(f"ROC Curve ({fold_name})")
    else:
        axes[2].text(0.5, 0.5, "Only 1 class", ha="center", va="center", fontsize=14)
        axes[2].set_title(f"ROC Curve ({fold_name})")

    # 4. PR Curve
    if len(np.unique(labels_test)) > 1:
        PrecisionRecallDisplay.from_predictions(labels_test, probs_test, ax=axes[3])
        axes[3].set_title(f"PR Curve ({fold_name})")
    else:
        axes[3].text(0.5, 0.5, "Only 1 class", ha="center", va="center", fontsize=14)
        axes[3].set_title(f"PR Curve ({fold_name})")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_aggregated_metrics(labels_all, preds_all, probs_all, scheme_name, save_path=None):
    """
    Plots aggregated metrics across all folds: Confusion Matrix, ROC, PR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Aggregated Confusion Matrix
    cm = confusion_matrix(labels_all, preds_all)
    ConfusionMatrixDisplay(cm).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title(f"Aggregated Confusion Matrix ({scheme_name})")

    # 2. Aggregated ROC
    if len(np.unique(labels_all)) > 1:
        RocCurveDisplay.from_predictions(labels_all, probs_all, ax=axes[1])
        axes[1].set_title(f"Aggregated ROC ({scheme_name})")
    else:
        axes[1].text(0.5, 0.5, "Only 1 class", ha="center", va="center", fontsize=14)

    # 3. Aggregated PR
    if len(np.unique(labels_all)) > 1:
        PrecisionRecallDisplay.from_predictions(labels_all, probs_all, ax=axes[2])
        axes[2].set_title(f"Aggregated PR Curve ({scheme_name})")
    else:
        axes[2].text(0.5, 0.5, "Only 1 class", ha="center", va="center", fontsize=14)

    overall_f1 = f1_score(labels_all, preds_all, average="macro")
    overall_acc = accuracy_score(labels_all, preds_all)
    fig.suptitle(
        f"{scheme_name} — Macro F1: {overall_f1:.4f} | Accuracy: {overall_acc:.4f}",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()