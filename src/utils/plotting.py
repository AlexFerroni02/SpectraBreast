import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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
        
        # 🌟 LA SOLUZIONE: Incolliamo il vero originale sulle parti visibili!
        # Se msk == 1 (giallo), usa la predizione del modello (recon)
        # Se msk == 0 (bianco), usa l'input originale perfetto (orig)
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