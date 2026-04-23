import torch
import torch.optim as optim
from tqdm.auto import tqdm
import torch.nn.functional as F

def _compute_hybrid_mae_loss(pred, target, mask):
    """MSE Loss computed only on masked positions."""
    mask_t = mask.transpose(1, 2)              
    mask_up = F.interpolate(mask_t.float(), size=target.shape[-1], mode='nearest')
    loss = ((pred - target) ** 2 * mask_up).sum() / (mask_up.sum() + 1e-8)
    return loss

def _compute_masked_cosine_similarity(pred, target, mask, architecture, model=None):
    """Computes Cosine Similarity strictly ONLY on masked regions."""
    if architecture == 'HybridMAE':
        mask_t = mask.transpose(1, 2)
        mask_up = F.interpolate(mask_t.float(), size=target.shape[-1], mode='nearest')
    else:
        # For Spectra_MAE, reconstruct the mask to match the sequence length
        if model is not None and hasattr(model, 'unpatchify'):
            if mask.dim() == 2:
                mask_det = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_size)
                mask_up = model.unpatchify(mask_det)
            else:
                mask_up = mask
        else:
            mask_up = mask

    # FIX: Usiamo .reshape(-1) invece di .view(-1) per gestire la memoria non contigua
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    mask_flat = mask_up.reshape(-1)

    masked_indices = (mask_flat == 1)
    if masked_indices.sum() == 0:
        return 0.0

    # Extract the ACTUAL predictions and targets
    p = pred_flat[masked_indices]
    t = target_flat[masked_indices]
    
    # Compute the overall similarity on masked points
    cos_sim = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0), dim=1)
    return cos_sim.item()

def forward_and_loss(model, spectra, architecture):
    """Unified forward pass returning loss, predictions, and mask."""
    if architecture == 'HybridMAE':
        pred, mask = model(spectra)          
        loss = _compute_hybrid_mae_loss(pred, spectra, mask)
        return loss, pred, mask
    elif architecture == "Spectra_MAE":
        return model(spectra)
    else:
        raise ValueError(f"Unsupported pre-train architecture: {architecture}")

def _check_spectra_shape(spectra):
    """Ensure shape is (B, 1, L)."""
    if spectra.dim() == 2:
        spectra = spectra.unsqueeze(1)
    elif spectra.dim() == 3 and spectra.shape[1] != 1:
        spectra = spectra.transpose(1, 2)
    return spectra

def train_mae_one_epoch(model, loader, optimizer, device, architecture):
    model.train()
    total_loss, total_cos_sim = 0, 0
    for batch in tqdm(loader, desc="  Train", leave=False):
        spectra = _check_spectra_shape(batch[0].to(device))
        optimizer.zero_grad()
        loss, pred, mask = forward_and_loss(model, spectra, architecture)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            cos_sim = _compute_masked_cosine_similarity(pred, spectra, mask, architecture, model)
            total_cos_sim += cos_sim
            
    return total_loss / max(1, len(loader)), total_cos_sim / max(1, len(loader))

def evaluate_mae(model, loader, device, architecture):
    model.eval()
    total_loss, total_cos_sim = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Val  ", leave=False):
            spectra = _check_spectra_shape(batch[0].to(device))
            loss, pred, mask = forward_and_loss(model, spectra, architecture)
            total_loss += loss.item()
            cos_sim = _compute_masked_cosine_similarity(pred, spectra, mask, architecture, model)
            total_cos_sim += cos_sim
            
    return total_loss / max(1, len(loader)), total_cos_sim / max(1, len(loader))

def train_mae(model, train_loader, val_loader, config, device, architecture):
    """
    Main MAE pre-training loop with improved logging.
    Returns best_state (state_dict) and history dict.
    """
    train_cfg = config.get('training', {})
    lr       = float(train_cfg.get('learning_rate', 1e-3))
    wd       = float(train_cfg.get('weight_decay', 0.05))
    epochs   = int(train_cfg.get('epochs', 50))
    patience = int(train_cfg.get('patience', 15))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Cosine Annealing Scheduler
    eta_min = float(train_cfg.get('scheduler', {}).get('eta_min', 1e-6))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    history = {'train_loss': [], 'val_loss': [], 'train_cos_sim': [], 'val_cos_sim': []}
    best_loss = float('inf')
    best_cos_sim = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None

    # Print configuration summary
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"🚀 MAE Pre-training ({architecture})")
    print(f"{'='*60}")
    print(f"  Epochs:       {epochs}")
    print(f"  Learning Rate: {lr:.1e}")
    print(f"  Weight Decay:  {wd:.1e}")
    print(f"  Patience:      {patience}")
    print(f"  Parameters:    {n_params:,}")
    print(f"  Device:        {device}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        train_loss, train_sim = train_mae_one_epoch(model, train_loader, optimizer, device, architecture)
        val_loss, val_sim   = evaluate_mae(model, val_loader, device, architecture)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cos_sim'].append(train_sim)
        history['val_cos_sim'].append(val_sim)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Check for improvement
        improved_marker = ""
        if val_loss < best_loss:
            best_loss = val_loss
            best_cos_sim = val_sim
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved_marker = " ✨ New best!"
        else:
            epochs_no_improve += 1

        # Formatted epoch log
        print(
            f"  Epoch {epoch+1:03d}/{epochs} │ LR: {current_lr:.2e} │ "
            f"Train Loss: {train_loss:.6f} (Cos: {train_sim:.4f}) │ "
            f"Val Loss: {val_loss:.6f} (Cos: {val_sim:.4f}) │ "
            f"[patience: {epochs_no_improve}/{patience}]"
            f"{improved_marker}"
        )

        if epochs_no_improve >= patience:
            print(f"\n  🛑 Early stopping at epoch {epoch+1}.")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 Pre-training Summary")
    print(f"{'='*60}")
    print(f"  Best Epoch:        {best_epoch}")
    print(f"  Best Val Loss:     {best_loss:.6f}")
    print(f"  Best Val Cos Sim:  {best_cos_sim:.4f}")
    print(f"  Total Epochs Run:  {min(epoch + 1, epochs)}")
    print(f"{'='*60}\n")

    return best_state, history