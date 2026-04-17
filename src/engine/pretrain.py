import torch
import torch.optim as optim
from tqdm.auto import tqdm

def _compute_hybrid_mae_loss(pred, target, mask):
    """
    Loss MSE only on masked positions.
    pred, target: (B, 1, L)
    mask:         (B, L, 1)  -- 1=masked, 0=visible
    """
    import torch.nn.functional as F
    mask_t = mask.transpose(1, 2)              # (B, 1, L_cnn)
    # Upsample mask to match target sequence length
    mask_up = F.interpolate(mask_t.float(), size=target.shape[-1], mode='nearest')
    loss = ((pred - target) ** 2 * mask_up).sum() / (mask_up.sum() + 1e-8)
    return loss

def forward_and_loss(model, spectra, architecture):
    """
    Unified forward pass that returns (loss, pred, mask) for any architecture.
    """
    if architecture == 'HybridMAE':
        pred, mask = model(spectra)          # pred: (B,1,L), mask: (B,L,1)
        loss = _compute_hybrid_mae_loss(pred, spectra, mask)
        return loss, pred, mask
    elif architecture == "Spectra_MAE":
        return model(spectra)
    else:
        raise ValueError(f"Architettura pre-train non supportata: {architecture}")

def _check_spectra_shape(spectra):
    """Ensure shape is (B, 1, L)."""
    if spectra.dim() == 2:
        spectra = spectra.unsqueeze(1)
    elif spectra.dim() == 3 and spectra.shape[1] != 1:
        spectra = spectra.transpose(1, 2)
    assert spectra.shape[1] == 1, f"Shape canale errata: {spectra.shape}"
    return spectra

def train_mae_one_epoch(model, loader, optimizer, device, architecture):
    model.train()
    total_loss = 0
    for batch in loader:
        spectra = _check_spectra_shape(batch[0].to(device))
        optimizer.zero_grad()
        loss, _, _ = forward_and_loss(model, spectra, architecture)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

def evaluate_mae(model, loader, device, architecture):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            spectra = _check_spectra_shape(batch[0].to(device))
            loss, _, _ = forward_and_loss(model, spectra, architecture)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))

def train_mae(model, train_loader, val_loader, config, device, architecture):
    train_cfg = config.get('training', {})
    lr       = float(train_cfg.get('learning_rate', 1e-3))
    wd       = float(train_cfg.get('weight_decay', 0.05))
    epochs   = int(train_cfg.get('epochs', 50))
    patience = int(train_cfg.get('patience', 10))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = None

    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_mae_one_epoch(model, train_loader, optimizer, device, architecture)
        val_loss   = evaluate_mae(model, val_loader, device, architecture)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Se best_state è sparito per qualche motivo, salviamo l'ultimo
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, history
