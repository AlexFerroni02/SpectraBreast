from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def _default_baseline_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def integrated_gradients_1d(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    target_class_idx: Optional[int] = None,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50,
) -> torch.Tensor:
    """Compute Integrated Gradients for a 1D spectrum input.

    Args:
        model: classification model returning logits of shape (B, C).
        input_tensor: tensor of shape (1, 1, L) or (B, 1, L).
        target_class_idx: if None, uses argmax class for each sample.
        baseline: if None, uses zeros baseline.
        steps: number of interpolation steps.

    Returns:
        Integrated gradients tensor of same shape as input_tensor.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")

    model.eval()

    if baseline is None:
        baseline = _default_baseline_like(input_tensor)

    # Ensure float tensor for gradient.
    input_tensor = input_tensor.float()
    baseline = baseline.float()

    # We'll compute gradients along a straight-line path.
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=input_tensor.device, dtype=input_tensor.dtype)

    grads = []
    for alpha in alphas:
        scaled = baseline + alpha * (input_tensor - baseline)
        scaled.requires_grad_(True)

        logits = model(scaled)
        if logits.ndim != 2:
            raise ValueError(f"Expected logits of shape (B, C), got {tuple(logits.shape)}")

        if target_class_idx is None:
            # Per-sample argmax.
            targets = logits.argmax(dim=1)
            score = logits.gather(1, targets.view(-1, 1)).sum()
        else:
            score = logits[:, int(target_class_idx)].sum()

        grad = torch.autograd.grad(score, scaled, retain_graph=False, create_graph=False)[0]
        grads.append(grad.detach())

    grads = torch.stack(grads, dim=0)  # (steps+1, B, 1, L)

    # Trapezoidal rule average of gradients.
    avg_grads = (grads[:-1] + grads[1:]) * 0.5
    avg_grads = avg_grads.mean(dim=0)  # (B, 1, L)

    ig = (input_tensor - baseline).detach() * avg_grads
    return ig
