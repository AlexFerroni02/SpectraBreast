from typing import Any, Dict


def _get_model_architecture(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    return model_cfg.get("architecture", model_cfg.get("model_type", "RamanCNN"))


def build_model_from_config(config: Dict[str, Any], device):
    """
    Build a classification model from the YAML config.
    Currently supports RamanCNN. Extend here for Transformer/Hybrid models.
    """
    model_arch = _get_model_architecture(config)
    if model_arch in ("RamanCNN", "CNN"):
        try:
            from src.models.cnn.raman_cnn_01 import RamanCNN
        except ImportError:
            from src.models.cnn.raman_cnn import RamanCNN
        input_length = int(config["model"].get("input_length", 500))
        n_classes = int(config["model"].get("n_classes", 2))
        return RamanCNN(input_length=input_length, n_classes=n_classes).to(device)

    elif model_arch == "Spectra_MAE":
        from src.models.transformer.Spectra_MAE import Spectra_MAE
        params = config.get('model', {}).get('parameters', {})
        return Spectra_MAE(
            sequence_length=params.get('sequence_length', 1738),
            patch_size=params.get('patch_size', 19),
            embedding_dim=params.get('embedding_dim', 256),
            encoder_depth=params.get('encoder_depth', 4),
            encoder_heads=params.get('encoder_heads', 8),
            decoder_depth=params.get('decoder_depth', 2),
            decoder_heads=params.get('decoder_heads', 8),
            mask_ratio=params.get('mask_ratio', 0.6),
        ).to(device)

    elif model_arch == "HybridCNNTransformer":
        from src.models.hybrid.cnn_transformer import HybridCNNTransformer
        return HybridCNNTransformer(
            input_length=input_length, 
            n_classes=n_classes,
            d_model=config["model"].get("d_model", 64),
            nhead=config["model"].get("nhead", 4),
            num_layers=config["model"].get("num_layers", 3)
        ).to(device)

    raise ValueError(f"Unsupported architecture in config: {model_arch}")


def build_model_from_config(config: Dict[str, Any], device):
    model_arch = _get_model_architecture(config)
    input_length = int(config["model"].get("input_length", 500))
    n_classes = int(config["model"].get("n_classes", 2))

    if model_arch in ("RamanCNN", "CNN"):
        from src.models.cnn.raman_cnn_01 import RamanCNN
        return RamanCNN(input_length=input_length, n_classes=n_classes).to(device)
        
    

    raise ValueError(f"Unsupported architecture: {model_arch}")