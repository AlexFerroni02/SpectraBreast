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

    raise ValueError(f"Unsupported architecture in config: {model_arch}")
