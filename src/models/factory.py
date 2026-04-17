from typing import Any, Dict

def _get_model_architecture(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    return model_cfg.get("architecture", model_cfg.get("model_type", "RamanCNN"))

def build_model_from_config(config: Dict[str, Any], device):
    """
    Build a classification/pretraining model from the YAML config.
    """
    model_arch = _get_model_architecture(config)
    
    # --- 1. MODELLO BASE CNN ---
    if model_arch in ("RamanCNN", "CNN"):
        try:
            from src.models.cnn.raman_cnn_01 import RamanCNN
        except ImportError:
            from src.models.cnn.raman_cnn import RamanCNN
        input_length = int(config["model"].get("input_length", 500))
        n_classes    = int(config["model"].get("n_classes", 2))
        num_layers   = int(config["model"].get("num_layers",   3))
        base_filters = int(config["model"].get("base_filters", 16))
        dropout      = float(config["model"].get("dropout",    0.5))
        kernel_size  = int(config["model"].get("kernel_size",  7))
        return RamanCNN(
            input_length=input_length,
            n_classes=n_classes,
            num_layers=num_layers,
            base_filters=base_filters,
            dropout=dropout,
            kernel_size=kernel_size,
        ).to(device)

    elif model_arch in ("ViT_1D", "ViT"):
        # Importiamo la classe ViT dal file ViT_1D.py
        from src.models.transformer.ViT_1D import ViT 
        
        # Mappiamo i parametri del tuo YAML ai nomi esatti richiesti dalla tua classe ViT
        return ViT(
            spectra_size=int(config["model"].get("input_length", 500)),
            patch_size=int(config["model"].get("patch_size", 20)),
            num_classes=int(config["model"].get("n_classes", 2)),
            dim=int(config["model"].get("embedding_dim", 256)),
            depth=int(config["model"].get("depth", 4)),
            heads=int(config["model"].get("heads", 8)),
            dim_mlp=int(config["model"].get("embedding_dim", 256) ), # Di default l'MLP è 4x la dimensione
            dropout=float(config["model"].get("dropout", 0.1)),
            emb_dropout=float(config["model"].get("dropout", 0.1)), # Passiamo lo stesso dropout per sicurezza
            sd=0.1 # Parametro di stochastic depth richiesto dalla tua classe
        ).to(device)
    # --- 2. MODELLO SPECTRA MAE (PRE-TRAINING) ---
    elif model_arch == "Spectra_MAE":
        from src.models.transformer.Spectra_MAE import Spectra_MAE
        # Assicurati che questi parametri matchino le dimensioni dei tuoi spettri!
        params = config.get('model', {}).get('parameters', {})
        return Spectra_MAE(
            sequence_length=params.get('sequence_length', 1738), # Cambia con la lunghezza vera
            patch_size=params.get('patch_size', 19),             # Cambia con il patch size corretto
            embedding_dim=params.get('embedding_dim', 256),
            encoder_depth=params.get('encoder_depth', 4),
            encoder_heads=params.get('encoder_heads', 8),
            decoder_depth=params.get('decoder_depth', 2),
            decoder_heads=params.get('decoder_heads', 8),
            mask_ratio=params.get('mask_ratio', 0.6),
        ).to(device)
        
    # --- 3. MODELLO IBRIDO (CNN + TRANSFORMER) ---
    elif model_arch == "HybridCNNTransformer":
        from src.models.hybrid.cnn_transformer import HybridCNNTransformer
        input_length = int(config["model"].get("input_length", 500))
        n_classes = int(config["model"].get("n_classes", 2))
        return HybridCNNTransformer(
            input_length=input_length, 
            n_classes=n_classes,
            d_model=config["model"].get("d_model", 64),
            nhead=config["model"].get("nhead", 4),
            num_layers=config["model"].get("num_layers", 3)
        ).to(device)
    
    # --- 4. MODELLO IBRIDO MAE (PRE-TRAINING) ---
    elif model_arch == "HybridMAE":
        from src.models.hybrid.hybrid_mae import HybridMAE
        from src.models.hybrid.cnn_transformer import HybridCNNTransformer
        
        # 1. Costruiamo prima l'encoder base
        input_length = int(config["model"].get("input_length", 500))
        n_classes = int(config["model"].get("n_classes", 2)) # Dummy per il pretrain
        d_model = int(config["model"].get("d_model", 64))
        nhead = int(config["model"].get("nhead", 4))
        num_layers = int(config["model"].get("num_layers", 3))
        
        encoder = HybridCNNTransformer(
            input_length=input_length, 
            n_classes=n_classes, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers
        )
        
        # 2. Lo passiamo al MAE
        mask_ratio = float(config["model"].get("mask_ratio", 0.5))
        return HybridMAE(encoder=encoder, mask_ratio=mask_ratio).to(device)

    raise ValueError(f"Unsupported architecture in config: {model_arch}")