from typing import Any, Dict

def _get_model_architecture(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    return model_cfg.get("architecture", model_cfg.get("model_type", "RamanCNN"))

def build_model_from_config(config: Dict[str, Any], device):
    """
    Build a classification/pretraining model from the YAML config.
    """
    model_arch = _get_model_architecture(config)
    model_cfg = config.get("model", {})
    
    # --- 1. MODELLO BASE CNN ---
    if model_arch in ("RamanCNN", "CNN"):
        try:
            from src.models.cnn.raman_cnn_01 import RamanCNN
        except ImportError:
            from src.models.cnn.raman_cnn import RamanCNN
        return RamanCNN(
            input_length=int(model_cfg.get("input_length", 500)),
            n_classes=int(model_cfg.get("n_classes", 2)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            base_filters=int(model_cfg.get("base_filters", 16)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            kernel_size=int(model_cfg.get("kernel_size", 7)),
        ).to(device)

    # --- 2. MODELLO VIT 1D (FINE-TUNING) ---
    elif model_arch in ("ViT_1D", "ViT"):
        from src.models.transformer.ViT_1D import ViT 
        
        embedding_dim = int(model_cfg.get("embedding_dim", 256))
        # Se non c'è dim_mlp nel yaml, usa in automatico 4x l'embedding
        dim_mlp = int(model_cfg.get("dim_mlp", embedding_dim * 4))
        dropout = float(model_cfg.get("dropout", 0.1))

        return ViT(
            spectra_size=int(model_cfg.get("input_length", 500)),
            patch_size=int(model_cfg.get("patch_size", 20)),
            num_classes=int(model_cfg.get("n_classes", 2)),
            dim=embedding_dim,
            depth=int(model_cfg.get("depth", 8)),
            heads=int(model_cfg.get("heads", 8)),
            dim_mlp=dim_mlp,
            dropout=dropout,
            emb_dropout=dropout, 
            sd=0.1 
        ).to(device)

    # --- 3. MODELLO SPECTRA MAE (PRE-TRAINING) ---
    elif model_arch == "Spectra_MAE":
        from src.models.transformer.Spectra_MAE import Spectra_MAE
        params = model_cfg.get('parameters', {})
        
        return Spectra_MAE(
            sequence_length=int(params.get('sequence_length', 500)),
            patch_size=int(params.get('patch_size', 20)),             
            embedding_dim=int(params.get('embedding_dim', 256)),
            encoder_depth=int(params.get('encoder_depth', 8)),
            encoder_heads=int(params.get('encoder_heads', 8)),
            # 👇 ECCO LA CORREZIONE: Passiamo i parametri mancanti!
            decoder_dim=int(params.get('decoder_dim', 128)),
            decoder_depth=int(params.get('decoder_depth', 2)),
            decoder_heads=int(params.get('decoder_heads', 8)),
            mask_ratio=float(params.get('mask_ratio', 0.75)),
            mlp_dim=int(params.get('dim_mlp', 512)), 
        ).to(device)
        
    # --- 4. MODELLO IBRIDO (CNN + TRANSFORMER) ---
    elif model_arch == "HybridCNNTransformer":
        from src.models.hybrid.cnn_transformer import HybridCNNTransformer
        return HybridCNNTransformer(
            input_length=int(model_cfg.get("input_length", 500)), 
            n_classes=int(model_cfg.get("n_classes", 2)),
            d_model=int(model_cfg.get("d_model", 64)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        ).to(device)
    
    # --- 5. MODELLO IBRIDO MAE (PRE-TRAINING) ---
    elif model_arch == "HybridMAE":
        from src.models.hybrid.hybrid_mae import HybridMAE
        from src.models.hybrid.cnn_transformer import HybridCNNTransformer
        
        encoder = HybridCNNTransformer(
            input_length=int(model_cfg.get("input_length", 500)), 
            n_classes=int(model_cfg.get("n_classes", 2)),
            d_model=int(model_cfg.get("d_model", 64)), 
            nhead=int(model_cfg.get("nhead", 4)), 
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        
        mask_ratio = float(model_cfg.get("mask_ratio", 0.5))
        return HybridMAE(encoder=encoder, mask_ratio=mask_ratio).to(device)

    raise ValueError(f"Unsupported architecture in config: {model_arch}")