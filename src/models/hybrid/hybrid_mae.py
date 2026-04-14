import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.hybrid.cnn_transformer import HybridCNNTransformer # (il modello che ti ho scritto prima)

class HybridMAE(nn.Module):
    def __init__(self, encoder: HybridCNNTransformer, mask_ratio=0.5):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d_model = encoder.transformer.layers[0].linear1.in_features # prende il d_model dal transformer
        
        # 1. Il token MASK che sostituirà le feature nascoste
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 2. Un mini-Transformer per il Decoder (aiuta a rimettere insieme i pezzi)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, activation="gelu")
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=2)
        
        # 3. CNN Inversa (De-Convolution) per ricostruire lo spettro 1D originale
        # Cerchiamo di fare il percorso inverso della tua CNN
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose1d(d_model, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Lunghezza_Spettro)
        B = x.shape[0]
        
        # --- ENCODER: ESTRAZIONE FEATURE ---
        features = self.encoder.cnn(x)          # (B, d_model, L_ridotta)
        features = features.transpose(1, 2)     # -> (B, L_ridotta, d_model)
        L = features.shape[1]
        
        # --- MASKING ---
        # Creiamo rumore casuale per decidere cosa mascherare
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        len_keep = int(L * (1 - self.mask_ratio)) # Es. se 0.5, teniamo metà patch
        
        # Sostituiamo i token mascherati con il parametro mask_token
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).unsqueeze(-1) # Ripristiniamo l'ordine originale
        
        # Token finali = (quelli veri dove mask è 0) + (mask_token dove mask è 1)
        features_masked = features * (1 - mask) + self.mask_token.expand(B, L, -1) * mask
        
        # --- AGGIUNTA CLS TOKEN E POSITIONAL EMBEDDING ---
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        x_enc = torch.cat((cls_tokens, features_masked), dim=1)
        x_enc = x_enc + self.encoder.pos_embedding
        
        # Passiamo al Transformer dell'Encoder
        latent = self.encoder.transformer(x_enc)
        
        # --- DECODER ---
        # Togliamo il CLS token per la ricostruzione
        latent_no_cls = latent[:, 1:, :]
        
        # Passiamo nel mini-transformer decoder
        dec_out = self.decoder_transformer(latent_no_cls)
        
        # Trasponiamo per ridare in pasto alla CNN inversa -> (B, d_model, L_ridotta)
        dec_out = dec_out.transpose(1, 2)
        
        # Ricostruzione dello spettro fisico
        pred_spectra = self.decoder_head(dec_out)
        
        # Trucco per assicurarsi che l'output sia ESATTAMENTE lungo quanto l'input 
        # (le divisioni intere delle CNN possono far perdere qualche punto)
        if pred_spectra.shape[-1] != x.shape[-1]:
            pred_spectra = F.interpolate(pred_spectra, size=x.shape[-1], mode='linear', align_corners=False)
            
        return pred_spectra, mask