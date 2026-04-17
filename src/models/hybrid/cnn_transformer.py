import torch
import torch.nn as nn

class HybridCNNTransformer(nn.Module):
    def __init__(self, input_length=500, n_classes=2, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(HybridCNNTransformer, self).__init__()
        
        # 1. CNN Feature Extractor
        # Utilizziamo convoluzioni con stride per ridurre la lunghezza della sequenza (Patching convoluzionale)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), # L dimezzata
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # L dimezzata
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=3, stride=2, padding=1), # L dimezzata
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # Calcoliamo dinamicamente la lunghezza della sequenza in uscita dalla CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            seq_len_after_cnn = self.cnn(dummy).shape[-1]
        
        # 2. Transformer Components
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Embedding posizionale apprendibile
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len_after_cnn + 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classificatore finale
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, 1, L)
        
        # Estrazione feature locali
        features = self.cnn(x) # -> (B, d_model, L')
        
        # Prepariamo per il transformer: (B, Sequenza, Embedding)
        features = features.transpose(1, 2) # -> (B, L', d_model)
        
        # Aggiunta CLS Token
        B = features.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1) # -> (B, L'+1, d_model)
        
        # Aggiunta Positional Embedding
        x = x + self.pos_embedding
        
        # Applicazione Transformer (Self-Attention)
        x = self.transformer(x)
        
        # Prendiamo solo l'output corrispondente al CLS token (indice 0)
        cls_out = x[:, 0, :]
        
        # Classificazione finale
        out = self.classifier(cls_out)
        return out