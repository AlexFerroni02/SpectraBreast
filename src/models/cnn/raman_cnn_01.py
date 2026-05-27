# src/models/cnn/raman_cnn_01.py
import torch.nn as nn


class RamanCNN(nn.Module):
    """
    CNN 1D dinamica per spettri Raman.

    I valori di default replicano esattamente la struttura originale a 3 layer
    (16→32→64 filtri, kernel 7→5→3, dropout 0.5), quindi tutto il codice
    esistente che non specifica questi parametri continua a funzionare invariato.

    Parametri aggiuntivi (usati da Optuna per la ricerca architetturale):
        num_layers   : numero di blocchi Conv1d (2-4). Default=3.
        base_filters : filtri nel primo layer; ogni layer successivo raddoppia. Default=16.
        dropout      : dropout prima dell'ultimo linear. Default=0.5.
        kernel_size  : kernel size usato in tutti i layer conv. Default=7.
    """

    def __init__(
        self,
        input_length: int = 500,
        n_classes: int = 2,
        num_layers: int = 3,
        base_filters: int = 16,
        dropout: float = 0.5,
        kernel_size: int = 7,
    ):
        super().__init__()

        # --- Blocchi Conv dinamici ---
        conv_blocks = []
        in_ch = 1
        for i in range(num_layers):
            out_ch = base_filters * (2 ** i)          # 16, 32, 64, 128 …
            padding = kernel_size // 2               # same-padding
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=1, padding=padding),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
            )
            in_ch = out_ch

        self.conv_layers = nn.ModuleList(conv_blocks)

        # Dimensione dopo tutti i max-pool (ogni pool dimezza)
        final_len = input_length // (2 ** num_layers)
        final_filters = base_filters * (2 ** (num_layers - 1))
        flattened_dim = final_filters * final_len

        # --- Head di classificazione ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return self.classifier(x)