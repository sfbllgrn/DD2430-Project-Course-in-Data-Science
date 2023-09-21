import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, n_feat, seq_len, d_model=512, n_head=8, n_layers=6, n_classes=2):
        super().__init__()

        self.input_embedding_layer = nn.Linear(
            in_features=n_feat,
            out_features=d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_layer = nn.Linear(
            in_features=d_model,
            out_features=n_feat
            )

        self.output_layer2 = nn.Linear(
                in_features=seq_len,
                out_features=n_classes
                )

    def forward(self, x):

        x = self.input_embedding_layer(x)

        x = self.transformer_encoder(x)

        x = self.output_layer(x)

        x = self.output_layer2(x.T)

        # print(torch.argmax(x, dim=1))

        return x
