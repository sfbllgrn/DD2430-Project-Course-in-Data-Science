
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

"""
input to encoding
(batch size, sequence length, features) 
- sequence length = number of time steps
- features = each input time series

transformer encoding block
- layernormalization 
- multiheadattention
- layer dropout
- feed forward part: layernorm, conv1d, dropout, conv1d

final Mlp classification head
- reduce output tensor of transforr encoder to vector of features
 for each data point in the current batch using pooling layer 1d
- dense layer with linear activation function. use dropout
"""


class Transformer(nn.Module):

    def __init__(self, n_feat, seq_len, d_model=256, n_head=8, n_layers=4, n_classes=2):
        super().__init__()

        self.train_loss_curve = []

        self.n_epochs = 0

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
                in_features=n_feat,
                out_features=n_classes
                )

    def fit(self, x_train, y_train, lr=0.001, n_epochs=20):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        self.n_epochs = n_epochs        
        self.train_loss_curve = []
        
        for n in range(n_epochs):
            y_pred = self.forward(x_train)
            loss = loss_fn(y_pred, y_train)
            self.train_loss_curve.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

    def evaluate(self, x_test, y_test):

        y_pred = self.forward(x_test, predict=True)

        acc = (y_pred.round() == y_test).float().mean()
        
        acc = float(acc)

        return acc
        
    def forward(self, x, predict=False):

        x = self.input_embedding_layer(x)

        x = self.transformer_encoder(x)

        x = self.output_layer(x)

        x = self.output_layer2(x)

        if predict:
            
            return torch.argmax(x, dim=1)

        return x
