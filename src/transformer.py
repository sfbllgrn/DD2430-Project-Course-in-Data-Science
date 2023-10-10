# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np


class Transformer(nn.Module):

    def __init__(self, seq_len: int, num_classes: int = 2, d_model=512, nhead=8, num_encoder_layers=6):
        super().__init__()

        self.input_embedding_layer = nn.Linear(in_features=seq_len, out_features=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(in_features=d_model, out_features=num_classes)
        self.train_loss_curve = []
        self.num_epochs = 0

    def fit(self, x_train, y_train, lr=0.001, n_epochs=20):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.num_epochs = n_epochs
        self.train_loss_curve = []

        for n in range(n_epochs):
            y_pred = self.forward(x_train)
            loss = loss_fn(y_pred, y_train)
            self.train_loss_curve.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()

    def evaluate(self, x_test, y_test):

        y_pred = self.forward(x_test)
        y_pred = torch.argmax(y_pred, dim=1)
        accuracy = (y_pred == y_test).float().mean()

        return y_pred, float(accuracy)

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``

        Return:
             Tensor, shape ''[batch_size, num_classes]''
        """
        x = self.input_embedding_layer(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x)

        return x
