# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:58:01 2024

@author: 24375
"""

# model_utils.py

import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder

class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """Dataset class for NYC Taxi data."""
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = torch.from_numpy(self._one_hot_X().toarray())  # Convert to tensor
        self.y = torch.from_numpy(self.y_train.values)
        self.X_enc_shape = self.X.shape[-1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def _one_hot_X(self):
        return self.one_hot_encoder.fit_transform(self.X_train)

class MLP(nn.Module):
    """Multilayer Perceptron for regression."""
    def __init__(self, encoded_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)
