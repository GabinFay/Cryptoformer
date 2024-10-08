#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:16:33 2024

@author: gabinfay
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the Transformer model for binary classification
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global average pooling
        output = self.fc(output)
        return self.sigmoid(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Normalization function
def normalize_features(matrix):
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)

# Generate synthetic data (Example)
np.random.seed(42)
T, p = 100, 10
matrix = np.random.randn(T, p)
labels = np.zeros(T)
labels[np.random.choice(T, size=1, replace=False)] = 1  # 0.5% anomaly

# Normalize features
normalized_matrix = normalize_features(matrix)

# Convert to PyTorch tensors
inputs = torch.tensor(normalized_matrix, dtype=torch.float32).unsqueeze(1)  # (T, 1, p)
targets = torch.tensor(labels, dtype=torch.float32)

# Hyperparameters
input_dim = p
d_model = 64
nhead = 8
num_encoder_layers = 2
dim_feedforward = 256
dropout = 0.1

# Initialize model, loss, and optimizer
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
class_weights = torch.tensor([0.995, 0.005])  # Adjust weights inversely proportional to class frequency
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output.squeeze(), targets)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Output probability of being an anomaly
model.eval()
with torch.no_grad():
    anomaly_probability = model(inputs).squeeze().numpy()
    print(anomaly_probability)
