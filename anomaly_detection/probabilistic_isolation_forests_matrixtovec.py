#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:52:26 2024

@author: gabinfay
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn

# Example data
# Replace this with your actual dataset loading method
n_samples, p_features, T_timesteps = 1000, 10, 20
data = np.random.rand(n_samples, p_features, T_timesteps)
labels = np.random.randint(0, 2, n_samples)

# Scaling the dataset
scaler = StandardScaler()
data = data.reshape(-1, p_features)  # Reshape to (n_samples * T_timesteps, p_features)
data = scaler.fit_transform(data)
data = data.reshape(n_samples, T_timesteps, p_features)  # Reshape back to (n_samples, T_timesteps, p_features)

# Define the Attention-based Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output

# Parameters
input_dim = p_features
hidden_dim = 64
output_dim = 32
n_layers = 1
n_epochs = 20
batch_size = 32

# Data preparation
dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = Encoder(input_dim, hidden_dim, output_dim, n_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()
for epoch in range(n_epochs):
    for batch in data_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, outputs)
        loss.backward()
        optimizer.step()

# Feature extraction
model.eval()
with torch.no_grad():
    vector_data = model(torch.tensor(data, dtype=torch.float32)).numpy()

# Isolation Forest training
iso_forest = IsolationForest(contamination=0.001, random_state=42)
iso_forest.fit(vector_data)

# Predict anomaly scores
anomaly_scores = -iso_forest.decision_function(vector_data)

# Convert anomaly scores to probabilities
min_score = np.min(anomaly_scores)
max_score = np.max(anomaly_scores)
probabilities = (anomaly_scores - min_score) / (max_score - min_score)

print("Anomaly probabilities:", probabilities)
