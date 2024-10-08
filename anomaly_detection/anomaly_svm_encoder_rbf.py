#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:26:15 2024

@author: gabinfay
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Attention
from tensorflow.keras.models import Model

class CustomScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scalers = [StandardScaler().fit(X[:, i, :]) for i in range(X.shape[1])]
        return self
    
    def transform(self, X, y=None):
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_scaled[:, i, :] = self.scalers[i].transform(X[:, i, :])
        return X_scaled

def build_encoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    
    attention = Attention()([x, x])
    x = Dense(128, activation='relu')(attention)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(32, activation='relu')(x)
    
    encoder = Model(inputs, encoded)
    return encoder

class DeepEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape):
        self.encoder = build_encoder(input_shape)
    
    def fit(self, X, y=None):
        self.encoder.compile(optimizer='adam', loss='mse')
        self.encoder.fit(X, X, epochs=10, batch_size=32, verbose=0)
        return self
    
    def transform(self, X, y=None):
        return self.encoder.predict(X)

# Assume X is the input dataset with shape (n_samples, p, T)
X = np.random.rand(1000, 10, 50)  # Example data

# Scale the dataset
scaler = CustomScaler()
X_scaled = scaler.fit_transform(X)

# Encode the dataset
encoder = DeepEncoder(input_shape=(X_scaled.shape[1], X_scaled.shape[2]))
X_encoded = encoder.fit_transform(X_scaled)

# Train One-Class SVM on the encoded dataset
oc_svm = OneClassSVM(kernel='rbf', gamma='auto')
oc_svm.fit(X_encoded)

# Anomaly predictions
y_pred = oc_svm.predict(X_encoded)
