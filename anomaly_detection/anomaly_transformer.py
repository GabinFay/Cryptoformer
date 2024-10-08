#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:15:33 2024

@author: gabinfay
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def normalize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Positional Encoding
    positions = np.arange(input_shape[0])[:, np.newaxis]
    d_model = input_shape[1]
    angles = positions / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    positional_encoding = angles[np.newaxis, ...]

    x = inputs + positional_encoding
    
    # Attention Mechanism
    x = LayerNormalization(epsilon=1e-6)(x)
    attn_output, _ = MultiHeadAttention(num_heads=8, key_dim=d_model)(x, x, return_attention_scores=True)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward
    x = Dense(2048, activation='relu')(x)
    x = Dense(d_model)(x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example data
X = np.random.rand(10000, 30, 10)  # 10000 samples, 30 timesteps, 10 features
y = np.random.choice([0, 1], size=(10000,), p=[0.995, 0.005])

# Normalize the features
X = normalize_data(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE for oversampling
smote = SMOTE()

# Keras Classifier for GridSearch
model = KerasClassifier(build_fn=lambda: build_transformer_model((X_train.shape[1], X_train.shape[2])), epochs=10, batch_size=32, verbose=1)

pipeline = ImbPipeline(steps=[('smote', smote), ('model', model)])

param_grid = {
    'model__epochs': [10, 20],
    'model__batch_size': [32, 64],
    'model__model__learning_rate': [1e-4, 1e-3]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
