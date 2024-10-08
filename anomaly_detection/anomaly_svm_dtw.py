#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:25:50 2024

@author: gabinfay
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from dtaidistance import dtw
from sklearn.metrics import pairwise_distances

# Assuming your data is in a 3D numpy array of shape (n_samples, T, p)
def scale_dataset(data):
    n_samples, T, p = data.shape
    data_reshaped = data.reshape(-1, p)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    return data_scaled.reshape(n_samples, T, p)

def compute_dtw_distance_matrix(data):
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            distance = dtw.distance(data[i], data[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

# Sample dataset
data = np.random.rand(100, 50, 10)  # Replace with your actual data

# Scale the dataset
scaled_data = scale_dataset(data)

# Compute the DTW distance matrix
distance_matrix = compute_dtw_distance_matrix(scaled_data)

# Fit OneClassSVM
ocsvm = OneClassSVM(kernel="precomputed")
ocsvm.fit(distance_matrix)

# Example of prediction
predictions = ocsvm.predict(distance_matrix)
