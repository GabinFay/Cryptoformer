#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:18:25 2024

@author: gabinfay
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Read the CSV file
def read_timeseries_csv(filename):
    return pd.read_csv(filename, parse_dates=['fetched_timestamp'], index_col='fetched_timestamp')

# Visualize data
def plot_time_series(data):
    for column in data.columns:
        data[column].plot(title=f"Time Series Plot of {column}")
        plt.show()

# Perform PCA
def perform_pca(data, n_components=2):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    return pd.DataFrame(data=principal_components, index=data.index)

# Clustering
def cluster_data(data, n_clusters=5):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data_scaled)
    return clusters

# Create shifted pairs for forecasting
def create_shifted_pairs(data, column, shift):
    X = data[[column]].iloc[:-shift]
    y = data[column].shift(-shift).dropna()
    return X, y

# Compute correlation and plot heatmap
def plot_correlation(data):
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

# Find most correlated individuals
def find_most_correlated_individuals(data, id_column):
    individual_correlations = {}
    unique_ids = data[id_column].unique()
    for i in unique_ids:
        for j in unique_ids:
            if i != j:
                subset_i = data[data[id_column] == i].drop(columns=[id_column])
                subset_j = data[data[id_column] == j].drop(columns=[id_column])
                correlation = subset_i.corrwith(subset_j, axis=0).mean()
                if np.isnan(correlation):
                    continue
                if (i, j) not in individual_correlations and (j, i) not in individual_correlations:
                    individual_correlations[(i, j)] = correlation
    sorted_correlations = sorted(individual_correlations.items(), key=lambda x: -abs(x[1]))
    return sorted_correlations

# Finding leading indicators
def find_leading_indicators(data, target_column):
    results = {}
    for column in data.columns:
        if column != target_column:
            shifted_corrs = [data[target_column].corr(data[column].shift(-shift)) for shift in range(-30, 31)]
            max_corr = max(shifted_corrs, key=abs)
            shift = shifted_corrs.index(max_corr) - 30
            results[column] = (shift, max_corr)
    return results

# Perform outlier detection
def detect_outliers(data):
    z_scores = zscore(data, axis=0)
    abs_z_scores = np.abs(z_scores)
    outlier_positions = np.where(abs_z_scores > 3)
    return data.iloc[outlier_positions]

# Main function to execute all
def analyze_timeseries(filename):
    data = read_timeseries_csv(filename)
    print("Data Read Complete.")
    plot_time_series(data)
    print("Visualization Complete.")
    
    pca_result = perform_pca(data)
    print("PCA Complete.")

    clusters = cluster_data(data)
    data['Cluster'] = clusters
    print("Clustering Complete.")

    X, y = create_shifted_pairs(data, 'id', 1)  # Change 'id' to the appropriate column name for shifting
    print("Shifted Pairs Created.")

    plot_correlation(data)
    print("Correlation Plot Complete.")

    correlated_individuals = find_most_correlated_individuals(data, 'id')
    print("Most Correlated Individuals:", correlated_individuals)

    leading_indicators = find_leading_indicators(data, 'id')  # Replace 'id' with target column name
    print("Leading Indicators Found:", leading_indicators)

    outliers = detect_outliers(data.drop(columns='id'))  # Assuming 'id' is not a numeric column for processing
    print("Outliers Detected.")

    return data, pca_result, correlated_individuals, leading_indicators, outliers

# Usage example
filename = 'your_timeseries_data.csv'
result = analyze_timeseries(filename)
