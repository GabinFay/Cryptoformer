#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:49:48 2024

@author: gabinfay
"""

import numpy as np

def get_token_tensor_from2D(df):
    # Create a pivot table
    pivot_df = df.pivot_table(index='id', columns='fetched_timestamp', aggfunc='first')
    
    # Get the unique tokens and timestamps
    tokens = df['id'].unique()
    timestamps = df['fetched_timestamp'].unique()
    
    # Initialize the 3D tensor with NaN values
    tensor = np.full((len(tokens), df.shape[1]-2, len(timestamps)), np.nan)
    
    # Create a dictionary to map token ids to their indices
    token_index = {token: idx for idx, token in enumerate(tokens)}
    
    # Populate the tensor
    for token in tokens:
        for timestamp in timestamps:
            if (token, timestamp) in pivot_df.index:
                tensor[token_index[token], :, np.where(timestamps == timestamp)[0][0]] = pivot_df.loc[(token, timestamp)].values
    return tensor
    
# Function to create sliding windows
def create_sliding_windows(tensor, T):
    # Get dimensions
    num_tokens, num_features, num_timesteps = tensor.shape
    
    # Create sliding windows for each token
    windows = np.lib.stride_tricks.sliding_window_view(tensor, (num_features, T), axis=(1, 2))
    
    # Reshape windows to (num_tokens, num_valid_windows_per_token, T, num_features)
    windows = windows.transpose(0, 3, 2, 1)
    
    # Reshape to (num_tokens * num_valid_windows_per_token, T, num_features)
    windows = windows.reshape(-1, T, num_features)
    
    # Filter out windows with NaN values
    valid_windows = windows[~np.isnan(windows).any(axis=(1, 2))]
    
    return valid_windows
    
