import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import MinMaxScaler
import itertools

# Function to create moving averages for all columns
def create_moving_averages(df, window_sizes):
    for window in window_sizes:
        for col in df.columns:
            df[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
    return df

# Function to lag features
def lag_features(df, lag_times):
    new_df = df.copy()
    for lag in lag_times:
        for col in df.columns:
            new_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    new_df = new_df.dropna()
    return new_df

# Feature selection using k-best
def select_features(X, y, task_type='regression', k=10):
    if task_type == 'regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif task_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)].tolist()
    return X_new, selected_features

# Creating a simple LSTM model
def build_lstm_model(input_shape, output_units, task_type='regression'):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_units))
    
    if task_type == 'regression':
        model.compile(loss='mean_squared_error', optimizer='adam')
    elif task_type == 'classification':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Train and evaluate the model
def train_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    predictions = model.predict(X_test)
    if len(y_test.shape) == 1 or y_test.shape[1] == 1:  # Regression or binary classification
        if predictions.shape[-1] > 1:
            predictions = predictions[:, 0]
    if model.loss == 'mean_squared_error':
        return np.sqrt(mean_squared_error(y_test, predictions))  # RMSE
    else:
        return accuracy_score(y_test, (predictions > 0.5).astype(int))  # Accuracy

# Preprocessing pipeline
def time_series_preprocessing_pipeline(df, target_column, look_back=1, forecast_horizon=1, task_type='regression'):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    # Create moving averages and lags
    window_sizes = [3, 5, 7]
    lag_times = [1, 2, 3]
    df_features = create_moving_averages(df_scaled, window_sizes)
    df_features = lag_features(df_features, lag_times)
    
    # Prepare features and target
    if task_type == 'regression':
        y = df_features[target_column].shift(-forecast_horizon)
    elif task_type == 'classification':
        y = (df_features[target_column].shift(-forecast_horizon) > df_features[target_column]).astype(int)
    df_features = df_features.dropna()
    y = y.dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=42)
    
    # Feature selection
    X_train_selected, selected_features = select_features(X_train, y_train, task_type)
    X_test_selected = X_test[selected_features]
    
    # Reshape input to fit LSTM model
    X_train_selected = X_train_selected.values.reshape((X_train_selected.shape[0], 1, X_train_selected.shape[1]))
    X_test_selected = X_test_selected.values.reshape((X_test_selected.shape[0], 1, X_test_selected.shape[1]))
    
    # Build and train the model
    model = build_lstm_model((1, X_train_selected.shape[2]), 1 if task_type == 'regression' else 2, task_type)
    performance = train_evaluate_model(model, X_train_selected, y_train, X_test_selected, y_test)
    
    return model, performance, selected_features

# Example usage
# df = pd.read_csv('your_timeseries.csv', parse_dates=['fetched_timestamp'])
# model, performance, features = time_series_preprocessing_pipeline(df, target_column='your_target_column', task_type='regression')
# print("Model performance (RMSE):", performance)
# print("Selected features:", features)
