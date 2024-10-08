# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:50:31 2024

@author: Gabin
"""

#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

# Suppose df is your 3D DataFrame of shape (750, number_of_features, number_of_timestamps)
# You need to reshape this DataFrame to 2D for easier manipulation, 
# keeping the last dimension (time) as part of your features.
#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Example data loading
# Assuming `data.csv` has columns: 'token_name', 'current_rank', 'market_cap', 'volume', 'rank_change_6h', etc.
df = pd.read_csv('data.csv')

# Convert the 3D DataFrame to 2D if needed. Here's a simplified example:
# df_2d = df.stack().reset_index()

# Example of calculating moving averages for one feature. Repeat for others as needed.
time_windows = {'week': 672, 'day': 96, 'six_hours': 24}  # 15min*4*hours
for window_name, window_size in time_windows.items():
    df[f'market_cap_ma_{window_name}'] = df['market_cap'].rolling(window=window_size).mean()

# Here, 'market_cap' should be replaced by the actual feature column name.


# Feature and target definition
X = df[['current_rank', 'market_cap', 'volume']]  # Example features
y = df['rank_change_6h']  # Target variable

#%% FEQTURE SELECTION

X = df.drop(['target'], axis=1)  # Your features
y = df['target']  # Your target variable

# Use SelectKBest for feature selection
selector = SelectKBest(score_func=f_regression, k='all')
X_new = selector.fit_transform(X, y)
# Get columns to keep and create new dataframe with those only
cols = selector.get_support(indices=True)
X_new = X.iloc[:,cols]



# Splitting dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% 

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# Calculate some error metric
error = sqrt(mean_squared_error(y_test, predictions))

# Assuming 'predictions' includes the future rank increase, 
# you'd sort your predictions to get the tokens with the highest expected rank increase.
top_predictions_idx = np.argsort(predictions)[-5:]
top_5_tokens = df.iloc[top_predictions_idx]



#%%
# Model training
model = LinearRegression()

model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

# Calculating the confidence using the model's score
confidence = model.score(X_test, y_test)

# Displaying predictions and confidence
for i, prediction in enumerate(predictions):
    print(f"Token {X_test.iloc[i].name}: Predicted rank change in 6h: {prediction:.2f}")

print(f"Confidence in predictions: {confidence:.2%}")

#%%

# Training the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")