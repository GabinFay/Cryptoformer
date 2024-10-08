import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from _database import Database

db = Database(ssh_pem_path='BotAmazon.pem', remote=True)

#%% number of chunks

# query = """
# SELECT COUNT(*) FROM data5000
# WHERE fetched_timestamp >= '2024-05-03'
# AND fetched_timestamp < '2024-05-17'
# ;
# """



# total_rows = pd.read_sql_query(query, con=db.engine).iloc[0, 0]
# chunksize = 1000
# total_chunks = (total_rows + chunksize - 1) // chunksize  # Ceiling division

# print(f"Total chunks: {total_chunks}")

#%%

# query = """
# SELECT * FROM data5000
# WHERE fetched_timestamp >= '2024-05-03'
# AND fetched_timestamp < '2024-05-17'
# ;
# """

# # query = """
# # SELECT * FROM data5000
# # WHERE fetched_timestamp == '2024-05-03'
# # ;
# # """

# # Initialize an empty DataFrame
# df = pd.DataFrame()

# # Read data in chunks and concatenate to the DataFrame
# for i, chunk in enumerate(pd.read_sql_query(query, con=db.engine, chunksize=100000)):
#     df = pd.concat([df, chunk], ignore_index=True)
#     print(i, df.memory_usage(deep=True).sum() / 1024**2, "MB")

#%%

# Specify the SQL query with the columns you want
columns = ['fetched_timestamp', 'market_cap', 'price_change_percentage_24h_in_currency',
           'id', 'symbol', 'name', 'market_cap_rank', 'fully_diluted_valuation', 'total_volume',
           'high_24h', 'low_24h', 'price_change_percentage_24h', 'market_cap_change_24h',
           'market_cap_change_percentage_24h', 'ath_change_percentage', 'ath_date',
           'atl_change_percentage', 'atl_date', 'price_change_percentage_1h_in_currency',
           'price_change_percentage_7d_in_currency', 'rank_change_1h', 'rank_percent_change_1h',
           'rank_change_1d', 'rank_percent_change_1d', 'rank_change_7d', 'rank_percent_change_7d',
           'rank_change_start', 'rank_percent_change_start']

# # Assuming you have a SQLAlchemy engine created as 'engine'
# query = """
# SELECT * FROM data5000
# WHERE fetched_timestamp >= '2024-05-03'
# AND fetched_timestamp < '2024-05-17'
# ;
# """

query = """
SELECT * FROM data5000
WHERE fetched_timestamp >= '2024-05-03'
AND fetched_timestamp < '2024-05-04'
;
"""

# for chunk in pd.read_sql_query(query, con=db.engine, chunksize=1000):
#     print(chunk.memory_usage(deep=True))
    
df = pd.read_sql_query(query, db.engine)
df = df[columns]


#%%

# Fill NaNs in one column with values from the other
df['market_cap'] = df['market_cap'].combine_first(df['fully_diluted_valuation'])
df['fully_diluted_valuation'] = df['fully_diluted_valuation'].combine_first(df['market_cap'])

# Drop rows where both columns are still NaN
df.dropna(subset=['market_cap', 'fully_diluted_valuation'], how='all', inplace=True)
df.drop_duplicates(subset=['id', 'fetched_timestamp'], keep='first', inplace=True)


#%%

df.reset_index(inplace=True)

#%%

df['fetched_timestamp'] = pd.to_datetime(df['fetched_timestamp'])
df['fetched_timestamp'] = df['fetched_timestamp'].dt.round('H') #deals with inconsistent fetched_times !

# df.set_index('fetched_timestamp', inplace=True)

#%%
# Set MultiIndex
# df.set_index(['id', 'fetched_timestamp'], inplace=True)

# Calculate the moving average without resetting the index
df = df.sort_values(by=['id', 'fetched_timestamp'])
df['MA_6H'] = df.groupby('id')['price_change_percentage_1h_in_currency'].transform(lambda x: x.rolling(window=6).mean())



#%%


df.reset_index(inplace=True)
df['date_of_first_appearance'] = df.groupby('id')['fetched_timestamp'].transform('min')

def binary_classify_change(x):
    if x >= 100:
        return 'big gainer'
    else:
        return 'non big gainer'
    
df['class'] = df['price_change_percentage_24h_in_currency'].shift(-24).apply(binary_classify_change)  # Assuming data is hourly

#%%

df = df[df['market_cap'] < 100000000] #smaller than 100m mcap


#%%

missing_values_count = df.isnull().sum()
print(missing_values_count)

#%%
# Handle NaN values
df = df.dropna()

#%%

df['date_of_first_appearance'] = df['date_of_first_appearance'].astype(int)
df['ath_date'] = df['ath_date'].astype(int)
df['atl_date'] = df['atl_date'].astype(int)

df['hour_of_day'] = pd.to_datetime(df['fetched_timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['fetched_timestamp']).dt.dayofweek

train_df = df[df['fetched_timestamp'] < '2024-05-10']
val_df = df[(df['fetched_timestamp'] >= '2024-05-10') & (df['fetched_timestamp'] < '2024-05-14')]
test_df = df[df['fetched_timestamp'] >= '2024-05-14']


#%% utils import

T = 48

from utils import get_token_tensor_from2D, create_sliding_windows
import numpy as np

train_tensor = get_token_tensor_from2D(train_df)
train_windows = create_sliding_windows(train_tensor, T)
test_tensor = get_token_tensor_from2D(test_df)
test_windows = create_sliding_windows(test_tensor, T)
val_tensor = get_token_tensor_from2D(val_df)
val_windows = create_sliding_windows(val_tensor, T)


ts_col_num = df.columns.get_loc('fetched_timestamp')
id_col_num = df.columns.get_loc('id')
class_col_num = df.columns.get_loc('class')
symbol_col_num = df.columns.get_loc('symbol')
name_col_num = df.columns.get_loc('name')

target = train_tensor[:, -1, class_col_num]
indices = ~np.isnan(target)
filtered_train_tensor = train_tensor[indices, :, :]

target = test_tensor[:, -1, class_col_num]
indices = ~np.isnan(target)
filtered_test_tensor = test_tensor[indices, :, :]

target = val_tensor[:, -1, class_col_num]
indices = ~np.isnan(target)
filtered_val_tensor = val_tensor[indices, :, :]

train_index = filtered_train_tensor[:, :, [id_col_num, ts_col_num]]
test_index = filtered_test_tensor[:, :, [id_col_num, ts_col_num]]
val_index = filtered_val_tensor[:, :, [id_col_num, ts_col_num]]

train_labels = filtered_train_tensor[:, :, class_col_num]
test_labels = filtered_test_tensor[:, :, class_col_num]
val_labels = filtered_val_tensor[:, :, class_col_num]

train_tensor_final = np.delete(filtered_train_tensor, [ts_col_num, id_col_num, class_col_num, symbol_col_num, name_col_num], axis=2)
test_tensor_final = np.delete(filtered_test_tensor, [ts_col_num, id_col_num, class_col_num, symbol_col_num, name_col_num], axis=2)
val_tensor_final = np.delete(filtered_val_tensor, [ts_col_num, id_col_num, class_col_num, symbol_col_num, name_col_num], axis=2)

#%%

from sklearn.preprocessing import StandardScaler

n, window_length, p = train_tensor_final.shape

scaler = StandardScaler()
tensor_scaled = scaler.fit_transform(train_tensor_final.reshape(n * window_length, p)).reshape(n, window_length, p)

val_tensor_scaled = scaler.transform(val_tensor.reshape(-1, p)).reshape(val_tensor.shape)
test_tensor_scaled = scaler.transform(test_tensor.reshape(-1, p)).reshape(test_tensor.shape)