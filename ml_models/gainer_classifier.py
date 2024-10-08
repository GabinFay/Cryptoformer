import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from _database import Database

#%%

# import psutil

# available_memory = psutil.virtual_memory().available
# print(f"Available RAM: {available_memory / (1024 ** 3):.2f} GB")


#%%

# Load your data
# Assuming df is your DataFrame loaded from the SQL table 'data5000'

db = Database(ssh_pem_path='BotAmazon.pem', remote=True)

db = Database(remote=True)

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

#%%

# query = f"""SELECT {', '.join(columns)} FROM data5000'"""
# query = f"""
#         SELECT {', '.join(columns)} FROM data5000
#         WHERE fetched_timestamp > '2024-01-05'
#         """

# # Initialize an empty DataFrame
# df = pd.DataFrame()

# # Read data in chunks and concatenate to the DataFrame
# for i, chunk in enumerate(pd.read_sql_query(query, con=db.engine, chunksize=100000)):
#     df = pd.concat([df, chunk], ignore_index=True)
    # Print memory usage of the DataFrame after each concatenation
    # print(i, df.memory_usage(deep=True).sum() / 1024**2, "MB")
#     # Print memory usage of the DataFrame after each concatenation
#     # print(i, df.memory_usage(deep=True).sum() / 1024**2, "MB")


#%%

# query = "SELECT * FROM data5000"
# for chunk in pd.read_sql_query(query, con=db.engine, chunksize=1000):
#     print(chunk.memory_usage(deep=True))



#%%
# df = pd.read_sql_table('data5000', con=db.engine)

#%%

import pandas as pd

# Assuming you have a SQLAlchemy engine created as 'engine'
query = """
SELECT * FROM data5000
WHERE fetched_timestamp >= '2024-05-01'
AND fetched_timestamp <= '2024-05-04'
;
"""

df = pd.read_sql_query(query, db.engine)
df = df[columns]

#%%

# print(df.memory_usage(deep=True).sum() / 1024**2, "MB")


#%%

# db_local.copy_from_stringio(df, 'data5000')

#%%

# df.to_csv('data5000.csv')

#%%

# df = pd.read_csv('data5000.csv')

#%%

# Fill NaNs in one column with values from the other
df['market_cap'] = df['market_cap'].combine_first(df['fully_diluted_valuation'])
df['fully_diluted_valuation'] = df['fully_diluted_valuation'].combine_first(df['market_cap'])

# Drop rows where both columns are still NaN
df.dropna(subset=['market_cap', 'fully_diluted_valuation'], how='all', inplace=True)

# Convert 'fetched_timestamp' to datetime and sort
# df['fetched_timestamp'] = pd.to_datetime(df['fetched_timestamp'])
# df.sort_values(by=['id', 'fetched_timestamp'], inplace=True)

#%%
df.drop_duplicates(subset=['id', 'fetched_timestamp'], keep='first', inplace=True)

#%%

df['fetched_timestamp'] = pd.to_datetime(df['fetched_timestamp'])
df.set_index('fetched_timestamp', inplace=True)
df['MA_6h'] = df.groupby('id')['price_change_percentage_1h_in_currency'].rolling('6H').mean().reset_index(0, drop=True)
df['MA_24h'] = df.groupby('id')['price_change_percentage_1h_in_currency'].rolling('24H').mean().reset_index(0, drop=True)
df['MA_7d'] = df.groupby('id')['price_change_percentage_1h_in_currency'].rolling('168H').mean().reset_index(0, drop=True)

 #%%
# Calculate moving averages for the last 1 hour, 24 hours, and 7 days
# df['ma_1h'] = df.groupby('id')['price_change_percentage_24h_in_currency'].transform(lambda x: x.rolling('1H').mean())
# df['ma_24h'] = df.groupby('id')['price_change_percentage_24h_in_currency'].transform(lambda x: x.rolling('24H').mean())
# df['ma_7d'] = df.groupby('id')['price_change_percentage_24h_in_currency'].transform(lambda x: x.rolling('168H').mean())
df.reset_index(inplace=True)
df['date_of_first_appearance'] = df.groupby('id')['fetched_timestamp'].transform('min')

# Calculate the average since the start of tracking
df['average_since_start'] = df.groupby('id')['price_change_percentage_24h_in_currency'].transform('mean')

# Add class based on future price change percentage
def classify_change(x):
    if x >= 50:
        return 'big gainer'
    elif 10 <= x < 50:
        return 'moderate gainer'
    elif -10 <= x < 10:
        return 'neutral'
    elif -50 <= x < -10:
        return 'moderate loser'
    else:
        return 'big loser'

df['class'] = df['price_change_percentage_24h_in_currency'].shift(-24).apply(classify_change)  # Assuming data is hourly

#%%

# Handle NaN values
df_cleaned = df.dropna()

#%%

# Convert date of first appearance to numerical format
df['date_of_first_appearance'] = df['date_of_first_appearance'].astype(int)
df['ath_date'] = df['ath_date'].astype(int)
df['atl_date'] = df['atl_date'].astype(int)

# Convert timestamp to hour of the day and day of the week
df['hour_of_day'] = pd.to_datetime(df['fetched_timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['fetched_timestamp']).dt.dayofweek


#%% READY TO CLASSIFY & DO PCA !

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# df = df.groupby('id').filter(lambda x: x['market_cap'].max() < 300000)


df.set_index(['id', 'fetched_timestamp'], inplace=True)

df.to_csv('first_classifier.csv')

train_df = df[df.fetched_timestamp <= '2024-05-03']
test_df = df[df.fetched_timestamp == '2024-05-05']

X_train = train_df.drop(['class', 'symbol', 'name'], axis=1)
y_train = train_df['class']

X_test = test_df.drop(['class', 'symbol', 'name'], axis=1)
y_test = test_df['class']

# Assuming df is your DataFrame and 'class' is the target variable
# X = df.drop(['class', 'symbol', 'name'], axis=1)
# y = df['class']

#%%

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Splitting
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 

# Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# import joblib
# joblib.dump(model, 'first_classifier.pkl')

#%%

# Prediction
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


#%% VISUALIZATION OF THE DATA

# Assuming df_cleaned is already loaded and prepared for analysis

# Generate boxplots for each variable for each class
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x='class', y=column, data=df)
    plt.xticks(rotation=45)
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# Generate 1D scatter plots of every variable with dots coloured by class
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns, 1):
    plt.subplot(3, 4, i)
    sns.stripplot(x='class', y=column, data=df, jitter=True)
    plt.xticks(rotation=45)
    plt.title(f'1D Scatter of {column}')
plt.tight_layout()
plt.show()

# Generate 2D pair plots for each variable pair with points coloured by class
sns.pairplot(df, hue='class', vars=df.columns[:-2])
plt.show()

# Perform PCA and plot the first two principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.drop('class', axis=1))
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
principal_df['class'] = df['class'].values

# Plotting the PCA results
sns.scatterplot(x='PC1', y='PC2', hue='class', data=principal_df)
plt.title('PCA - Principal Component Analysis')
plt.show()


