import pandas as pd
import glob
import os

# Get all category CSV files and sort them by date
files = sorted(glob.glob('data/categories/category_data_*.csv'), key=os.path.getmtime)

# Load the latest CSV and the one from an hour before
latest_df = pd.read_csv(files[-1], quotechar='"', escapechar='\\')
previous_df = pd.read_csv(files[-168], quotechar='"', escapechar='\\')

# Merge dataframes on 'id'
merged_df = pd.merge(latest_df, previous_df, on='id', suffixes=('_latest', '_prev'))

# Calculate rank difference
merged_df['rank_difference'] = merged_df['rank_prev'] - merged_df['rank_latest']

# Calculate rank difference
merged_df['relative_rank_difference'] = merged_df['rank_difference'] / merged_df['rank_prev'] *100
# Calculate the market cap change
merged_df['market_cap_change'] = merged_df['market_cap_latest'] - merged_df['market_cap_prev']

# Calculate the relative market cap change
merged_df['relative_market_cap_change'] = merged_df['market_cap_change'] / merged_df['market_cap_prev'] *100

# Get top 5 categories with the largest increase in rank
top_5_categories = merged_df.nlargest(5, 'rank_difference')[['name_latest', 'rank_difference']]
# Get the top 5 gainers by relative market cap change
top_5_gainers = merged_df.nlargest(5, 'relative_market_cap_change')[['id', 'name_latest', 'relative_market_cap_change', 'rank_difference', 'relative_rank_difference', 'market_cap_latest']]

print(top_5_gainers)