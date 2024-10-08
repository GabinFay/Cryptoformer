import os
import pandas as pd
from datetime import datetime, timedelta

def filter_new_tokens(path, hours_ago):
    # Get all filenames in the given directory
    files = [f for f in os.listdir(path) if f.endswith('.csv') and 'top_5000_tokens' in f]
    # Sort files based on the datetime in the filename
    files.sort(key=lambda x: datetime.strptime(x.split('_')[3] + x.split('_')[4][:-4], '%Y-%m-%d%H-%M-%S'), reverse=True)

    # Determine the cutoff datetime for filtering
    cutoff_datetime = datetime.now() - timedelta(hours=hours_ago)
    
    # Load the most recent file
    latest_file_path = os.path.join(path, files[0])
    latest_df = pd.read_csv(latest_file_path)

    # Initialize an empty DataFrame to hold tokens that were not present 'hours_ago'
    new_tokens_df = pd.DataFrame()

    # Iterate over the sorted files to find the first one before the cutoff datetime
    for file in files:
        file_datetime_str = file.split('_')[3] + file.split('_')[4][:-4]
        file_datetime = datetime.strptime(file_datetime_str, '%Y-%m-%d%H-%M-%S')
        if file_datetime < cutoff_datetime:
            # Load this file into a DataFrame
            df_past = pd.read_csv(os.path.join(path, file))
            # Find tokens in the latest_df that are not in df_past
            new_tokens = latest_df[~latest_df['id'].isin(df_past['id'])]
            new_tokens_df = pd.concat([new_tokens_df, new_tokens])
            break

    return new_tokens_df.drop_duplicates()

# Example usage
path = 'brc-20'
path = 'solana-meme-coins'
path='data5000'
# For tokens not in the top 250 a day before
new_tokens_24h = filter_new_tokens(path, 24)
print("New tokens in the last 24 hours:", new_tokens_24h)

# For tokens not in the top 250 6 hours before
new_tokens_6h = filter_new_tokens(path, 6)
print("New tokens in the last 6 hours:", new_tokens_6h)
