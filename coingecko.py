import pandas as pd
from datetime import datetime, timedelta
import glob
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram import send_telegram_message
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import time

import os
print(os.getcwd())

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('coingecko-key.json', scope)
client = gspread.authorize(creds)

def compare_arbitrary(path='data5000', prefix='top_5000_tokens', recent=1, old=2,
                      threshold_inf=0, threshold_sup=5500,
                      mcap_threshold_inf=0, mcap_threshold_sup = 10**14, hourly=True, n=10, volume_threshold = 100000):
    recent_file, old_file = find_files_by_order(path=path, prefix=prefix, recent=recent, old=old)
    print(recent_file, old_file)
    df_recent = pd.read_csv(recent_file)  # The latest file
    df_old = pd.read_csv(old_file)  # The oldest file
    top_movers = compare_dfs(df_old, df_recent, threshold_inf=threshold_inf, threshold_sup=threshold_sup, hourly=hourly,
                             mcap_threshold_inf=mcap_threshold_inf, mcap_threshold_sup=mcap_threshold_sup, n=n, volume_threshold=volume_threshold)
    # top_movers = keep_relevant_info(top_movers)
    return top_movers

def compare_arbitrary_top_gainers_sheet(path='data5000', prefix='top_5000_tokens', recent=1, old=2,
                      threshold_inf=0, threshold_sup=5500,
                      mcap_threshold_inf=0, mcap_threshold_sup = 10**14, hourly=True, n=10, volume_threshold = 100000):
    recent_file, old_file = find_files_by_order(path=path, prefix=prefix, recent=recent, old=old)
    print(recent_file, old_file)
    df_recent = pd.read_csv(recent_file)  # The latest file
    df_old = pd.read_csv(old_file)  # The oldest file
    top_movers = compare_dfs(df_old, df_recent, threshold_inf=threshold_inf, threshold_sup=threshold_sup, hourly=hourly,
                             mcap_threshold_inf=mcap_threshold_inf, mcap_threshold_sup=mcap_threshold_sup, n=n, volume_threshold=volume_threshold)
    recent_timestamp = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', recent_file).group(1)
    fetched_timestamp = datetime.strptime(recent_timestamp, '%Y-%m-%d_%H-%M-%S')
    top_movers['fetched_timestamp'] = fetched_timestamp
    top_movers = top_movers[['id','rank_old', 'rank_%', '1h_new', 'fetched_timestamp']]
    # top_movers = keep_relevant_info(top_movers)
    return top_movers



def compare_dfs(df_old, df_recent, threshold_inf = 0, threshold_sup = 5500, hourly=True,
                mcap_threshold_inf=0, mcap_threshold_sup = 10**14, n=10, volume_threshold=100000):
    """Compare two DataFrames and return the top movers based on rank change."""
    merged_df = df_old.merge(df_recent, on='id', suffixes=('_old', '_new'))
    # merged_df['high_24h_pct'] = (merged_df['high_24h_old'] - merged_df['current_price_old']) / merged_df['current_price_old'] * 100
    # merged_df['low_24h_pct'] = (merged_df['low_24h_old'] - merged_df['current_price_old']) / merged_df['current_price_old'] * 100
    merged_df['rank_change'] = merged_df['market_cap_rank_old'] - merged_df['market_cap_rank_new']
    merged_df['volume_change_pct'] = (merged_df['total_volume_new'] - merged_df['total_volume_old']) / merged_df['total_volume_old'] * 100
    merged_df['rank_change_pct'] = merged_df['rank_change'] / merged_df['market_cap_rank_old'] * 100

    merged_df = merged_df[merged_df['market_cap_rank_old'] < threshold_sup]  # Filter out only the biggest coins
    merged_df = merged_df[merged_df['market_cap_rank_old'] > threshold_inf]  # Filter out only the biggest coins
    merged_df = merged_df[merged_df['market_cap_old'] < mcap_threshold_sup]  # Filter out only the biggest coins #HARDCODED : if BTC goes up 100T it crashes
    merged_df = merged_df[merged_df['market_cap_old'] > mcap_threshold_inf]  # Filter out only the biggest coins
    merged_df = merged_df[merged_df['total_volume_new'] > volume_threshold]  # Filter out low volume tokens #HARDCODED
    merged_df['ath_date_old'] = pd.to_datetime(merged_df['ath_date_old'], errors='coerce')
    merged_df['atl_date_old'] = pd.to_datetime(merged_df['atl_date_old'], errors='coerce')
    merged_df = merged_df[merged_df['ath_date_old'] > '2023-01-01']
    merged_df = merged_df[merged_df['atl_date_old'] > '2023-01-01']
    top_movers = merged_df[merged_df['rank_change_pct'] != 0].nlargest(n, 'rank_change_pct', keep='all')
    # top_movers['atl_date_new'] = top_movers['atl_date_new'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if hourly:
        top_movers = top_movers[['id', 'market_cap_rank_old', 'rank_change', 'rank_change_pct',
                                 'total_volume_old', 'volume_change_pct',
                                 'market_cap_old', 'market_cap_new','price_change_percentage_1h_in_currency_new',
                             'price_change_percentage_24h_in_currency_new', 'price_change_percentage_7d_in_currency_new']]
    else: #means daily
        top_movers = top_movers[['id',
                                 'market_cap_rank_old', 'market_cap_rank_new', 'rank_change', 'rank_change_pct',
                                 'market_cap_old', 'market_cap_new',
                                 'price_change_percentage_1h_in_currency_old', 'price_change_percentage_1h_in_currency_new',
                                 'price_change_percentage_24h_in_currency_old', 'price_change_percentage_24h_in_currency_new',
                                 'price_change_percentage_7d_in_currency_old', 'price_change_percentage_7d_in_currency_new',
                                 'total_volume_old', 'total_volume_new', 'volume_change_pct']]
        top_movers['total_volume_new'] = top_movers['total_volume_new'] / 1000000
        top_movers['total_volume_new'] = top_movers['total_volume_new'].round(2)
    top_movers = top_movers.map(lambda x: np.round(x * 2) / 2 if isinstance(x, float) else x)
    top_movers['market_cap_old'] = top_movers['market_cap_old'] / 1000000
    top_movers['market_cap_new'] = top_movers['market_cap_new'] / 1000000
    top_movers['total_volume_old'] = top_movers['total_volume_old'] / 1000000
    top_movers['market_cap_old'] = top_movers['market_cap_old'].round(3)
    top_movers['market_cap_new'] = top_movers['market_cap_new'].round(3)
    top_movers['total_volume_old'] = top_movers['total_volume_old'].round(3)
    top_movers.rename(columns={'price_change_percentage_1h_in_currency_new': '1h_new'}, inplace=True)
    top_movers.rename(columns={'price_change_percentage_24h_in_currency_new': '24h_new'}, inplace=True)
    top_movers.rename(columns={'price_change_percentage_7d_in_currency_new': '7d_new'}, inplace=True)

    top_movers.replace([np.inf, -np.inf], np.nan, inplace=True)
    top_movers.fillna(0, inplace=True)
    top_movers.rename(columns={'market_cap_rank_old': 'rank_old'}, inplace=True)
    top_movers.rename(columns={'rank_change_pct': 'rank_%'}, inplace=True)
    top_movers.rename(columns={'total_volume_old': 'vol_old'}, inplace=True)
    top_movers.rename(columns={'volume_change_pct': 'vol_%'}, inplace=True)
    top_movers.rename(columns={'market_cap_old': 'mcap_old'}, inplace=True)
    top_movers.rename(columns={'market_cap_new': 'mcap_new'}, inplace=True)
    top_movers.rename(columns={'price_change_percentage_1h_in_currency_old': '1h_old'}, inplace=True)
    top_movers.rename(columns={'price_change_percentage_24h_in_currency_old': '24h_old'}, inplace=True)
    top_movers.rename(columns={'price_change_percentage_7d_in_currency_old': '7d_old'}, inplace=True)
    # df.dropna(inplace=True)
    return top_movers

def find_files_by_order(path='data5000', prefix='top_5000_tokens', recent=1, old=2):
    list_of_files = glob.glob(f'data/{path}/{prefix}_*.csv')
    list_of_files.sort(key=lambda x: datetime.strptime(re.search(rf'{prefix}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}).csv', x).group(1), '%Y-%m-%d_%H-%M-%S'))
    list_of_files.reverse()
    recent_file = None
    old_file = None
    if 0 < recent <= len(list_of_files):
        recent_file = list_of_files[recent-1]
    if 0 < old <= len(list_of_files) or old == -1:
        if old == -1:
            old_file = list_of_files[-1]
        else:
            old_file = list_of_files[old-1]
    return recent_file, old_file

def regular_top_movers():
    already_in = {}
    for k in range(1, 24):
        # print(k)
        top_movers = compare_arbitrary(recent=k, old=k+1)
        for symbol in top_movers['id']:
            if symbol in already_in.keys():
                already_in[symbol] += 1
            else:
                already_in[symbol] = 1
    already_in = sorted(already_in.items(), 
                       key = lambda item: item[1],
                       reverse=True)
    return already_in

def save_df_to_sheets(df, filename, blank_line = 2, prefix = ''):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    for _ in range(6):  # Retry up to 5 times
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name('coingecko-key.json', scope)
            client = gspread.authorize(creds)
            spreadsheet = client.open(filename)
            sheet = spreadsheet.get_worksheet(0)
            current_time = datetime.now().strftime('%H H, %d %B %Y')
            values = [['Data fetched at:', current_time, prefix], df.columns.values.tolist()] + df.values.tolist() + blank_line * [['']]
            sheet.insert_rows(values, 2)
            break
        except gspread.exceptions.APIError as e:
            if e.response.status_code == 503:
                time.sleep((2 ** _) + (random.randint(0, 1000) / 1000))  # Exponential backoff + jitter
            else:
                raise e

    # sheet.update([df.columns.values.tolist()] + df.values.tolist())

def send_df_over_telegram(df):
    BOT_TOKEN = '6620700365:AAEu1nInz_VNETdg9wI_v12AzpZI84r4JcI'
    CHAT_ID = '5687811141'
    for index, row in df.iterrows():
        send_telegram_message(BOT_TOKEN, CHAT_ID, str(row.to_dict()))
    send_telegram_message(BOT_TOKEN, CHAT_ID, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def create_spreadsheet(filename):
    spreadsheet = client.create(filename)
    spreadsheet.share('gabin.fay@gmail.com', perm_type='user', role='writer')

def make_writer(filename):
    spreadsheet = client.open(filename)
    spreadsheet.share('gabin.fay@gmail.com', perm_type='user', role='writer')

def hourly_sheets_update_5000_flat(test=False, recent=1, old=-1):
    df = compare_arbitrary(recent=recent, old=old, n=10, volume_threshold=40000) #HARDCODED
    if not test:
        save_df_to_sheets(df, '5000_rank_percent.csv', blank_line=2, prefix ='all mcap')
    else:
        save_df_to_sheets(df, 'test.csv', blank_line=2, prefix ='all mcap')

def hourly_sheets_update_bins(test=False, recent=1, old=-1):
    df1 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**9, mcap_threshold_sup=10**10, n=4)
    df2 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**8, mcap_threshold_sup=10**9, n=4)
    df3 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**7, mcap_threshold_sup=10**8, n=4, volume_threshold = 500000) #HARDCODED
    df4 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**6, mcap_threshold_sup=10**7, n=4)
    df5 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**5, mcap_threshold_sup=10**6, n=4, volume_threshold = 10000)
    if not test:
        save_df_to_sheets(df5, 'rank_percent.csv', blank_line=2, prefix = '100k to 1m')
        save_df_to_sheets(df4, 'rank_percent.csv', blank_line=0, prefix = '1m to 10m')
        save_df_to_sheets(df3, 'rank_percent.csv', blank_line=0, prefix = '10m to 100m')
        save_df_to_sheets(df2, 'rank_percent.csv', blank_line=0, prefix = '10m to 1b')
        save_df_to_sheets(df1, 'rank_percent.csv', blank_line=0, prefix = '1b to 10b')
    else:
        save_df_to_sheets(df5, 'test.csv', blank_line=2, prefix = '100k to 1m')
        save_df_to_sheets(df4, 'test.csv', blank_line=0, prefix = '1m to 10m')
        save_df_to_sheets(df3, 'test.csv', blank_line=0, prefix = '10m to 100m')
        save_df_to_sheets(df2, 'test.csv', blank_line=0, prefix = '10m to 1b')
        save_df_to_sheets(df1, 'test.csv', blank_line=0, prefix = '1b to 10b')

def hourly_sheets_update_small_mcap(test=False, recent=1, old=2):
    df3 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**7, mcap_threshold_sup=10**8)
    df4 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**6, mcap_threshold_sup=10**7)
    df5 = compare_arbitrary(recent=recent, old=old, mcap_threshold_inf = 10**5, mcap_threshold_sup=10**6, volume_threshold = 10000)
    # df3 = compare_arbitrary(recent=1, old=2, threshold_inf = 3000)
    if not test:
        save_df_to_sheets(df5, 'degen.csv', blank_line=2, prefix = '100k to 1m')
        save_df_to_sheets(df4, 'degen.csv', blank_line=0, prefix = '1m to 10m')
        save_df_to_sheets(df3, 'degen.csv', blank_line=0, prefix = '10m to 100m')
    else:
        save_df_to_sheets(df5, 'test.csv', blank_line=2, prefix = '100k to 1m')
        save_df_to_sheets(df4, 'test.csv', blank_line=0, prefix = '1m to 10m')
        save_df_to_sheets(df3, 'test.csv', blank_line=0, prefix = '10m to 100m')

def hourly_sheets_update_big_mcap(test=False):
    df1 = compare_arbitrary(recent=1, old=2, mcap_threshold_inf=10**9, mcap_threshold_sup=10**10)
    df2 = compare_arbitrary(recent=1, old=2, mcap_threshold_inf = 10**8, mcap_threshold_sup=10**9)
    if not test:
        save_df_to_sheets(df2, 'top_movers.csv', blank_line=2, prefix = '100m to 1b')
        save_df_to_sheets(df1, 'top_movers.csv', blank_line=0, prefix = '1b to 10b')
    else:
        save_df_to_sheets(df2, 'test.csv', blank_line=2, prefix = '100m to 1b')
        save_df_to_sheets(df1, 'test.csv', blank_line=0, prefix = '1b to 10b')

def daily_sheets_update(test=False):
    df1 = compare_arbitrary(recent=1, old=2, mcap_threshold_inf=10**9)
    df2 = compare_arbitrary(recent=1, old=2, mcap_threshold_inf = 10**8, mcap_threshold_sup=10**9)
    df3 = compare_arbitrary(recent=1, old=2, mcap_threshold_inf = 10**7, mcap_threshold_sup=10**8)
    df4 = compare_arbitrary(recent=1, old=2, mcap_threshold_inf = 10**6, mcap_threshold_sup=10**7)
    df5 = compare_arbitrary(recent=1, old=2, mcap_threshold_inf = 10**5, mcap_threshold_sup=10**6, volume_threshold=10000)
    # df3 = compare_arbitrary(recent=1, old=2, threshold_inf = 3000)
    if not test:
        save_df_to_sheets(df5, 'daily_top_movers.csv', blank_line=2, prefix = '100k to 1m')
        save_df_to_sheets(df4, 'daily_top_movers.csv', blank_line=0, prefix = '1m to 10m')
        save_df_to_sheets(df3, 'daily_top_movers.csv', blank_line=0, prefix = '10m to 100m')
        save_df_to_sheets(df2, 'daily_top_movers.csv', blank_line=0, prefix = '10m to 1b')
        save_df_to_sheets(df1, 'daily_top_movers.csv', blank_line=0, prefix = '1b to 10b')
        df = compare_arbitrary(path='solana-meme-coins', prefix='top_250_tokens', recent=1, old=25)
        save_df_to_sheets(df, 'solana-meme-coins.csv', blank_line=2)
        df = compare_arbitrary(path='brc-20', prefix='top_250_tokens', recent=1, old=25)
        save_df_to_sheets(df, 'brc-20.csv', blank_line=2)
        df = compare_arbitrary(path='base-meme-coins', prefix='top_250_tokens', recent=1, old=25)
        save_df_to_sheets(df, 'base-meme-coins.csv', blank_line=2)
    else:
        save_df_to_sheets(df5, 'test.csv', blank_line=2)
        save_df_to_sheets(df4, 'test.csv', blank_line=0)
        save_df_to_sheets(df3, 'test.csv', blank_line=0)
        save_df_to_sheets(df2, 'test.csv', blank_line=0)
        save_df_to_sheets(df1, 'test.csv', blank_line=0)
        df = compare_arbitrary(path='solana-meme-coins', prefix='top_250_tokens', recent=1, old=25)
        save_df_to_sheets(df, 'test.csv', blank_line=2)
        df = compare_arbitrary(path='brc-20', prefix='top_250_tokens', recent=1, old=25)
        save_df_to_sheets(df, 'test.csv', blank_line=2)
        df = compare_arbitrary(path='base-meme-coins', prefix='top_250_tokens', recent=1, old=25)
        save_df_to_sheets(df, 'test.csv', blank_line=2)

# def daily_sheets_update(test=False):
#     df1 = compare_arbitrary(recent=1, old=25, threshold_sup = 1500, hourly=False)
#     df2 = compare_arbitrary(recent=1, old=25, threshold_inf = 1500, threshold_sup = 3000, hourly=False)
#     df3 = compare_arbitrary(recent=1, old=25, threshold_inf = 3000, hourly=False)
#     if not test:
#         save_df_to_sheets(df3, 'daily_top_movers.csv', blank_line=2)
#         save_df_to_sheets(df2, 'daily_top_movers.csv', blank_line=0)
#         save_df_to_sheets(df1, 'daily_top_movers.csv', blank_line=0)
#         df = compare_arbitrary(path='solana-meme-coins', prefix='top_250_tokens', recent=1, old=25)
#         save_df_to_sheets(df, 'solana-meme-coins.csv', blank_line=2)
#         df = compare_arbitrary(path='brc-20', prefix='top_250_tokens', recent=1, old=25)
#         save_df_to_sheets(df, 'brc-20.csv', blank_line=2)
#     else:
#         save_df_to_sheets(df3, 'test.csv', blank_line=2)
#         save_df_to_sheets(df2, 'test.csv', blank_line=0)
#         save_df_to_sheets(df1, 'test.csv', blank_line=0)
#%% new functs

import pandas as pd
import glob
from datetime import datetime

def extract_token_data(path='data5000', prefix='top_5000_tokens', token_id='bitcoin'):
    """Extracts historical data for a single token across multiple CSV files."""
    list_of_files = sorted(glob.glob(f'data/{path}/{prefix}_*.csv'), reverse=True)
    frames = []
    for file in list_of_files:
        df = pd.read_csv(file)
        token_data = df[df['id'] == token_id]
        if not token_data.empty:
            token_data['timestamp'] = datetime.strptime(file.split('_')[-1].split('.')[0], '%H-%M-%S')
            frames.append(token_data)
    return pd.concat(frames, ignore_index=True)

def compare_token_data(df_old, df_recent):
    """Compares two dataframes for a single token to calculate changes in rank and market cap."""
    df_old = df_old.add_suffix('_old')
    df_recent = df_recent.add_suffix('_new')
    merged_df = pd.concat([df_old, df_recent], axis=1)
    merged_df['rank_change'] = merged_df['market_cap_rank_old'] - merged_df['market_cap_rank_new']
    merged_df['market_cap_change'] = merged_df['market_cap_new'] - merged_df['market_cap_old']
    return merged_df[['timestamp_new', 'rank_change', 'market_cap_change']]

def update_top_movers_with_duration(regular_hours=24):
    """Updates the top movers with the duration they stayed in the top list within a given timeframe."""
    result = {}
    for hour in range(1, regular_hours+1):
        hourly_movers = compare_arbitrary(recent=hour, old=hour+1)
        for token in hourly_movers['id']:
            if token in result:
                result[token] += 1
            else:
                result[token] = 1
    return pd.DataFrame(list(result.items()), columns=['id', 'Hours in Top Movers']).sort_values(by='Hours in Top Movers', ascending=False)

def extract_focus_tokens(df, n_rows=3):
    """Extracts the first n rows from the dataframe."""
    return df.head(n_rows)

def plot_token_features(token_ids, features):
    """Plots given features over time for the specified tokens."""
    plt.figure(figsize=(14, 7))
    for token_id in token_ids:
        df = extract_token_data(token_id=token_id)
        for feature in features:
            plt.plot(df['timestamp'], df[feature], label=f'{token_id} - {feature}')
    plt.xlabel('Time')
    plt.ylabel('Feature Values')
    plt.title('Token Feature Changes Over Time')
    plt.legend()
    plt.show()

def link_and_plot(df, features):
    """Links the token data extraction and plotting for the first three rows of a given dataframe."""
    focus_tokens = extract_focus_tokens(df)
    token_ids = focus_tokens['id'].tolist()
    plot_token_features(token_ids, features)
    
#%%    

def get_sorted_files(path='data5000', prefix='top_5000_tokens'): #from recent to old
    list_of_files = glob.glob(f'data/{path}/{prefix}_*.csv')
    list_of_files.sort(key=lambda x: datetime.strptime(re.search(rf'{prefix}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}).csv', x).group(1), '%Y-%m-%d_%H-%M-%S'))
    list_of_files.reverse()
    return list_of_files

def add_rank_change_columns(df_recent, df_old):
    if df_old is not None:
        merged = df_recent.merge(df_old[['id', 'market_cap_rank']], on='id', how='left', suffixes=('', '_old'))
        merged['rank_change'] = (merged['market_cap_rank_old'] - merged['market_cap_rank']).astype('Int64')
        merged['rank_percent_change'] = merged['rank_change'] / merged['market_cap_rank_old'] * 100
        return merged.drop(columns=['market_cap_rank_old'])
    else:
        merged = pd.DataFrame({
                    'id': df_recent['id'],
                    'rank_change': np.nan,
                    'rank_percent_change': np.nan
                })
        return merged

#%%

#%%


# if __name__=='__main__':
    # df = extract_token_data(token_id='bitcoin')
    # historical_data = compare_token_data(df.iloc[0], df.iloc[-1])  # Comparison between the first and last data point
    # top_movers_duration = update_top_movers_with_duration()
    # link_and_plot(top_movers_duration, ['market_cap_rank', 'total_volume'])

    # regular_top_movers()
    # hourly_sheets_update_small_mcap(test=True, recent=1, old=2)
    # df = compare_arbitrary(recent=1, old=2, threshold_sup = 1500)    
    
    