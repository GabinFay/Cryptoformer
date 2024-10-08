#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:39:35 2024

@author: gabinfay
"""

import requests
import pandas as pd
from datetime import datetime
from credentials import COINGECKO_KEY  # Import the API key
import time
import argparse

def fetch_categories(debug=False):
    url = "https://api.coingecko.com/api/v3/coins/categories?order=market_cap_desc"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": COINGECKO_KEY
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    df['fetched_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['rank'] = range(1, len(df) + 1)  # Create a category rank column
    df.set_index('id', inplace=True)
    if debug:
        print(df.head())
        return df
    return df[["name", "rank", "market_cap", "market_cap_change_24h", "content", "volume_24h", "updated_at", 'fetched_timestamp']]

def archive_category_data(debug=False):
    df = fetch_categories(debug)
    if not debug:
        filename = f"data/categories/category_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        df.to_csv(filename)
        print(f"Data stored in {filename}")
    else:
        print("Debug mode: Data not saved")

def fetch_coingecko_data(vs_currency='usd', category=None, per_page=250, page=1, price_change_percentage='1h,24h,7d', locale='en', precision=3, debug=False):
    # Constructing the URL with parameters
    if category:
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}&category={category}&per_page={per_page}&page={page}&price_change_percentage={price_change_percentage}&locale={locale}&precision={precision}"
    else:
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}&per_page={per_page}&page={page}&price_change_percentage={price_change_percentage}&locale={locale}&precision={precision}"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": COINGECKO_KEY  # Using the imported API key
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    # Adding fetched timestamp
    df['market_cap_rank'] = df.index + 1 + (page-1)*per_page  # Adjust market cap rank based on page number
    df.set_index('id', inplace=True)
    if debug:
        print(df.head())
    return df

def fetch_and_store_top_tokens(vs_currency='usd', pages=20, category=None, debug=False):
    dfs = []
    for page in range(1, pages + 1):
        df = fetch_coingecko_data(vs_currency=vs_currency, category=category, page=page, debug=debug)
        dfs.append(df)
    
    big_df = pd.concat(dfs)
    big_df['fetched_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not debug:
        # Identifiable name with date
        if category==None:
            filename = f"data/data5000/top_5000_tokens_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        else:
            filename = f"data/{category}/top_250_tokens_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        big_df.to_csv(filename)
        print(f"Data stored in {filename}")
    else:
        print(f"Debug mode: {'Top 5000 tokens' if category is None else f'Top 250 tokens for {category}'} data not saved")
        print(big_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and store cryptocurrency data")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (no data saved)")
    args = parser.parse_args()

    archive_category_data(args.debug)

    fetch_and_store_top_tokens(pages=20, category=None, debug=args.debug)
    fetch_and_store_top_tokens(pages=1, category='brc-20', debug=args.debug)
    fetch_and_store_top_tokens(pages=1, category='base-meme-coins', debug=args.debug)
    fetch_and_store_top_tokens(pages=1, category="solana-meme-coins", debug=args.debug)
    fetch_and_store_top_tokens(pages=1, category="politifi", debug=args.debug)
    fetch_and_store_top_tokens(pages=1, category="ton-meme-coins", debug=args.debug)

    time.sleep(75)

    # ... rest of the function calls with debug=args.debug ...

# Fetch and store top 5000 tokens
archive_category_data()

#%%
fetch_and_store_top_tokens(pages=20, category=None)
fetch_and_store_top_tokens(pages=1, category='brc-20')
fetch_and_store_top_tokens(pages=1, category='base-meme-coins')
fetch_and_store_top_tokens(pages=1, category="solana-meme-coins")
fetch_and_store_top_tokens(pages=1, category="politifi")
fetch_and_store_top_tokens(pages=1, category="ton-meme-coins")

time.sleep(75)

fetch_and_store_top_tokens(pages=1, category="layer-3-l3")
fetch_and_store_top_tokens(pages=1, category="meme-token")
fetch_and_store_top_tokens(pages=1, category="zero-knowledge-zk")
fetch_and_store_top_tokens(pages=1, category="ton-ecosystem")

fetch_and_store_top_tokens(pages=1, category="elon-musk-inspired-coins") # elon halfly fetched

# less fetched : api rate reached issue
fetch_and_store_top_tokens(pages=1, category="cat-themed-coins")
fetch_and_store_top_tokens(pages=1, category="parody-meme-coins")
fetch_and_store_top_tokens(pages=1, category="degen-ecosystem")
fetch_and_store_top_tokens(pages=1, category="bitcoin-ecosystem")
fetch_and_store_top_tokens(pages=1, category="runes")


