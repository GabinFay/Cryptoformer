# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:33:30 2024

@author: Gabin
"""

import requests
import pandas as pd
import time
from datetime import datetime
import random

base_url = "https://ordinalscan.net/api/home/brc20/ranking"
headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Referer": "https://ordinalscan.net/ranking/brc20",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin"
}

all_data = pd.DataFrame()


for page in range(1, 21):
    params = {
        "limit": 10,
        "page": page,
        "sort": "desc"
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    
    page_data = pd.json_normalize(response.json()['data']['items'])
    all_data = pd.concat([all_data, page_data], ignore_index=True)
    
    time.sleep(random.uniform(0.7, 1.3))

all_data.set_index('tick', inplace=True)
all_data['market_cap_rank'] = range(1, len(all_data) + 1)  # Create a category rank column

filename = f"data/ordinalscan/top_tokens_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
all_data.to_csv(filename)
print(f"Data stored in {filename}")



