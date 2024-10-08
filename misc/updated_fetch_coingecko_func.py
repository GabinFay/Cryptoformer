# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:26:57 2024

@author: Gabin
"""

import time
import requests

def fetch_coingecko_data(vs_currency='usd', category=None, per_page=250, page=1, price_change_percentage='1h,24h,7d', locale='en', precision=3):
    for attempt in range(4):
        try:
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
            if response.status_code == 200 and response.content:
                data = response.json()
                df = pd.DataFrame(data)
                df['fetched_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df['market_cap_rank'] = df.index + 1 + (page-1)*per_page  # Adjust market cap rank based on page number
                df.set_index('id', inplace=True)
                return df
            else:
                raise ValueError("Empty or invalid response")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error occurred - {str(e)}")
            if attempt < 3:
                time.sleep(10)  # wait for 10 seconds before retrying
            else:
                raise
