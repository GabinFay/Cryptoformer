# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:24:18 2024

@author: Gabin
"""

import requests
import pandas as pd

# URL and headers as specified
url = "https://www.okx.com/priapi/v1/nft/inscription/rc20/tokens?scope=4&page=1&size=50&sortBy=&sort=&tokensLike=&timeType=1&tickerType=4&walletAddress=&t=1713972041106"
headers = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "App-Type": "web",
    "Authorization": "eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJleDExMDE3MTMxOTkwNjMzNTg1NDhFMzBCREREREE2MUVBMURzZEsiLCJ1aWQiOiJFSFdjN2ZkSlkwL0NsZUh5QktCaU1nPT0iLCJzdGEiOjAsIm1pZCI6IkVIV2M3ZmRKWTAvQ2xlSHlCS0JpTWc9PSIsImlhdCI6MTcxMzE5OTA2MywiZXhwIjoxNzEzODAzODYzLCJiaWQiOjAsImRvbSI6Ind3dy5va3guY29tIiwiZWlkIjo3LCJpc3MiOiJva2NvaW4iLCJkaWQiOiJIdjZqRjkwM05YVFRrMFlqa1dpQ2cwdmdNRmpFNndqRnc2eWpZeVdkUmFFREE0T2V2UDBDSzRMNUNiZFk0Ky9vIiwibGlkIjoiRUhXYzdmZEpZMC9DbGVIeUJLQmlNZz09IiwidWZiIjoiUFR5QThXMDl6RlVKQkdKNllSTkdZdz09IiwidXBiIjoiaUJyYTJWaE5va3lSaWh4aUovM3pFdz09Iiwia3ljIjowLCJzdWIiOiJGNEZBQTc5ODM1Njk3MkFGMjRGNDQwMjg2RjY4N0VERiJ9.7cQm4Vbtt8VBSUYsf-P4fAmNtl7S0w8hzOg0zASHRMfOUTzfvK09eTM1PwBhwW_T2xRr3sQuRXB335Ie7O5K9w",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# Send GET request
response = requests.get(url, headers=headers)

# Convert response to JSON
data_json = response.json()

# Extract data to DataFrame
df = pd.DataFrame(data_json['data']['list'])

# Save DataFrame to CSV file
df.to_csv('output.csv', index=False)
