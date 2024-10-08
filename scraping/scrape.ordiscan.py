# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:09:20 2024

@author: Gabin
"""

import requests

# The URL you're trying to access
url = 'https://ordiscan.com/api/trpc/brc20.getTokens?batch=1&input=%7B%220%22%3A%7B%22json%22%3A%7B%22order%22%3A%22market-cap%22%2C%22cursor%22%3A60%7D%7D%7D'

# Headers from your browser's request
headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Host': 'ordiscan.com',
    'Referer': 'https://ordiscan.com/brc20',
    'Sec-Ch-Ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'X-Ordiscan-Tsit': 'yJOEEg',
    # Add any other headers observed in the request
}

# Make the request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Process the response
    data = response.json()
    print(data)
else:
    print(f'Failed to fetch data: {response.status_code}')
