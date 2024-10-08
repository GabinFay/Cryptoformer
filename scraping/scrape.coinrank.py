#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:14:04 2024

@author: gabinfay
"""

import requests
from bs4 import BeautifulSoup

url = 'https://cryptorank.io/gainers'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Example of finding a specific element, adjust as needed
for row in soup.find_all('tr'):
    print(row.text)
