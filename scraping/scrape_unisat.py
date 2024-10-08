# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:59:59 2024

@author: Gabin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:39:37 2024

@author: Gabin
"""

import requests
import pandas as pd

url = "https://api.unisat.io/market-v4/runes/auction/runes_types_many"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://unisat.io/",
    "X-Appid": "1adcd7969603261753f1812c9461cd36",
    "X-Channel": "UniSat",
    "X-Front-Version": "195",
    "X-Sign": "bdd308be61a41b551290aa99da5c3b55",
    "X-Ts": "1713969879",
    "Cf-Token": "wj71zb1btfm4ad18urd1nlv3d",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
response = requests.post(url, headers=headers)
data = response.json()['data']['list']
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)
