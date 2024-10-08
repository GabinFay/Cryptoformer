# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:08:35 2024

@author: Gabin
"""

import requests
import pandas as pd

def fetch_token_data(token_id):
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}?sparkline=true"
    headers = {'accept': 'application/json', 'x-cg-demo-api-key': 'CG-SvuUKZyuHRZ2JLdvovtt5LwA'}
    response = requests.get(url, headers=headers)
    return response.json()

def load_data_as_dataframe(token_id):
    data = fetch_token_data(token_id)
    df = pd.json_normalize(data)
    return df

df = load_data_as_dataframe('joe-coin')

# List of language and currency codes to filter out
codes = ["de", "es", "fr", "it", "pl", "ro", "hu", "nl", "pt", "sv", "vi", "tr", "ru", "ja", "zh", "zh-tw", "ko", "ar", "th", "id", "cs", "da", "el", "hi", "no", "sk", "uk", "he", "fi", "bg", "hr", "lt", "sl", "aed", "ars", "aud", "bch", "bdt", "bhd", "bmd", "bnb", "brl", "btc", "cad", "chf", "clp", "cny", "czk", "dkk", "dot", "eos", "eth", "eur", "gbp", "gel", "hkd", "huf", "idr", "ils", "inr", "jpy", "krw", "kwd", "lkr", "ltc", "mmk", "mxn", "myr", "ngn", "nok", "nzd", "php", "pkr", "pln", "rub", "sar", "sek", "sgd", "thb", "try", "twd", "uah", "vef", "vnd", "xag", "xau", "xdr", "xlm", "xrp", "yfi", "zar", "bits", "link", "sats"]

# Constructing the filter condition
filter_condition = df.columns.str.contains('|'.join([fr"\.{code}$" for code in codes]))



# Applying the filter to keep only the desired columns
filtered_df = df.loc[:, ~filter_condition]
