#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:36:40 2024

@author: gabinfay
"""

from _database import Database

db = Database(user='coingecko', password='coingecko', dbname='coingecko')
db.add_csv_to_db(path='data5000')
db.add_csv_to_db(path='brc-20')
db.add_csv_to_db(path='base-meme-coins')
db.add_csv_to_db(path="solana-meme-coins")
db.add_csv_to_db(path="politifi")
db.add_csv_to_db(path="ton-meme-coins")
db.add_csv_to_db(path="layer-3-l3")
db.add_csv_to_db(path="meme-token")
db.add_csv_to_db(path="zero-knowledge-zk")
db.add_csv_to_db(path="ton-ecosystem")
db.add_csv_to_db(path="elon-musk-inspired-coins")
db.add_csv_to_db(path="cat-themed-coins")
db.add_csv_to_db(path="parody-meme-coins")
db.add_csv_to_db(path="degen-ecosystem")
db.add_csv_to_db(path="bitcoin-ecosystem")
db.add_csv_to_db(path="runes")
