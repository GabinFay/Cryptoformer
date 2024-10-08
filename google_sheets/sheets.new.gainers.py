#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:07:42 2024

@author: gabinfay
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from _database import Database

db_ssh = Database(ssh_pem_path='BotAmazon.pem', remote=True)
db = Database()

#%%

# Assuming you have a SQLAlchemy engine created as 'engine'
query = """
SELECT * FROM data5000
WHERE fetched_timestamp >= '2024-04-23'
;
"""

df = pd.read_sql_query(query, db.engine)

#%%

from coingecko import save_df_to_sheets

save_df_to_sheets()







