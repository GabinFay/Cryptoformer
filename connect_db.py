#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:38:11 2024

@author: gabinfay
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from credentials import db_user, db_password, db_name, db_ip, ssh_username

from _database import Database
# from schema import sql_schema

db =  Database(user=db_user,
               password=db_password,
               dbname=db_name,
               remote=False,
               ip=db_ip,
               ssh_port=22,
               ssh_username=ssh_username,
               ssh_password=None)

# Connect to PostgreSQL database
# engine = create_engine('postgresql://username:password@host:port/dbname')
query = "SELECT * FROM cryptocurrencies"
df = pd.read_sql(query, db.engine)

# Ensure fetched_timestamp is of a consistent type, ideally datetime if not already
df['fetched_timestamp'] = pd.to_datetime(df['fetched_timestamp'])

df = df.drop_duplicates(subset=['id', 'fetched_timestamp'], keep='first')

# Use pivot instead of pivot_table to avoid aggregation
pivoted_df = df.pivot(index='id', columns='fetched_timestamp')

#%%
# Optional: Flatten the columns multi-index if necessary
pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]

# Now, each feature for each token at each timestamp is a separate column
# Convert to 3D numpy array if needed
data_3d = np.stack([pivoted_df[col].values for col in sorted(pivoted_df.columns.unique()) if col.startswith('feature')], axis=-1)
