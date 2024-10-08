from _database import Database
from schema import sql_schema

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

# db.create_table_if_not_exists(sql_schema)

import os
import pandas as pd

directory_path = 'data/data5000'
file_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
file_paths.sort()

df_list = []
for file_path in file_paths:
    df_temp = pd.read_csv(file_path)
    df_list.append(df_temp)
df = pd.concat(df_list, ignore_index=True)

db.copy_from_stringio(df, 'cryptocurrencies')