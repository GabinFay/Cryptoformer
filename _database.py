import psycopg2
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from io import StringIO
import xml.etree.ElementTree as ET

import pandas as pd
import glob
import re
from datetime import datetime
import numpy as np

from coingecko import get_sorted_files, add_rank_change_columns
from memory_profiler import profile

import csv

# from configparser import ConfigParser
# config = ConfigParser()

# try:
#     config.read('../config/keys_config.cfg')
#     config.get('deutsche_bahn', 'client_id') #EXAMPLE OF HOW TO ACCESS THE KEYS
# except:
#     config.read('../../config/keys_config.cfg')
from credentials import db_user, db_password, db_name, db_ip, ssh_username


class Database:
    def __init__(self, user=db_user, password=db_password, dbname=db_name, remote=False, ip=db_ip, ssh_port=22, ssh_username=ssh_username, ssh_password=None, ssh_pem_path=None): #HARDCODED
        self.user = user
        self.password = password
        self.dbname = dbname
        self.server = None  # SSH tunnel reference
        if remote:
            self.conn, self.engine = self.connect_remote_db(ip, ssh_port, ssh_username, ssh_password, ssh_pem_path)
        else:
            self.conn, self.engine = self.connect_local_db()
    
    def add_single_csv_to_db(self, i, recent_file, list_of_files, table_name='data5000'):
        with open(recent_file) as recent:
            df_recent = pd.read_csv(recent)
            for hours_ago, suffix in zip([1, 24, 7*24, -1], ['1h', '1d', '7d', 'start']):
                df_old = None
                if hours_ago == -1:
                    with open(list_of_files[-1]) as old:
                        df_old = pd.read_csv(old)
                elif i + hours_ago < len(list_of_files):
                    with open(list_of_files[i + hours_ago]) as old:
                        df_old = pd.read_csv(old)
                df_update = add_rank_change_columns(df_recent, df_old)
                df_recent[f'rank_change_{suffix}'] = df_update['rank_change']
                df_recent[f'rank_percent_change_{suffix}'] = df_update['rank_percent_change']
                if df_old is not None:
                    del df_old
            self.copy_from_stringio(df_recent, table_name)
            del df_recent
    
    def add_csv_to_db(self, path='data5000', all_csv=False):
        table_name = path.replace("-", "_")
        sql_schema = get_sql_schema(table_name)
        self.create_table_if_not_exists(sql_schema)
        if path == 'data5000':
            prefix = 'top_5000_tokens'
        else:
            prefix = 'top_250_tokens'
        list_of_files = get_sorted_files(path=path, prefix=prefix)
        if not all_csv:
            self.add_single_csv_to_db(0, list_of_files[0], list_of_files, table_name=table_name)
        else:
            for i, recent_file in enumerate(list_of_files):
                self.add_single_csv_to_db(i, recent_file, list_of_files, table_name=table_name)

    def add_categories_to_db(self, all_csv=False):
        path='categories'
        self.create_table_if_not_exists(category_schema)
        prefix = 'category_data'
        list_of_files = get_sorted_files(path=path, prefix=prefix)
        if not all_csv:
            self.add_single_category_to_db(list_of_files[0])
        else:
            for filename in list_of_files:
                self.add_single_category_to_db(filename)

    def add_single_category_to_db(self, filename):
        table_name='categories'
        with open(filename) as recent:
            df_recent = pd.read_csv(recent, sep=',', quotechar='"')
            self.copy_from_stringio(df_recent, table_name)

    def connect_local_db(self):
        conn = psycopg2.connect(database=self.dbname, user=self.user, host="localhost", password=self.password)
        engine = create_engine(f'postgresql://{self.user}:{self.password}@localhost:5432/{self.dbname}', 
                               echo=False)
        return conn, engine
    
    def connect_remote_db(self, ip, ssh_port, ssh_username, ssh_password, ssh_pem_path):
        try:
            if ssh_password:
                self.server = SSHTunnelForwarder(
                    (ip, ssh_port),
                    ssh_username=ssh_username,
                    ssh_password=ssh_password,
                    remote_bind_address=('localhost', 5432))
            elif ssh_pem_path:
                self.server = SSHTunnelForwarder(
                (ip, ssh_port),
                ssh_username=ssh_username,
                ssh_pkey=ssh_pem_path,
                remote_bind_address=('localhost', 5432))
            
            self.server.start()
            params = {
                'database': self.dbname,
                'user': self.user,
                'password': self.password,
                'host': 'localhost',
                'port': self.server.local_bind_port
                }
        
            conn = psycopg2.connect(**params)
            engine = create_engine(f'postgresql://{self.user}:{self.password}@localhost:{self.server.local_bind_port}/{self.dbname}', 
                                   echo=False)
            return conn, engine
        except Exception as e:
            print(f"Connection to the remote server failed: {e}")
            raise Exception(f"Failed to connect: {e}")

    
    # @profile
    def copy_from_stringio(self, df, table):
        """
        Copy DataFrame into a table defined in the schema of the db.
        """
        # Save dataframe to an in-memory buffer
        buffer = StringIO()
        df.to_csv(buffer, index_label='id', header=False, index=False, sep=';')
        buffer.seek(0)
        # Use the existing cursor
        try:
            with self.conn.cursor() as cursor:
                # cursor.execute(f"""SELECT * FROM {table}""")
                cursor.copy_from(buffer, table, sep=";", null='')
                self.conn.commit()
                print("Successfully added the DataFrame into the table")
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
            # self.conn.rollback() #no need to rollback cause the with statement already rollbacks
            return 1
        
    def create_table_if_not_exists(self, sql_schema):
        for statement in sql_schema.split(';')[:-1]:
            tokens = statement.split()
            if 'CREATE' in tokens and 'TABLE' in tokens:
                table_name_index = tokens.index('TABLE') + 1
                if table_name_index < len(tokens):
                    table_name = tokens[table_name_index].strip('"').split('(')[0]
                    if not self.table_exists(table_name):
                        try:
                            with self.conn.cursor() as cursor:
                                cursor.execute(statement)
                                self.conn.commit()
                                print(f"Created table {table_name}")
                        except psycopg2.Error as err:  # Catch all psycopg2 errors
                            print("An error occurred:", err)
                            self.conn.rollback()  # Rollback transaction on error
                    else:
                        print(f"Table {table_name} already exists. Skipping creation.")
                else:
                    print("Could not determine table name from SQL statement.")
            else:
                print("No CREATE TABLE statement found.")

    def __del__(self):
        # Destructor to ensure cleanup
        self.stop_remote_connection()

        # Safely incorporating table names into queries
    def is_safe_table_name(self, table_name):
        # Define a list of allowed table names
        allowed_table_names = ['xmlchanges', 'xmltimetable']
        return table_name in allowed_table_names
    
    def table_exists(self, table_name):
        # This query checks if the given table name exists in the database
        try:
            with self.conn.cursor() as cursor:
                query = """
                SELECT EXISTS (
                    SELECT FROM pg_catalog.pg_tables 
                    WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'
                    AND tablename = %s
                );
                """
                cursor.execute(query, (table_name,))
                result = cursor.fetchone()[0]
                if result:
                    return True
                else:
                    # print("No data found for the specified date.")
                    return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def stop_remote_connection(self):
        if self.conn:
            self.conn.close()
        if self.engine:
            self.engine.dispose()
        if self.server:
            self.server.stop()


    # def create_table(self, sql_schema):
    #     for i in sql_schema.split(';'):
    #         try:
    #             with self.conn.cursor() as cursor:
    #                 cursor.execute(i)
    #                 self.conn.commit()
    #         except psycopg2.ProgrammingError as err:
    #             # print("we caught the exception : don't mind")
    #             # print(err)
    #             self.conn.rollback()

# WARNING : danger zone        
    # def reset_database(self):
    #     with self.conn.cursor() as cursor:
        #     cursor.execute('DROP SCHEMA public CASCADE')
        #     self.conn.commit()
        #     cursor.execute('CREATE SCHEMA public')
        #     self.conn.commit()
        #     cursor.execute('GRANT ALL ON SCHEMA public TO postgres')
        #     self.conn.commit()
        #     cursor.execute('GRANT ALL ON SCHEMA public TO public')
        #     self.conn.commit()

category_schema = """
CREATE TABLE categories (
    id VARCHAR(255),
    name VARCHAR(255),
    rank INT,
    market_cap NUMERIC,
    market_cap_change_24h NUMERIC,
    content TEXT,
    volume_24h NUMERIC,
    updated_at TIMESTAMP,
    fetched_timestamp TIMESTAMP
);
"""


def get_sql_schema(table_name):
    print(table_name)
    return f"""
CREATE TABLE {table_name} (
    id VARCHAR(255),
    symbol VARCHAR(255),
    name VARCHAR(255),
    image TEXT,
    current_price NUMERIC,
    market_cap NUMERIC,
    market_cap_rank NUMERIC,
    fully_diluted_valuation NUMERIC,
    total_volume NUMERIC,
    high_24h NUMERIC,
    low_24h NUMERIC,
    price_change_24h NUMERIC,
    price_change_percentage_24h NUMERIC,
    market_cap_change_24h NUMERIC,
    market_cap_change_percentage_24h NUMERIC,
    circulating_supply NUMERIC,
    total_supply NUMERIC,
    max_supply NUMERIC,
    ath NUMERIC,
    ath_change_percentage NUMERIC,
    ath_date TIMESTAMP,
    atl NUMERIC,
    atl_change_percentage NUMERIC,
    atl_date TIMESTAMP,
    roi TEXT,
    last_updated TIMESTAMP,
    price_change_percentage_1h_in_currency NUMERIC,
    price_change_percentage_24h_in_currency NUMERIC,
    price_change_percentage_7d_in_currency NUMERIC,
    fetched_timestamp TIMESTAMP,
    rank_change_1h FLOAT,
    rank_percent_change_1h FLOAT,
    rank_change_1d FLOAT,
    rank_percent_change_1d FLOAT,
    rank_change_7d FLOAT,
    rank_percent_change_7d FLOAT,
    rank_change_start FLOAT,
    rank_percent_change_start FLOAT
);
"""
    

if __name__ == '__main__':
    db = Database(user='coingecko', password='coingecko', dbname='coingecko')    
    # db.add_csv_to_db(path='data5000', all_csv=True)
    # db.add_csv_to_db(path='brc-20', all_csv=True)
    # db.add_csv_to_db(path='base-meme-coins', all_csv=True)
    # db.add_csv_to_db(path="solana-meme-coins", all_csv=True)
    # db.add_csv_to_db(path="politifi", all_csv=True)
    # db.add_csv_to_db(path="ton-meme-coins", all_csv=True)
    # db.add_csv_to_db(path="layer-3-l3", all_csv=True)
    # db.add_csv_to_db(path="meme-token", all_csv=True)
    # db.add_csv_to_db(path="zero-knowledge-zk", all_csv=True)
    # db.add_csv_to_db(path="ton-ecosystem", all_csv=True)
    # db.add_csv_to_db(path="elon-musk-inspired-coins", all_csv=True)
    # db.add_csv_to_db(path="cat-themed-coins", all_csv=True)
    # db.add_csv_to_db(path="parody-meme-coins", all_csv=True)
    # db.add_csv_to_db(path="degen-ecosystem", all_csv=True)
    # db.add_csv_to_db(path="bitcoin-ecosystem", all_csv=True)
    db.add_categories_to_db(all_csv=True)






