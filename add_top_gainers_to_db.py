from coingecko import compare_arbitrary_top_gainers_sheet
from _database import Database
import pandas as pd

db=Database()

# for i in range(2, 459):
#     df = compare_arbitrary_top_gainers_sheet(recent=i-1, old=i, n=35)
#     db.copy_from_stringio(df, 'top_gainers')

df = compare_arbitrary_top_gainers_sheet(recent=1, old=2, n=35)
db.copy_from_stringio(df, 'top_gainers')