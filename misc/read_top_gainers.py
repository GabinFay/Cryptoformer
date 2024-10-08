import pandas as pd
from _database import Database

db = Database()

query = """
SELECT id, COUNT(*) AS times_top_gainer
FROM top_gainers
GROUP BY id
ORDER BY times_top_gainer DESC;
"""

# Execute the SQL query and read into a DataFrame
df = pd.read_sql(query, db.engine)

# query = """
# SELECT id, COUNT(*) AS times_top_gainer
# FROM top_gainers
# WHERE fetched_timestamp >= NOW() - INTERVAL '1 day'
# GROUP BY id;
# """

# # Execute the SQL query and read into a DataFrame
# df = pd.read_sql(query, db.engine)