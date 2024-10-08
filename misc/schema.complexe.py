sql_schema = """
CREATE TABLE cryptocurrencies (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(255),
    name VARCHAR(255),
    image TEXT,
    current_price DECIMAL,
    market_cap BIGINT,
    market_cap_rank INT,
    fully_diluted_valuation BIGINT,
    total_volume BIGINT,
    high_24h DECIMAL,
    low_24h DECIMAL,
    price_change_24h DECIMAL,
    price_change_percentage_24h DECIMAL,
    market_cap_change_24h BIGINT,
    market_cap_change_percentage_24h DECIMAL,
    circulating_supply BIGINT,
    total_supply BIGINT,
    max_supply BIGINT,
    ath DECIMAL,
    ath_change_percentage DECIMAL,
    ath_date TIMESTAMP,
    atl DECIMAL,
    atl_change_percentage DECIMAL,
    atl_date TIMESTAMP,
    roi TEXT,
    last_updated TIMESTAMP,
    price_change_percentage_1h_in_currency DECIMAL,
    price_change_percentage_24h_in_currency DECIMAL,
    price_change_percentage_7d_in_currency DECIMAL,
    fetched_timestamp TIMESTAMP
);
"""
