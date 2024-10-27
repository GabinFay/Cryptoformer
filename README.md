# Cryptoformer : leaderboard rank increase forecast for crypto tokens 

Fetches and stores the top 5000 tokens on the coingecko leaderboard every hour
Computes the rank increase with different timescales (hourly, daily, weekly) : because it's relative to the performance of neighbouring tokens, it's more powerful than just tracking price increase  
Create a dataset with classes : will a token : looses price by more than -30% the next day, stay roughly neutral, will increase by more than +30%
Idea is that you can't forecast the price, but you can detect FOMO and exploding tokens might have a common pattern
Transformer model to process the time series, compute the enriched last timestep vector with attention, and use this enriched 1D vector for each token to map it to the category and train a simple classifier
