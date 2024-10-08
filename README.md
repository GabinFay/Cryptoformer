# Cryptoformer  

Fetches the top 5000 on coingekco every hour and stores them in a postgresSQL DB on a server  
Automatically adds to a google sheets which tokens have experienced the biggest rank increase (+ relative increase in %), it's a powerful metric to detect trends (& can then be extended to other crypto data sources but also reddit for fun useful life info (see /scraping)  
Then, the idea was/is to train a transformer model to classify wether a token wil experience a drastic decrease in price/rank after a given horizon (a day, 12h, a week ?), will stay roughly neutral or will explode in price   

Had to quit for the summer, still cooking !
