from pandas.plotting import autocorrelation_plot
# import lag_plot function
from pandas.plotting import lag_plot

# lag scatter plot
lag_plot(sales_data)
plt.show()
# autocorrelation plot
autocorrelation_plot(sales_data)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Create a sample DataFrame
data = {'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        'Series1': [1, 2, 3],
        'Series2': [4, 5, 6],
        'Series3': [7, 8, 9]}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot a stacked area chart
df.plot(kind='area', stacked=True)
plt.show()
