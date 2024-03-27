import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "LE_per_country.csv"
dataframe = pd.read_csv(FILENAME)

first = dataframe.groupby('Region').first()
plt.hist(first['LE'], bins=20, density=True)
theta = stats.gumbel_r.fit(first['LE'])
print(theta)

xs = np.linspace(0, 70)
plt.plot(xs, stats.gumbel_r.pdf(xs, *theta), label='gamma')
plt.show()
