import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

"""
Explore the distribution accross stage III countries of the initial TFR
Fitting an exponential distribution for 2-TFR.
We find, across stage III countries, a scale parameter equal to 0.04029496402877704.
This parameter will later be reused to define the stage III prior.
"""

FILENAME = "TFR_per_country_stage_II.csv"
dataframe = pd.read_csv(FILENAME)

first = dataframe.groupby('Region').first()
theta = stats.norm.fit(first['TFR'])
plt.hist(first['TFR'])
plt.show()

print(*theta)
xs = np.linspace(2, 9, 10)
plt.plot(xs, stats.norm.pdf(xs, *theta), label='gamma')
plt.show()
