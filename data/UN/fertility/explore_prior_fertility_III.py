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

FILENAME = "TFR_per_country_stage_III.csv"
dataframe = pd.read_csv(FILENAME)

first = dataframe.groupby('Region').first()

plt.hist(first['TFR'])
plt.show()
plt.hist(2-first['TFR'])
theta = stats.expon.fit(2-first['TFR'])
print(*theta)
xs = np.linspace(0, 0.2, 10)
plt.plot(xs, stats.expon.pdf(xs, *theta), label='gamma')
plt.show()