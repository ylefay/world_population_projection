import pandas as pd

FILENAME = "LE_per_country.csv"
dataframe = pd.read_csv(FILENAME)
dataframe = dataframe[dataframe["Year"] == 2021]["Region"]
dataframe.to_csv("countries.csv", index=False)
