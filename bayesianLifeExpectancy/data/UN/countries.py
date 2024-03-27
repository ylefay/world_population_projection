import pandas as pd

FILENAME = "LE_per_country.csv"
dataframe = pd.read_csv(FILENAME)
countries = dataframe[dataframe["Year"] == 2021]["Region"]
countries.to_csv("countries.csv", index=False)
