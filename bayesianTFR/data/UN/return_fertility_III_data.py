import pandas as pd

FILENAME = "TFR_per_country.csv"

dataframe = pd.read_csv(FILENAME)

dataframe["TFR"] *= 5

dataframe = dataframe[dataframe["TFR"] <= 2]
dataframe.to_csv('TFR_per_country_stage_III.csv', index=False)

dataframe = dataframe['Region'].drop_duplicates()
dataframe.to_csv('countries_in_stage_III.csv', index=False)
