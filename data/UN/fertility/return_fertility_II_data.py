import pandas as pd

FILENAME = "TFR_per_country.csv"

dataframe = pd.read_csv(FILENAME)

dataframe["TFR"] *= 5

countries_in_stage_II = dataframe[dataframe["TFR"] > 2][dataframe["Year"] == 2021]["Region"]

dataframe = dataframe[dataframe["Region"].isin(countries_in_stage_II)]

dataframe.to_csv('TFR_per_country_stage_II.csv', index=False)

countries_in_stage_II.to_csv('countries_in_stage_II.csv', index=False)
