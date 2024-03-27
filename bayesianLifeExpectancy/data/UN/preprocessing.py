import pandas as pd

FILENAME = "WPP2022_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.csv"

dataframe = pd.read_csv(FILENAME, encoding='utf_8')
dataframe.set_index('Index', inplace=True)
dataframe.drop([str(i) for i in range(1, 100)], axis=1, inplace=True)
dataframe.drop(
    ['ISO3 Alpha-code', 'ISO2 Alpha-code', 'SDMX code**', 'Type', 'Variant', 'Location code', 'Notes', 'Parent code',
     '100+'], axis=1, inplace=True)
dataframe.rename(columns={'0': 'LE', 'Region, subregion, country or area *': 'Region'}, inplace=True)
dataframe.dropna(inplace=True)
dataframe.to_csv('LE_per_country.csv', index=False)
