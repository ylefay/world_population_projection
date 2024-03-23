import pandas as pd

# Loading the data: UN Age-specific fertility rates by country, by 5 years, for 1 000 women.
FILENAME = "WPP2022_FERT_F02_FERTILITY_RATES_BY_5-YEAR_AGE_GROUPS_OF_MOTHER.csv"

dataframe = pd.read_csv(FILENAME, encoding='utf_8')
dataframe.set_index('Index', inplace=True)
dataframe.drop(
    ['Variant', 'Location code', 'Notes', 'ISO2 Alpha-code', 'ISO3 Alpha-code', 'SDMX code**', 'Type', 'Parent code'],
    axis=1, inplace=True)
dataframe.rename(columns={'Region, subregion, country or area *': 'Region'}, inplace=True)
dataframe.dropna(inplace=True)
age_columns = ['10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54']
dataframe = dataframe.astype({age_range: float for age_range in age_columns})
dataframe['TFR'] = dataframe[age_columns].sum(axis=1) / 1000
dataframe = dataframe[['Region', 'Year', 'TFR']]
dataframe['Region'] = dataframe['Region'].str.replace("/", "_")
dataframe['Region'] = dataframe['Region'].str.replace(" ", "_")
dataframe.to_csv('TFR_per_country.csv', index=False)
