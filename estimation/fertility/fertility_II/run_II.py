from estimation.fertility.fertility_II import PMMH_fertility_II
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

LIST_OF_STAGE_II_COUNTRIES = pd.read_csv("../../../data/UN/fertility/countries_in_stage_II.csv")
SAVING_FILE_PREFIX = "../../../output/estimation/fertility_II/PMMH_fertility_II_"  # Add the country name to this prefix
TFR_OF_STAGE_II_COUNTRIES = pd.read_csv("../../../data/UN/fertility/TFR_per_country_stage_II.csv")


def run(country):
    SAVING_FILE = SAVING_FILE_PREFIX + country + ".pkl"
    if os.path.exists(SAVING_FILE):
        return
    DATA = TFR_OF_STAGE_II_COUNTRIES[TFR_OF_STAGE_II_COUNTRIES['Region'] == country]['TFR']
    DATA = np.array(DATA)

    my_pmmh = PMMH_fertility_II.run_PMMH(DATA, niter=1000, N_particles=100)

    with open(f'{SAVING_FILE}', 'wb') as handle:
        pickle.dump(my_pmmh.chain, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for country in tqdm(LIST_OF_STAGE_II_COUNTRIES['Region']):
        run(country)
