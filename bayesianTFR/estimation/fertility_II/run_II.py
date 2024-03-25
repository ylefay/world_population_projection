from bayesianTFR.estimation.fertility_II import PMMH_fertility_II
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import yaml

LIST_OF_STAGE_II_COUNTRIES = pd.read_csv("../../data/UN/countries_in_stage_II.csv")
SAVING_FILE_PREFIX = "../../output/estimation/fertility_II/PMMH_fertility_II_"  # Add the country name to this prefix
TFR_OF_STAGE_II_COUNTRIES = pd.read_csv("../../data/UN/TFR_per_country_stage_II.csv")

with open('../../parameters/fertility_II.yaml', 'r') as yaml_config:
    fertility_II_config = yaml.safe_load(yaml_config)


def run(country):
    SAVING_FILE = SAVING_FILE_PREFIX + country + ".pkl"
    if os.path.exists(SAVING_FILE):
        return

    DATA = TFR_OF_STAGE_II_COUNTRIES[TFR_OF_STAGE_II_COUNTRIES['Region'] == country]
    starting_period = DATA['Year'].iloc[0]
    DATA = DATA['TFR']
    DATA = np.array(DATA)

    kwargs = {'t0': starting_period, 'tau_c': fertility_II_config[country]['tau_c']}
    my_pmmh = PMMH_fertility_II.run_PMMH(DATA, niter=1000, N_particles=100, **kwargs)

    with open(f'{SAVING_FILE}', 'wb') as handle:
        pickle.dump(my_pmmh.chain, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for country in tqdm(LIST_OF_STAGE_II_COUNTRIES['Region']):
        run(country)
