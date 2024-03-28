from bayesianLifeExpectancy.estimation import PMMH
import pandas as pd
import pickle
from tqdm import tqdm
import os
import yaml
import numpy as np

LIST_OF_COUNTRIES = pd.read_csv("../data/UN/countries.csv")
SAVING_FILE_PREFIX = "../output/estimation/PMMH_"
LE = pd.read_csv("../data/UN/LE_per_country.csv")

with open('../parameters/parameters.yaml', 'r') as yaml_config:
    config = yaml.safe_load(yaml_config)


def run(country):
    SAVING_FILE = SAVING_FILE_PREFIX + country + ".pkl"
    if os.path.exists(SAVING_FILE):
        return

    DATA = np.array(LE[LE['Region'] == country]['LE'])
    my_chain = PMMH.run_PMMH(DATA, niter=10000, N_particles=100, TRY=1)

    with open(f'{SAVING_FILE}', 'wb') as handle:
        pickle.dump(my_chain, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for country in tqdm(LIST_OF_COUNTRIES['Region']):
        if country == 'WORLD':
            run(country)
