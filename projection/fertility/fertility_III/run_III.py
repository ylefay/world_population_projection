import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
from projection.fertility.fertility_III import fertility_III

LIST_OF_STAGE_III_COUNTRIES = pd.read_csv("../../../data/UN/fertility/countries_in_stage_III.csv")
SAVING_FILE_PREFIX = "../../../output/projection/fertility_III/MC_fertility_"  # Add the country name to this prefix
TFR_OF_STAGE_III_COUNTRIES = pd.read_csv("../../../data/UN/fertility/TFR_per_country_stage_III.csv")
POSTERIOR_DISTRIBUTION_OF_PARAMETERS_PREFIX = "../../../output/estimation/fertility_III/PMMH_fertility_III_"

N_samples = 100
BURN_IN_ratio = 0.1
N_years_ahead = 25


def run(country):
    SAVING_FILE = SAVING_FILE_PREFIX + country + ".pkl"
    if os.path.exists(SAVING_FILE):
        return
    DATA = TFR_OF_STAGE_III_COUNTRIES[TFR_OF_STAGE_III_COUNTRIES['Region'] == country]['TFR']
    DATA = np.array(DATA)
    POSTERIOR_DISTRIBUTION_OF_PARAMETERS = POSTERIOR_DISTRIBUTION_OF_PARAMETERS_PREFIX + country + ".pkl"
    with open(f'{POSTERIOR_DISTRIBUTION_OF_PARAMETERS}', 'rb') as handle:
        theta_samples = pickle.load(handle).theta
    theta_samples = theta_samples[int(BURN_IN_ratio * len(theta_samples)):]
    thetas = theta_samples[np.random.choice(len(theta_samples), size=N_samples, replace=False)]
    initial_fertility = DATA[-1]
    simulations = [fertility_III.fertility_III(initial_fertility, *tuple(thetas[i])) for i in range(N_samples)]
    for simulation in simulations:
        simulation.simulate(N_years_ahead)

    with open(f'{SAVING_FILE}', 'wb') as handle:
        pickle.dump(simulations, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for country in tqdm(LIST_OF_STAGE_III_COUNTRIES['Region']):
        run(country)
