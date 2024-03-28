import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
import yaml
from bayesianLifeExpectancy.projection.model import LifeExpectancy

LIST_OF_COUNTRIES = pd.read_csv("../data/UN/countries.csv")
SAVING_FILE_PREFIX = "../output/projection/LE_"
LE = pd.read_csv("../data/UN/LE_per_country.csv")
POSTERIOR_DISTRIBUTION_OF_PARAMETERS_PREFIX = "../output/estimation/PMMH_"

with open('../parameters/parameters.yaml', 'r') as yaml_config:
    config = yaml.safe_load(yaml_config)

N_samples = 100
BURN_IN_ratio = 0.1
N_years_ahead = 25


def run(country):
    SAVING_FILE = SAVING_FILE_PREFIX + country + ".pkl"
    if os.path.exists(SAVING_FILE):
        return
    DATA = np.array(LE[LE['Region'] == country]['LE'])
    POSTERIOR_DISTRIBUTION_OF_PARAMETERS = POSTERIOR_DISTRIBUTION_OF_PARAMETERS_PREFIX + country + ".pkl"
    with open(POSTERIOR_DISTRIBUTION_OF_PARAMETERS, 'rb') as handle:
        theta_samples = pickle.load(handle)
    theta_samples = theta_samples[int(BURN_IN_ratio * N_samples):].theta
    thetas = theta_samples[np.random.choice(range(len(theta_samples)), size=N_samples, replace=False)]
    initial_LE = DATA[-1]
    simulations = []
    for i in range(N_samples):
        _, _, _, _, _, _, _, _, _, _, _, _, omega, triangle_1c, triangle_2c, triangle_3c, triangle_4c, k_c, z_c = \
        thetas[i]
        thetas_c = (triangle_1c, triangle_2c, triangle_3c, triangle_4c, k_c, z_c)
        simulations.append(LifeExpectancy(initial_LE, thetas_c, omega))

    for simulation in simulations:
        simulation.simulate(N_years_ahead)

    with open(SAVING_FILE, 'wb') as handle:
        pickle.dump(simulations, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for country in tqdm(LIST_OF_COUNTRIES['Region']):
        if country == "WORLD":
            run(country)
