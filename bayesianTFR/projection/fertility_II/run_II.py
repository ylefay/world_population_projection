import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os
from bayesianTFR.projection.fertility_II import fertility_II
from bayesianTFR.projection.fertility_II.fertility_II import from_country_specific_parameters_to_delta
import yaml

LIST_OF_STAGE_II_COUNTRIES = pd.read_csv("../../data/UN/countries_in_stage_II.csv")
SAVING_FILE_PREFIX = "../../output/projection/fertility_II/MC_fertility_"  # Add the country name to this prefix
TFR_OF_STAGE_II_COUNTRIES = pd.read_csv("../../data/UN/TFR_per_country_stage_II.csv")
POSTERIOR_DISTRIBUTION_OF_PARAMETERS_PREFIX = "../../../output/estimation/fertility_II/PMMH_fertility_II_"
with open('../../parameters/fertility_II.yaml', 'r') as yaml_config:
    fertility_II_config = yaml.safe_load(yaml_config)

N_samples = 100
BURN_IN_ratio = 0.1
N_years_ahead = 25


def run(country):
    SAVING_FILE = SAVING_FILE_PREFIX + country + ".pkl"
    if os.path.exists(SAVING_FILE):
        return
    DATA = TFR_OF_STAGE_II_COUNTRIES[TFR_OF_STAGE_II_COUNTRIES['Region'] == country]
    starting_date = DATA['Year'].iloc[0]
    DATA = DATA['TFR']
    DATA = np.array(DATA)
    POSTERIOR_DISTRIBUTION_OF_PARAMETERS = POSTERIOR_DISTRIBUTION_OF_PARAMETERS_PREFIX + country + ".pkl"
    with open(f'{POSTERIOR_DISTRIBUTION_OF_PARAMETERS}', 'rb') as handle:
        theta_samples = pickle.load(handle).theta
    theta_samples = theta_samples[int(BURN_IN_ratio * len(theta_samples)):]
    thetas = theta_samples[np.random.choice(len(theta_samples), size=N_samples, replace=False)]
    initial_fertility = DATA[-1]
    simulations = []
    for i in range(N_samples):
        S, U_c, a, alpha_1, alpha_2, alpha_3, b, c1975, chi, d_c_star, delta_1sq, delta_2sq, delta_3sq, delta_4sq, gamma1_c, gamma2_c, gamma3_c, m_tau, phi, psi2, s_tausq, sigma0, triangle_4, triangle_4c_star = \
            thetas[i]
        delta_c = from_country_specific_parameters_to_delta((gamma1_c, gamma2_c, gamma3_c), U_c, d_c_star,
                                                            triangle_4c_star)
        simulations.append(fertility_II.fertility_II(initial_fertility, delta_c, c1975, sigma0, S, a, b,
                                                     fertility_II_config[country]['tau_c'], s_tausq, starting_date, phi))
    for simulation in simulations:
        simulation.simulate(N_years_ahead)

    with open(f'{SAVING_FILE}', 'wb') as handle:
        pickle.dump(simulations, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for country in tqdm(LIST_OF_STAGE_II_COUNTRIES['Region']):
        run(country)
