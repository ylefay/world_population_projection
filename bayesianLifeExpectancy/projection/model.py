import numpy as np
import yaml

with open('../../parameters/parameters.yaml', 'r') as yaml_config:
    config = yaml.safe_load(yaml_config)

const_A1 = 4.4
const_A2 = 0.5

"""
Bayesian life expectancy model, ref: Bayesian Probabilistic Projections of Life Expectancy
for All Countries
"""


def decrement(life_expectancy, theta_c):
    triangle_1c, triangle_2c, triangle_3c, triangle_4c, k_c, z_c = theta_c
    sum_triangle = triangle_1c + triangle_2c + triangle_3c
    _ = k_c / (1 + np.exp(-const_A1 / triangle_2c * (life_expectancy - triangle_1c - const_A2 * triangle_2c)))
    _ += (z_c - k_c) / (1 + np.exp(-const_A1 / triangle_4c * (life_expectancy - sum_triangle - const_A2 * triangle_4c)))
    return _

def sigma(life_expectancy):
    raise NotImplementedError
