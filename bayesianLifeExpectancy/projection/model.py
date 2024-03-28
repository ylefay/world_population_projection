import numpy as np
import yaml

with open('../parameters/parameters.yaml', 'r') as yaml_config:
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
    """
    Innovation noise, I was not able to find the exact function
    I will assume a linear decreasing function, using Fig. 3,
    I estimated:
    sigma(l) = 1.25 - (l-30)/50 * 1.15,
    such that sigma(30) = 1.25 and sigma(80) = 0.1
    """
    return np.maximum((1.25 - (life_expectancy - 30) / 50 * 1.15), 0.1)


class LifeExpectancy:
    """
    Life expectancy model: a double logistic regression for the decrement function
    """

    def __init__(self, initial_LE, theta_c, omega):
        self.path = np.array([initial_LE])
        self.theta_c = theta_c
        self.omega = omega

    def run(self):
        self.path = np.append(self.path, [
            self.path[-1] + np.random.normal(loc=decrement(self.path[-1], self.theta_c),
                                             scale=self.omega * sigma(self.path[-1]))
        ])

    def simulate(self, n_year):
        for i in range(n_year):
            self.run()
