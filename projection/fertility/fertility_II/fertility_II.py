import numpy as np
import yaml

with open('../../../parameters/fertility_III.yaml', 'r') as yaml_config:
    fertility_III_config = yaml.safe_load(yaml_config)


def five_year_decrement(fertility_rate, delta_c):
    cond1 = fertility_rate >= 1.0
    triangle_1c, triangle_2c, triangle_3c, triangle_4c, d_c = delta_c
    cst = -2 * np.log(9)
    _ = d_c * (1 / (1 + np.exp(cst * (fertility_rate - triangle_4c - 0.5 * triangle_3c) / triangle_3c)) - 1 / (
            1 + np.exp(
        cst * (
                fertility_rate - triangle_1c - triangle_2c - triangle_3c - triangle_4c + 0.5 * triangle_1c) / triangle_1c)))
    return cond1 * _


def from_country_specific_parameters_to_delta(gammas_c, U_c, d_c_star, triangle_4c_star):
    triangle_4c = (-1. - 2.5 * np.exp(1) ** triangle_4c_star) / (-1. - np.exp(1) ** triangle_4c_star)
    d_c = (-0.25 - 2.5 * np.exp(1) ** d_c_star) / (-1. - np.exp(1) ** d_c_star)
    p_cs = np.exp(gammas_c) / np.sum(np.exp(gammas_c))
    triangle_cs = p_cs * (U_c - triangle_4c)  # for i = 1, 2, 3
    delta = (*triangle_cs, triangle_4c, d_c)
    return delta


def fun_sigma(c1975, sigma0, S, a, b, tau_c, s_tausq, t, fertility_rate):
    if t == tau_c:
        return s_tausq ** 0.5
    if isinstance(fertility_rate, float):
        fertility_rate = np.array([fertility_rate])
    S = np.ones(fertility_rate.shape) * S
    cond1 = S <= fertility_rate
    cond2 = fertility_rate <= S
    c_1975t = c1975 if t <= 1975 else 1.0
    _ = -a * cond1 + b * cond2
    return c_1975t * (sigma0 + (fertility_rate - S) * _)


class fertility_II:
    """
    Fertility dynamic for phase II countries
    """

    def __init__(self, initial_fertility, delta_c, c1975, sigma0, S, a, b, tau_c, s_tausq, t0):
        self.path = np.array([initial_fertility])
        self.delta_c = delta_c
        self.t0 = t0
        self.c1975 = c1975
        self.sigma0 = sigma0
        self.S = S
        self.a = a
        self.b = b
        self.s_tausq = s_tausq
        self.tau_c = tau_c
        self.phaseIII = False

    def run(self):
        """
        Run the model for one year
        """
        if not self.phaseIII:
            sigma = fun_sigma(self.c1975, self.sigma0, self.S, self.a, self.b, self.tau_c, self.s_tausq, self.t0 + self.path.shape[0],
                              self.path[-1])
            noise = sigma * np.random.normal(loc=0., scale=1.0)
            five_year_decrements = five_year_decrement(self.path[-1], self.delta_c)
            self.path = np.append(self.path, self.path[-1] - five_year_decrements + noise, axis=0)
            self.phaseIII = self.path[-1] <= 2.0 and self.path[-2] <= 2.0
        else:
            self.path = np.append(self.path,
                                  [fertility_III_config['world']['mu'] + fertility_III_config['world']['rho'] * (
                                          self.path[-1] - fertility_III_config['world']['mu']) +
                                   fertility_III_config['world'][
                                       'sigma_b'] * np.random.normal(loc=0., scale=1.0)], axis=0)

    def simulate(self, n_years):
        """
        Run the model for n_years
        """
        for i in range(n_years):
            self.run()
