import numpy as np


def five_year_decrement(fertility_rate, delta_c):
    triangle_1c, triangle_2c, triangle_3c, triangle_4c, d_c = delta_c
    cst = -2 * np.log(9)
    _ = d_c * (1 / (1 + np.exp(cst * (fertility_rate - triangle_4c - 0.5 * triangle_3c) / triangle_3c)) - 1 / (
            1 + np.exp(
        cst * (fertility_rate - triangle_2c - triangle_3c - triangle_4c + 0.5 * triangle_1c) / triangle_1c)))
    return _


def from_country_specific_parameters_to_delta(gammas_c, U_c, d_c_star, triangle_4c_star):
    triangle_4c = (-1. - 2.5 * np.exp(1) ** triangle_4c_star) / (-1. - np.exp(1) ** triangle_4c_star)
    d_c = (-0.25 - 2.5 * np.exp(1) ** d_c_star) / (-1. - np.exp(1) ** d_c_star)
    p_cs = np.exp(gammas_c) / np.sum(np.exp(gammas_c))
    triangle_cs = p_cs * (U_c - triangle_4c)  # for i = 1, 2, 3
    delta = (*triangle_cs, triangle_4c, d_c)
    return delta


class fertility_II:
    """
    Fertility dynamic for phase II countries
    """

    def __init__(self, N_samples, initial_fertility, delta_c, sigma):
        self.N_samples = N_samples
        self.path = np.array([initial_fertility]) * N_samples
        self.delta_c = delta_c
        self.sigma = sigma

    def run(self):
        """
        Run the model for one year
        """
        sigma = np.vectorize(lambda x: self.sigma(self.path.shape[0], x))(self.path[-1, :])
        noise = sigma * np.random.multivariate_normal(0., scale=np.eye(self.N_samples))
        five_year_decrements = np.vectorize(lambda x: five_year_decrement(x, self.delta_c))(self.path[-1, :])
        self.path = np.append(self.path, self.path[-1, :] - five_year_decrements + noise, axis=0)

    def simulate(self, n_years):
        """
        Run the model for n_years
        """
        for i in range(n_years):
            self.run()
