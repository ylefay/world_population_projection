import numpy as np


def five_year_decrement(fertility_rate, delta_c):
    nabla_1c, nabla_2c, nabla_3c, nabla_4c, d_c = delta_c
    cst = -2 * np.log(9)
    _ = d_c * (1 / (1 + np.exp(cst * (fertility_rate - nabla_4c - 0.5 * nabla_3c) / nabla_3c)) - 1 / (
            1 + np.exp(cst * (fertility_rate - nabla_2c - nabla_3c - nabla_4c + 0.5 * nabla_1c) / nabla_1c)))
    return _


class fertility_rate:
    """
    Fertility dynamic for phase II countries
    """

    def __init__(self, initial_fertility, delta_c, sigma):
        self.path = np.array([initial_fertility])
        self.delta_c = delta_c
        self.sigma = sigma

    def run(self):
        """
        Run the model for one year
        """
        noise = self.sigma(len(self.path, self.path[-1])) * np.random.normal(0., scale=1.0)
        self.path = np.append(self.path, self.path[-1] - five_year_decrement(self.path[-1], self.delta_c) + noise)

    def simulate(self, n_years):
        """
        Run the model for n_years
        """
        for i in range(n_years):
            self.run()
