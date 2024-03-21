import numpy as np


class fertility_rate:
    """
    AR model for fertility projection for the countries in Phase III
    """

    def __init__(self, initial_fertility, rho, sigma):
        self.rho = rho
        self.sigma = sigma
        self.path = np.array([initial_fertility])

    def run(self):
        """
        Run the model for one year
        """
        self.path = np.append(self.path,
                  [self.mu + self.rho * (self.path[-1] - self.mu) + self.sigma * np.random.normal(0., scale=1.0)])

    def simulate(self, n_years):
        """
        Run the model for n_years
        """
        for i in range(n_years):
            self.run()
