import numpy as np


class fertility_III:
    """
    AR model for fertility projection for the countries in Phase III
    """

    def __init__(self, initial_fertility, mu, rho, sigma_b, N_samples=1):
        self.N_samples = N_samples
        self.mu = mu
        self.rho = rho
        self.sigma_b = sigma_b
        self.path = np.ones((1, N_samples)) * initial_fertility

    def run(self):
        """
        Run the model for one year
        """
        self.path = np.append(self.path,
                              [self.mu + self.rho * (self.path[-1, :] - self.mu)
                               + self.sigma_b * np.random.multivariate_normal(np.zeros(self.N_samples), np.eye(self.N_samples))],
                              axis=0)

    def simulate(self, n_years):
        """
        Run the model for n_years
        """
        for i in range(n_years):
            self.run()
