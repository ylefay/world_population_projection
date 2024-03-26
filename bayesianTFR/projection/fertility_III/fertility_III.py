import numpy as np


class fertility_III:
    """
    AR model for fertility projection for the countries in Phase III
    """

    def __init__(self, initial_fertility, mu, mu_bar, rho, rho_bar, sigma_mu, sigma_rho, sigma_eps):
        self.mu = mu
        self.rho = rho
        self.sigma_mu = sigma_mu
        self.sigma_rho = sigma_rho
        self.mu_bar = mu_bar
        self.rho_bar = rho_bar
        self.sigma_eps = sigma_eps
        self.path = np.array([initial_fertility])

    def run(self):
        """
        Run the model for one year
        """
        new = self.mu + self.rho * (self.path[-1] - self.mu)
        + self.sigma_eps * np.random.normal(loc=0., scale=1.0)
        self.path = np.append(self.path,
                              [new if new > 0 else 0.],
                              axis=0)

    def simulate(self, n_years):
        """
        Run the model for n_years
        """
        for i in range(n_years):
            self.run()
