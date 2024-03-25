import numpy as np


class fertility_III:
    """
    AR model for fertility projection for the countries in Phase III
    """
    ('mu', 'mu_bar', 'rho', 'rho_bar', 'sigma_mu', 'sigma_rho')

    def __init__(self, initial_fertility, mu, mu_bar, rho, rho_bar, sigma_mu, sigma_rho, sigma_bsq):
        self.mu = mu
        self.rho = rho
        self.sigma_mu = sigma_mu
        self.sigma_rho = sigma_rho
        self.mu_bar = mu_bar
        self.rho_bar = rho_bar
        self.sigma_b = sigma_bsq ** 0.5
        self.path = np.array([initial_fertility])

    def run(self):
        """
        Run the model for one year
        """
        self.path = np.append(self.path,
                              [self.mu + self.rho * (self.path[-1] - self.mu)
                               + self.sigma_b * np.random.normal(loc=0., scale=1.0)],
                              axis=0)

    def simulate(self, n_years):
        """
        Run the model for n_years
        """
        for i in range(n_years):
            self.run()
