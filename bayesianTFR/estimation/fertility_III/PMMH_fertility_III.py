from particles import mcmc
from particles import distributions as dists
from bayesianTFR.estimation.fertility_III import ssm_fertility_III
import yaml

with open('../../parameters/fertility_III.yaml', 'r') as yaml_config:
    fertility_III_config = yaml.safe_load(yaml_config)

default_prior = dists.StructDist(
    {
        'mu_bar': dists.Uniform(a=0, b=fertility_III_config['world']['mu_bar_up']),
        'sigma_mu': dists.Uniform(a=0, b=fertility_III_config['world']['sigma_mu_up']),
        'rho_bar': dists.Uniform(a=0, b=1),
        'sigma_rho': dists.Uniform(a=0, b=fertility_III_config['world']['sigma_rho_up']),
        'mu': dists.Cond(lambda theta: dists.Normal(loc=theta['mu_bar'], scale=theta['sigma_mu'])),
        'rho': dists.Cond(lambda theta: dists.Normal(loc=theta['rho_bar'], scale=theta['sigma_rho'])),
        'sigma_bsq': dists.Gamma()
    }
)


def run_PMMH(data, prior=default_prior, N_particles=200, niter=1000):
    pmmh = mcmc.PMMH(ssm_cls=ssm_fertility_III.fertility_III, data=data, prior=prior, Nx=N_particles, niter=niter)
    pmmh.run()
    return pmmh
