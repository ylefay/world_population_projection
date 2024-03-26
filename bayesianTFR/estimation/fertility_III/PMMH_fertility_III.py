from particles import mcmc
from particles import distributions as dists
from collections import OrderedDict
from bayesianTFR.estimation.fertility_III import ssm_fertility_III
import yaml
import numpy as np

with open('../../parameters/fertility_III.yaml', 'r') as yaml_config:
    fertility_III_config = yaml.safe_load(yaml_config)

default_prior = OrderedDict()
default_prior['mu_bar'] = dists.Uniform(a=0., b=fertility_III_config['world']['mu_bar_up'])
default_prior['sigma_mu'] = dists.Uniform(a=0., b=fertility_III_config['world']['sigma_mu_up'])
default_prior['rho_bar'] = dists.Uniform(a=0., b=1)
default_prior['sigma_rho'] = dists.Uniform(a=0., b=fertility_III_config['world']['sigma_rho_up'])
default_prior['sigma_eps'] = dists.Uniform(a=0., b=0.5)
default_prior['mu'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['mu_bar'], sigma=theta['sigma_mu'], a=0., b=2.1))
default_prior['rho'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['rho_bar'], sigma=theta['sigma_rho'], a=0., b=1.))
default_prior = dists.StructDist(default_prior)


def run_PMMH(data, prior=default_prior, N_particles=200, niter=1000, TRY=10):
    L = -np.inf
    for i in range(TRY):
        pmmh = mcmc.PMMH(ssm_cls=ssm_fertility_III.fertility_III, data=data, prior=prior, Nx=N_particles, niter=niter)
        pmmh.run()
        if pmmh.chain.lpost[-1] > L:
            L = pmmh.chain.lpost[-1]
            best_chain = pmmh.chain
    print("log-likelihood:", L)
    return best_chain
