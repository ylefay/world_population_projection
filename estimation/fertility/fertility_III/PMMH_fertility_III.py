from particles import mcmc
from particles import distributions as dists
from estimation.fertility.fertility_III import ssm_fertility_III
import yaml

with open('../../../parameters/fertility_III.yaml', 'r') as yaml_config:
    fertility_III_config = yaml.safe_load(yaml_config)

default_prior = dists.StructDist(
    {'mu': dists.TruncNormal(mu=fertility_III_config['world']['mu'], sigma=0.1, a=0.0, b=3.0),
     'rho': dists.Uniform(a=0.0, b=1.0),
     'sigma_b': dists.Gamma()})


def run_PMMH(data, prior=default_prior, N_particles=200, niter=1000):
    pmmh = mcmc.PMMH(ssm_cls=ssm_fertility_III.fertility_III, data=data, prior=prior, Nx=N_particles, niter=niter)
    pmmh.run()
    return pmmh
