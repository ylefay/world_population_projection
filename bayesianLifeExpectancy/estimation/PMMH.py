from particles import mcmc
from particles import distributions as dists
from bayesianLifeExpectancy.estimation.ssm import LifeExpectancy
import numpy as np
import yaml

from collections import OrderedDict

with open('../parameters/parameters.yaml', 'r') as yaml_config:
    config = yaml.safe_load(yaml_config)

default_prior = OrderedDict()
# world priors
default_prior['triangle_1'] = dists.TruncNormal(mu=config['world']['a1'], sigma=config['world']['delta_1_sq'] ** 0.5,
                                                a=0., b=100.)
default_prior['triangle_2'] = dists.TruncNormal(mu=config['world']['a2'], sigma=config['world']['delta_2_sq'] ** 0.5,
                                                a=0., b=100.)
default_prior['triangle_3'] = dists.TruncNormal(mu=config['world']['a3'], sigma=config['world']['delta_3_sq'] ** 0.5,
                                                a=0., b=100.)
default_prior['triangle_4'] = dists.TruncNormal(mu=config['world']['a4'], sigma=config['world']['delta_4_sq'] ** 0.5,
                                                a=0., b=100.)
default_prior['z'] = dists.TruncNormal(mu=config['world']['a6'], sigma=config['world']['delta_5_sq'] ** 0.5, a=0.,
                                       b=1.15)
default_prior['k'] = dists.TruncNormal(mu=config['world']['a5'], sigma=config['world']['delta_6_sq'] ** 0.5, a=0.,
                                       b=10.)
default_prior['sigma_k_sq'] = dists.InvGamma(a=2, b=config['world']['rate_5'] ** 2)
default_prior['sigma_z_sq'] = dists.InvGamma(a=2, b=config['world']['rate_6'] ** 2)
default_prior['sigma_triangle1_sq'] = dists.InvGamma(a=2, b=config['world']['rate_1'] ** 2)
default_prior['sigma_triangle2_sq'] = dists.InvGamma(a=2, b=config['world']['rate_2'] ** 2)
default_prior['sigma_triangle3_sq'] = dists.InvGamma(a=2, b=config['world']['rate_3'] ** 2)
default_prior['sigma_triangle4_sq'] = dists.InvGamma(a=2, b=config['world']['rate_4'] ** 2)
default_prior['omega'] = dists.Uniform(a=0., b=1.)
# country dependent conditional priors
default_prior['triangle_1c'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['triangle_1'], sigma=theta['sigma_triangle1_sq'] ** 0.5, a=0., b=100.))
default_prior['triangle_2c'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['triangle_2'], sigma=theta['sigma_triangle2_sq'] ** 0.5, a=0., b=100.))
default_prior['triangle_3c'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['triangle_3'], sigma=theta['sigma_triangle3_sq'] ** 0.5, a=0., b=100.))
default_prior['triangle_4c'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['triangle_4'], sigma=theta['sigma_triangle4_sq'] ** 0.5, a=0., b=100.))
default_prior['k_c'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['k'], sigma=theta['sigma_k_sq'] ** 0.5, a=0., b=10.))
default_prior['z_c'] = dists.Cond(
    lambda theta: dists.TruncNormal(mu=theta['z'], sigma=theta['sigma_z_sq'] ** 0.5, a=0., b=1.15))
default_prior = dists.StructDist(default_prior)


def run_PMMH(data, N_particles=200, niter=1000, TRY=1):
    L = -np.inf
    for i in range(TRY):
        pmmh = mcmc.PMMH(ssm_cls=LifeExpectancy, data=data,
                         prior=default_prior, Nx=N_particles, niter=niter)
        pmmh.run()
        if pmmh.chain.lpost[-1] > L:
            L = pmmh.chain.lpost[-1]
            best_chain = pmmh.chain
        if L > 0:
            break
    print("log-likelihood:", L)
    return best_chain
