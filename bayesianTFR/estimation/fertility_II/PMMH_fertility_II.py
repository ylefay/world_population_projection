from particles import mcmc
from particles import distributions as dists
from bayesianTFR.estimation.fertility_II import ssm_fertility_II
import numpy as np
import yaml
from scipy.signal import find_peaks
from numbers import Number

with open('../../parameters/fertility_II.yaml', 'r') as yaml_config:
    fertility_II_config = yaml.safe_load(yaml_config)

default_prior = {  # First defining the prior on the world-level parameters
    'triangle_4': dists.Normal(loc=fertility_II_config['world']['triangle40'],
                               scale=fertility_II_config['world']['delta40']),
    'delta_4sq': dists.InvGamma(a=1.0, b=1 / fertility_II_config['world']['delta40'] ** 2),
    'delta_3sq': dists.InvGamma(a=1.0, b=1 / fertility_II_config['world']['delta0'] ** 2),
    'delta_2sq': dists.InvGamma(a=1.0, b=1 / fertility_II_config['world']['delta0'] ** 2),
    'delta_1sq': dists.InvGamma(a=1.0, b=1 / fertility_II_config['world']['delta0'] ** 2),
    'alpha_1': dists.Normal(loc=fertility_II_config['world']['alpha01'],
                            scale=fertility_II_config['world']['delta0']),

    'alpha_2': dists.Normal(loc=fertility_II_config['world']['alpha02'],
                            scale=fertility_II_config['world']['delta0']),

    'alpha_3': dists.Normal(loc=fertility_II_config['world']['alpha03'],
                            scale=fertility_II_config['world']['delta0']),
    'psi2': dists.InvGamma(a=1.0, b=1 / fertility_II_config['world']['psi0'] ** 2),
    'chi': dists.Normal(loc=fertility_II_config['world']['chi0'], scale=fertility_II_config['world']['psi0']),
    # now defining the prior on the country-specific parameters
    'gamma1_c': dists.Cond(lambda theta: dists.Normal(loc=theta['alpha_1'], scale=theta['delta_1sq'] ** 0.5)),
    'gamma2_c': dists.Cond(lambda theta: dists.Normal(loc=theta['alpha_2'], scale=theta['delta_2sq'] ** 0.5)),
    'gamma3_c': dists.Cond(lambda theta: dists.Normal(loc=theta['alpha_3'], scale=theta['delta_3sq'] ** 0.5)),
    'd_c_star': dists.Cond(lambda theta: dists.Normal(loc=theta['chi'], scale=theta['psi2'] ** 0.5)),
    'triangle_4c_star': dists.Cond(
        lambda theta: dists.Normal(loc=theta['triangle_4'], scale=theta['delta_4sq'] ** 0.5)),
    'a': dists.Uniform(fertility_II_config['world']['a_low'], fertility_II_config['world']['a_up']),
    'b': dists.Uniform(fertility_II_config['world']['b_low'], fertility_II_config['world']['b_up']),
    'sigma0': dists.Uniform(fertility_II_config['world']['sigma0_low'], fertility_II_config['world']['sigma0_up']),
    'c1975': dists.Uniform(fertility_II_config['world']['const_low'], fertility_II_config['world']['const_up']),
    'S': dists.Uniform(fertility_II_config['world']['S_low'], fertility_II_config['world']['S_up']),
    'm_tau': dists.Normal(loc=fertility_II_config['world']['mean_eps_tau0'],
                          scale=fertility_II_config['world']['std_eps_tau0']),
    's_tausq': dists.InvGamma(a=1.0, b=1 / fertility_II_config['world']['std_eps_tau0'] ** 2),
    # 'phi': dists.Uniform(a=0., b=1.0)
}


def get_tau_c(data):
    # the start period of phase II
    peaks, _ = find_peaks(data, rel_height=None)
    if len(peaks) == 0:
        peaks = data
    maximum = max(data)

    def get_local_maximum(t):
        s = np.max([peak if peak <= t else -np.inf for peak in peaks])
        if s == -np.inf:
            return t
        return s

    _ = np.zeros(len(data))
    for t in range(len(data)):
        local_maximum = data[int(get_local_maximum(t))]
        _[t] = maximum - local_maximum
        if local_maximum <= 5.5 or _[t] > 0.5:
            _[t] = -np.inf
    return np.argmax(_)


def get_prior(data, **kwargs):
    prior = default_prior.copy()
    tau_c = kwargs.pop('tau_c', get_tau_c(data))
    if isinstance(tau_c, Number):
        prior['U_c'] = dists.Dirac(loc=data[tau_c])
    else:
        prior['U_c'] = dists.Uniform(np.minimum(5.5, np.max(data)), 8.8)
    return dists.StructDist(prior)


def run_PMMH(data, N_particles=200, niter=1000, **kwargs):
    default_params = fertility_II_config['world']
    default_params['t0'] = kwargs.pop('t0', 1950)
    default_params['tau_c'] = kwargs.get('tau_c', get_tau_c(data))

    prior = get_prior(data, **kwargs)

    pmmh = mcmc.PMMH(ssm_cls=ssm_fertility_II.get_fertility_II_class(default_params), data=data,
                     prior=prior, Nx=N_particles, niter=niter)
    pmmh.run()
    return pmmh
