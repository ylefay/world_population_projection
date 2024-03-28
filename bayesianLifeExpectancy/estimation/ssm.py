from particles import state_space_models as ssm
from particles import distributions as dists
from bayesianLifeExpectancy.projection.model import decrement, sigma
import scipy.stats as stats
import yaml

with open('../parameters/parameters.yaml', 'r') as yaml_config:
    config = yaml.safe_load(yaml_config)


class RightSkewedGambel(dists.ProbDist):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def rvs(self, size=None):
        return stats.gumbel_r.rvs(loc=self.loc, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.gumbel_r.logpdf(x=x, loc=self.loc, scale=self.scale)


class LifeExpectancy(ssm.StateSpaceModel):
    default_params = config['world']

    def PX0(self):
        return RightSkewedGambel(loc=self.default_params['gambel_loc'], scale=self.default_params['gambel_scale'])
        # return dists.Normal(loc=self.default_params['gambel_loc'], scale=self.default_params['gambel_scale'])

    def PX(self, t, xp):
        theta_c = (self.triangle_1c, self.triangle_2c, self.triangle_3c, self.triangle_4c, self.k_c, self.z_c)
        loc = xp + decrement(xp, theta_c)
        scale = 1 * sigma(xp)
        return dists.Normal(loc=loc, scale=scale)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=0.02)
