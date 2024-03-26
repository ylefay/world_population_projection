from particles import state_space_models as ssm
from particles import distributions as dists
from bayesianLifeExpectancy.projection.model import decrement, sigma
import yaml

with open('../../parameters/parameters.yaml', 'r') as yaml_config:
    config = yaml.safe_load(yaml_config)


class LifeExpectancy(ssm.StateSpaceModel):
    default_params = config['world']

    def PX0(self):
        raise NotImplementedError

    def PX(self, t, xp):
        theta_c = (self.triangle_1c, self.triangle_2c, self.triangle_3c, self.triangle_4c, self.k_c, self.z_c)
        loc = xp - decrement(xp, theta_c)
        scale = sigma(xp)
        return dists.Normal(loc=loc, scale=scale)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=0.02)
