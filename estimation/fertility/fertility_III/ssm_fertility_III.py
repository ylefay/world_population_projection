from particles import state_space_models as ssm
from particles import distributions as dists
import yaml

with open('../../../parameters/fertility_III.yaml', 'r') as yaml_config:
    fertility_III_config = yaml.safe_load(yaml_config)


class fertility_III(ssm.StateSpaceModel):
    default_params = fertility_III_config['world']

    def PX0(self):
        return dists.LinearD(dists.Gamma(a=1.0, b=1 / self.scale), a=-1.0, b=2.0)

    def PX(self, t, xp):
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma_b)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=0.02)
