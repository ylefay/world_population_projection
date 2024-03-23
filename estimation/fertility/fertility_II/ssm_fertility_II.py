from particles import state_space_models as ssm
from particles import distributions as dists
from projection.fertility.fertility_II.fertility_II import five_year_decrement
import yaml

with open('../../../projection/parameters/fertility_II.yaml', 'r') as yaml_config:
    fertility_II_config = yaml.safe_load(yaml_config)


class fertility_III(ssm.StateSpaceModel):
    # Assuming constant diffusion
    default_params = fertility_II_config['world']

    def PX0(self):
        raise NotImplementedError

    def PX(self, t, xp):
        # need to compute triangle (nabla) values...
        return dists.Normal(
            loc=xp - five_year_decrement(xp, (self.triangle_1c, self.triangle_2c, self.triangle_3c, self.triangle_4c, self.d_c)),
            scale=self.sigma)  # (t, xp)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=0.01)
