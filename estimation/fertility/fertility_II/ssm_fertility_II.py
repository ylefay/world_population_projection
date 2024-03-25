from particles import state_space_models as ssm
from particles import distributions as dists
from projection.fertility.fertility_II.fertility_II import five_year_decrement, \
    from_country_specific_parameters_to_delta, fun_sigma
import yaml
import numpy as np

with open('../../../parameters/fertility_II.yaml', 'r') as yaml_config:
    fertility_II_config = yaml.safe_load(yaml_config)


def get_fertility_II_class(_default_params):
    class fertility_II(ssm.StateSpaceModel):
        # Assuming constant diffusion
        # default_params = fertility_II_config['world']
        # default_params['t0'] = 1972 #pour l'instant..
        default_params = _default_params

        def PX0(self):
            return dists.Normal(loc=self.default_params['prior_mu'], scale=self.default_params['prior_sigma'])

        def PX(self, t, xp):
            # need to compute triangle values...
            delta_c = from_country_specific_parameters_to_delta((self.gamma1_c, self.gamma2_c, self.gamma3_c), self.U_c,
                                                                self.d_c_star,
                                                                self.triangle_4c_star)  # country specific values
            return dists.Normal(
                loc=xp - five_year_decrement(xp, delta_c),
                scale=np.abs(fun_sigma(self.c1975, self.sigma0, self.S, self.a, self.b, self.tau_c, self.s_tausq, t + self.t0, xp)))  # (t, xp)

        def PY(self, t, xp, x):
            return dists.Normal(loc=x, scale=0.02)

    return fertility_II
