import utils

import pudb

'''
This file defines the structure and conversions between a params dict
and a parameter list (theta) that is used both in MCMC sampling and
in numba functions that won't accept a dict

pars: dict
    A dictionary that holds *at least* the defined params for the model

theta: list or np.array
    A list or numpy array that has the model pararmeters in a fixed order

TODO: To make this future-proof, what we should do is make a function that
      *returns* PARS_ORDER given a velocity model name. This way it is accessible
      throughout, but allows things to be a bit more flexible
'''

class Pars(object):
    '''
    Holds all of the parameters for a needed MCMC run, both
    sampled and meta parameters
    '''

    def __init__(self, sampled_pars, meta_pars):
        '''
        sampled_pars: list of str's
            A list of parameter names to be sampled in the MCMC.
            Their order will be used to define pars2theta
        meta_pars: dict
            A dictionary of meta parameters and their values for
            a particular experiment and MCMC run
        '''

        args = {
            'sampled_pars': (sampled_pars, list),
            'meta_pars': (meta_pars, dict)
            }
        utils.check_types(args)

        for name in sampled_pars:
            utils.check_type(name, 'sampled_par_val', str)

        pars_order = dict(zip(sampled_pars, range(len(sampled_pars))))
        self.sampled = SampledPars(pars_order)
        self.meta = MetaPars(meta_pars)

        return

    def pars2theta(self, pars):
        return self.sampled.pars2theta(pars)

    def theta2pars(self, theta):
        return self.sampled.theta2pars(theta)

class SampledPars(object):
    '''
    Sets the structure for arbitrary sampled parameters, which
    are stored internally as a fixed list in the samplers
    '''

    def __init__(self, pars_order):
        '''
        pars_order: dict
            A dictionary that defines the par_name: sampler_index
            relationship in the used sampler

            For example:
            pars_order = {
                'g1': 0,
                'g2': 1,
                ...
            }
        '''

        utils.check_type(pars_order, 'pars_order', dict)

        for key, val in pars_order.items():
            if not isinstance(val, int):
                raise TypeError('pars_order must have int values!')

        self.pars_order = pars_order

        return

    def theta2pars(self, theta):
        '''
        uses pars_order to convert list of sampled params to dict
        '''

        assert len(theta) == len(self.pars_order)

        pars = {}
        for key, indx in self.pars_order.items():
            pars[key] = theta[indx]

        return pars

    def pars2theta(self, pars):
        '''
        convert dict of parameters to theta list
        '''

        assert len(pars) == len(self.pars_order)

        # initialize w/ junk that will fail if not set correctly
        theta = len(PARS_ORDER) * ['']

        for name, indx in PARS_ORDER.items():
            theta[indx] = pars[name]

        return theta

    def __repr__(self):
        return str(self.pars_order)

    def __len__(self):
        return len(self.pars_order.keys())

class MetaPars(object):
    '''
    Class that defines structure for the general parameters
    used in MCMC sampling for a given experiment, modeling
    choices, etc.

    Some essential fields are:
    priors:
        g1:
        g2:
        theta_int:
        sini:
        v0:
        vcirc:
        rscale:
    model_dimension:
        Nx:
        Ny:
        scale:
        lambda_range:
        lambda_res:
    intensity:
        type:
        hlr:
    velocity:
        v_unit:
        r_unit:
    sed:
        template:
        wave_type:
        flux_type:
        z:
        wave_range:
        obs_cont_norm:
        lines:
        line_sigma_int:
    observations:
        number_of_observations:
        obs_1:
        obs_2:
        ...

    '''

    _req_fields = ['priors', 'model_dimension', 'intensity', 'velocity',
                    'sed', 'observations']

    def __init__(self, pars):
        '''
        pars: dict
            Dictionary of meta pars
        '''

        utils.check_type(pars, 'pars', dict)

        self._check_pars(pars)
        self.pars = pars

        return

    @classmethod
    def _check_pars(cls, pars):
        '''
        Make sure that the general parameter list
        contains a few required entries
        '''

        for key in cls._req_fields:
            if key not in pars:
                raise KeyError(f'{key} is a required field ' +\
                                'in the parameter list!')

        return

    def __getitem__(self, key):
        return self.pars[key]

    def __setitem__(self, key, val):
        self.pars[key] = val
        return

    def __delitem__(self, key):
        del self.pars[key]
        return

    def __iter__(self):
        return iter(self.pars)

    def __repr__(self):
        return str(self.pars)

    def keys(self):
        return self.pars.keys()

    def items(self):
        return self.pars.items()

    def values(self):
        return self.pars.values()

# order of sampled mcmc parameters
# NOTE: This won't be accessible if we use numba
PARS_ORDER = {
    'g1': 0,
    'g2': 1,
    'theta_int': 2,
    'sini': 3,
    'v0': 4,
    'vcirc': 5,
    'rscale': 6
    }

def theta2pars(theta):
    '''
    uses PARS_ORDER to convert list of sampled params to dict
    '''

    assert len(theta) == len(PARS_ORDER)

    pars = {}

    for key, indx in PARS_ORDER.items():
        pars[key] = theta[indx]

    return pars

def pars2theta(pars):
    '''
    convert dict of paramaeters to theta list
    '''

    # initialize w/ junk
    theta = len(PARS_ORDER) * ['']

    for name, indx in PARS_ORDER.items():
        theta[indx] = pars[name]

    return theta

