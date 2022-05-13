import utils
import yaml
import pudb
import os
import priors
from astropy.units import Unit



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

    def __init__(self, 
        dynamic_pars = None, meta_pars = None, param_is_fixed = None,
        yaml_file=None):
        ''' Initializing Pars object

        Note: can either init by passing (sampled_pars, meta_pars, sampled_flag)
        and yaml_file.

        TODO: override yaml settings if both are provided

        sampled_flag: list of string
            A list of parameters name, the sequence will be used to set
            parameters order
        meta_pars: dict
            A dictionary of meta parameters and their values for
            a particular experiment and MCMC run
        param_is_fixed: list of boolean
            flags showing whether the parameter is fixed
        yaml_file: string
            path to a YAML file, which records the parameter settings
            
        '''
        #Interpret inputs
        flag_dict = (dynamic_pars is not None) and (meta_pars is not None)
        flag_yaml = (yaml_file is not None)
        assert (flag_yaml != flag_dict), \
            "Either initialize Pars object with dict or YAML file!"

        # load parameters from yaml file
        if flag_yaml:
            # interpret YAML dict
            if (os.path.isfile(yaml_file)):
                with open(yaml_file, 'r') as f:
                    meta_pars, dynamic_pars, self.dynamic_fid, param_is_fixed=\
                            self._interpret_yaml_file(f)
            else:
                meta_pars, dynamic_pars, self.dynamic_fid, param_is_fixed = \
                            self._interpret_yaml_file(yaml_file)
        # load parameters from arguments
        else:
            self.dynamic_fid = None
            if param_is_fixed == None:
                param_is_fixed = [False]*len(dynamic_pars)

        # input information are summarized in
        #   - meta_pars: dictionary
        #   - dynamic_pars: list of string, the full sample-able parameters 
        #                   listed in sequence
        #   - self.dynamic_fid: list of float, the fiducial values of those 
        #                       sample-able parameters
        #   - param_is_fixed: list of boolean, True for parameter being fixed
        args = {
            'dynamic_pars': (dynamic_pars, list),
            'meta_pars': (meta_pars, dict),
            'param_is_fixed': (param_is_fixed, list),
            }
        utils.check_types(args)

        for name in dynamic_pars:
            utils.check_type(name, 'dynamic_par_val', str)

        sampled_pars = []
        self.sampled_fid = []
        for i,is_fixed in enumerate(param_is_fixed):
            if(not is_fixed):
                sampled_pars.append(dynamic_pars[i])
                if self.dynamic_fid != None:
                    self.sampled_fid.append(self.dynamic_fid[i])
        sampled_pars_order = dict(zip(sampled_pars, range(len(sampled_pars))))
        self.param_is_fixed = param_is_fixed
        self.dynamic_pars_order = dict(zip(dynamic_pars, 
            range(len(dynamic_pars))))
        self.sampled = SampledPars(sampled_pars_order)
        self.meta = MetaPars(meta_pars)
        self.Npars = len(dynamic_pars)
        self.Nsampled = len(sampled_pars)
        self.Nfixed = self.Npars - self.Nsampled

        return

    def sampled_pars_2_dynamic_theta(self, pars):
        _pars = self.complete_sampled_pars(pars)
        return self.sampled.pars2theta(_pars)

        theta = len(PARS_ORDER) * ['']

        for name, indx in self.dynamic_pars_order.items():
            theta[indx] = _pars[name]

        return theta

    def sampled_theta_2_dynamic_pars(self, theta):
        _theta = self.complete_sampled_theta(theta)

        pars = {}
        for key, indx in self.dynamic_pars_order.items():
            pars[key] = _theta[indx]

        return pars

    def complete_sampled_theta(self, theta):
        """ Complete the theta param list with fixed params
        Input:
            theta: list of parameters being sampled by MCMC sampler
        Output:
            theta_full: list of model parameters 
        """
        assert len(theta)==(self.Nsampled), f'Number of parameters in theta '+\
            f'is not {self.Nsampled}!'

        theta_full = [0.0,] * self.Npars
        ct = 0
        for i in range(self.Npars):
            if self.param_is_fixed[i]:
                theta_full[i] = self.dynamic_fid[i]
            else:
                theta_full[i] = theta[ct]
                ct += 1

        return theta_full

    def complete_sampled_pars(self, pars):
        """ Complete the pars dict with fixed pars
        Input:
            pars: dict of parameters being sampled by MCMC sampler
        Output:
            pars_full: dict of model parameters
        """
        assert len(pars)==self.Nsampled, f'Number of parameters in pars '+\
            f'is not {self.Nsampled}!'

        pars_full = pars.copy()
        for key, val in self.dynamic_pars_order.items():
            if key not in pars_full.keys():
                pars_full[key] = self.dynamic_fid[val]
        return pars_full

    def theta2pars(self, theta):
        '''
        uses pars_order to convert list of sampled params to dict
        '''

        assert len(theta) == len(self.dynamic_pars_order)

        pars = {}
        for key, indx in self.dynamic_pars_order.items():
            pars[key] = theta[indx]

        return pars

    def pars2theta(self, pars):
        '''
        convert dict of parameters to theta list
        '''

        assert len(pars) == len(self.dynamic_pars_order)

        # initialize w/ junk that will fail if not set correctly
        theta = len(self.dynamic_pars_order) * ['']

        for name, indx in self.dynamic_pars_order.items():
            theta[indx] = pars[name]

        return theta

    @classmethod
    def _interpret_yaml_file(cls, yaml_file):
        print("Reading parameter settings from YAML file ({})".format(type(yaml_file)))
        #with open(yaml_file, 'r') as file:
        pars_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        print(pars_dict)

        # 1. get dynamic parameters and their sequence
        # Note that some of the fixed parameters are also included here
        Npars = len(pars_dict['dynamic_pars'].keys())
        print(f'{Npars} elements are used for model evaluation')
        pars_priors = {} # prior dictionary
        fid_dynamic = [0.] * Npars # fiducial values
        sampled_flag = [False] * Npars # flag showing param sampled or not

        dynamic_pars_dict = {}
        for key, val in pars_dict['dynamic_pars'].items():
            dynamic_pars_dict[key] = val['order']
        dynamic_pars = [k for k,v in \
            sorted(dynamic_pars_dict.items(), key=lambda item: item[1])]
        dynamic_pars_dict = dict(zip(dynamic_pars, range(len(dynamic_pars))))

        # loop through model parameters
        for key, val in pars_dict['dynamic_pars'].items():
            order = dynamic_pars_dict[key]
            fid = val['fid']
            fid_dynamic[order] = fid
            sampled_flag[order] = val.get('fixed', False)
            print(f'model param {order+1}: {key}')
            print(f'fiducial value = {fid}')

            if not sampled_flag[order]:
                # which priors?
                flag_flat = 'min' in val.keys() and 'max' in val.keys()
                flag_norm = 'mean' in val.keys() and 'std' in val.keys()
                assert flag_flat!=flag_norm, f'Either a flat prior or a '+\
                    f'Gaussian prior should be specified for {key}!'
                if flag_flat:
                    _min, _max = val['min'], val['max']
                    _inclusive = val.get('inclusive', False)
                    assert isinstance(_inclusive, bool), f'inclusive should'+\
                    f' be bool but is {_inclusive}!'
                    print(f'prior = priors.UniformPrior({_min}, {_max},'+\
                        f' inclusive={_inclusive})')
                    pars_priors[key] = priors.UniformPrior(_min, _max,
                        inclusive = _inclusive)
                else:
                    _mean, _std = val['mean'], val['std']
                    _clip_sigmas = val.get('clip_sigmas', None)
                    _zero_boundary = val.get('zero_boundary', None)
                    print(f'prior = priors.GaussPrior({_mean}, {_std}, '+\
                        f'clip_sigmas = {_clip_sigmas}, '+\
                        f'zero_boundary = {_zero_boundary})')
                    pars_priors[key] = priors.GaussPrior(_mean, _std,
                        clip_sigmas = _clip_sigmas,
                        zero_boundary = _zero_boundary)
            else:
                print(f'parameter kept fixed, no priors')
        del pars_dict['dynamic_pars']
        pars_dict['priors'] = pars_priors

        # 2. interpret velocity model units Unit('km / s')
        v_unit = Unit(pars_dict['velocity']['v_unit'])
        r_unit = Unit(pars_dict['velocity']['r_unit'])
        pars_dict['velocity']['v_unit'] = v_unit
        pars_dict['velocity']['r_unit'] = r_unit

        return pars_dict, dynamic_pars, fid_dynamic, sampled_flag


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

    def keys(self):
        return self.pars_order.keys()

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
        model_name: default/centered
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

    def pop(self, key):
        return self.pars.pop(key)

    def get(self, key, default=None):
        return self.pars.get(key, default)

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

