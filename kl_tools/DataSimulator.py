import os
import copy
import pickle
from abc import abstractmethod
from time import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import galsim as gs

import utils
import intensity
from velocity import VelocityMap
from spectrum import Spectrum
from parameters import Pars
from cube import DataVector, GrismGenerator, ImageGenerator
import pudb


GENERATOR_TYPES = {
    'grism': GrismGenerator,
    'photometry': ImageGenerator,
    #'ifu': ,
}
class DataSimulator():
    ''' Object simulating observation data based on input parameters.
    
        This class works as a "data simulator": it accepts `Pars` object as 
        input arguments during initialization, and generates model vector with
        input sampled parameters list.

        The galaxy model and shear information is recorded in a 3d theory model
        numpy.ndarray, which contains astronomical images of a source at
        various wavelength slices. The DataSimulator object can generate theory
        model cube given input sampled parameters by building velocity map, sed
        and intensity map. The cube is then passed to GrismGenerator object to
        get observed images.
    '''

    def __init__(self, pars):
        ''' Initialize DataSimulator with Pars object
        
        The initialization will save the Pars object, set data dimension

        Note:
        1. Bandpass objects are initialized in Generator objects, not in 
        DataSimulator, because they are observation-related properties.
        2. For Basis Function intensity profile model, an input data is needed
        to build a profile model. But in InclinedExponential model, we only 
        need model parameters. We need a flexible API to accommodate that.

        Input:
        ======
        pars: `Pars` object
            parameters for initialization  
        '''
        print("Initializing DataSimulator...")
        utils.check_type(pars, 'pars', Pars)
        self.pars = copy.deepcopy(pars)
        
        # set theory model cube dimension
        # note that each call during MCMC will create a new cube, so we don't 
        # need to set it as a self.object
        self.Nx = self.pars.meta['model_dimension']['Nx']
        self.Ny = self.pars.meta['model_dimension']['Ny']
        self.scale = self.pars.meta['model_dimension']['scale']
        self.X, self.Y = utils.build_map_grid(self.Nx, self.Ny, 
            scale=self.scale)
        lam = self.pars.meta['model_dimension']['lambda_range']
        dlam = self.pars.meta['model_dimension']['lambda_res']
        blue_edge = np.arange(lam[0], lam[1], dlam)
        self.lambdas = np.array([(l, l+dlam) for l in blue_edge])
        self.lambda_unit = self.pars.meta['model_dimension']['lambda_unit']
        self.lambda_cen = np.mean(self.lambdas, axis=1)
        self.Nspec = self.lambdas.shape[0]
        self.shape = (self.Nspec, self.Ny, self.Nx)
        self._data = None
        
        # initialize data generators for each observation
        _op = self.pars.meta.pop('observations')
        self.Nobs = _op['number_of_observations']
        self.obs_pars = []
        self.data_generators = []
        for i in range(self.Nobs):
            _pars = copy.deepcopy(self.pars)
            _pars.meta['observations'] = _op['obs_%d'%(i+1)]
            self.obs_pars.append( _pars )
            _gen_type = _op['obs_%d'%(i+1)]['type'].lower()
            self.data_generators.append( 
                GENERATOR_TYPES[_gen_type](self.obs_pars[-1]) 
                )

        # Initialize VelocityMap, IntensityMap and Spectrum objects
        # During MCMC, we will cal the `render` (or equivalent routines) to 
        # get updated data, rather than construct a new object.
        if self.pars.meta['intensity'].get('type','default') == 'inclined_exp':
            self.datacube = None
        else:
            print("Using BasisIntensityMap, need input datacube!")
            # set trivial modelcube and bandpasses
            bandpasses = [ gs.Bandpass(1.0, self.lambda_unit, 
                blue_limit=l[0], red_limit=l[1], 
                zeropoint=zp) for l in self.lambdas]
            # set theory model cube (trivial)
            self.datacube = DataCube(shape=self.shape, 
                bandpasses=bandpasses, pix_scale=self.scale)
        self._setup_imap()
        self._setup_vmap()
        self._setup_SED()
        self.use_numba = self.pars.meta.get('use_numba', False)
        print("DataSimulator initialized!")
        return

    def _setup_vmap(self):
        ''' Setup the observer-plane LoS velocity distribution
        
        theta: list of real
            sampled parameters
        '''
        # collect related pars
        vmap_pars = {
            'v_unit': self.pars.meta['velocity']['v_unit'],
            'r_unit': self.pars.meta['velocity']['r_unit'],
        }
        model_name = self.pars.meta['velocity']['model_name']
        self.vmap = VelocityMap(model_name, self.pars.sampled, vmap_pars)
        return

    def _setup_imap(self):
        ''' Setup the observer-plane intensity profile
        
        '''
        # collect related pars
        imap_pars = {
            'flux': 1, # SED will deal with physical flux
            'hlr': self.pars.meta['intensity']['hlr'],
            'scale': self.scale,
            'Nx': self.Nx,
            'Ny': self.Ny,
        }
        # default/basis/inclined_exp
        imap_type = self.pars.meta['intensity']['type']
        # IntensityMap object
        self.imap = intensity.build_intensity_map(imap_type, self.datacube, 
            imap_pars)
        return

    def _setup_SED(self):
        ''' Initialize the SED interpolation

        theta: 1d array
            sampled parameters. Currently the parameters being sampled
            (g1, g2, theta_int, sini, v0, vcirc, rscale) will not affect
            sed. But in general, we may consider update the SED during 
            MCMC sampling.

        '''
        self.sed = Spectrum(self.pars.meta['sed'])
        return

    def evaluateTheoryModel(self, theta):
        ''' Evaluate the theory model cube based on theta and Pars
    
        theta: 1d array
            parameters being sampled

        '''
        # evaluate intensity profile, velocity distribution and SED
        # currently do not add SF knots and PSF
        self.imap_img, self.gal = self.imap.render(
            self.pars.theta2pars(theta), self.datacube, 
            {'return_GSObject_blob': True}, redo=True
            )
        if(True):
            print("[debug]: add slit mask")
            # JX: slit mask, for debug
            _slit_mask = np.zeros((self.Nx, self.Ny))
            _slit_mask[self.Nx//2-1::self.Nx//2+2,:] = 1.0
            assert _slit_mask.shape == self.imap_img.data.shape, \
            f'Inconsistent array'+\
            f' shape! _slit_mask = {_slit_mask.shape} slice = {self.imap_img.data.shape}'
            self.imap_img *= _slit_mask

        # velocity map for Doppler shift calculation
        self.vmap_img = self.vmap(
            theta, 'obs', self.X, self.Y, 
            normalized=True, use_numba=self.use_numba
            )
        # velocity map for display and debug
        #V = self.vmap('obs', theta, X, Y, use_numba=self.use_numba)

        # build Doppler-shifted datacube
        w_mesh = np.outer(self.lambda_cen, 1./(1.+self.vmap_img))
        w_mesh = w_mesh.reshape(self.lambda_cen.shape+self.vmap_img.shape)
        dc_array = self.sed.spectrum(w_mesh.flatten())
        dc_array = dc_array.reshape(w_mesh.shape) * \
                        self.imap_img[np.newaxis, :, :]
        self._data = dc_array

        return

    def evaluateSimulatedData(self, theta, force_noise_free = False):
        datavector = []
        covariance = []
        self.evaluateTheoryModel(theta)
        assert (self._data.shape == self.shape), "Invalid _data shape!"

        for _ds in self.data_generators:
            if _ds.TYPE == 'grism':
                _img, _cov = _ds.stack(
                    self._data, self.lambdas, force_noise_free
                    )
            elif _ds.TYPE == 'photometry':
                assert (self.gal is not None), "galaxy model missing!"
                _gal_chromatic = self.gal * self.sed.spectrum
                _img, _cov = _ds.stack(
                    _gal_chromatic, self.lambdas, force_noise_free
                    )
            else:
                raise TypeError(f'{_ds.TYPE} is not supported now!')
            datavector.append(_img)
            covariance.append(_cov)
        return datavector, covariance

    def truncate(self, blue_cut, red_cut, trunc_type='edge'):
        '''
        Return a truncated DataCube to slices between blue_cut and
        red_cut using either the lambda on a slice center or edge
        '''

        for l in [blue_cut, red_cut]:
            if (not isinstance(l, float)) and (not isinstance(l, int)):
                raise ValueError('Truncation wavelengths must be ints or floats!')

        if (blue_cut >= red_cut):
            raise ValueError('blue_cut must be less than red_cut!')

        if trunc_type not in ['edge', 'center']:
            raise ValueError('trunc_type can only be at the edge or center!')

        if trunc_type == 'center':
            # truncate on slice center lambda value
            lambda_means = np.mean(self.lambdas, axis=1)

            cut = (lambda_means >= blue_cut) & (lambda_means <= red_cut)

        else:
            # truncate on slice lambda edge values
            lambda_blues = np.array([self.lambdas[i][0] for i in range(self.Nspec)])
            lambda_reds  = np.array([self.lambdas[i][1] for i in range(self.Nspec)])

            cut = (lambda_blues >= blue_cut) & (lambda_reds  <= red_cut)

        # could either update attributes or return new DataCube
        # for now, just return a new one
        trunc_data = self._data[cut,:,:]

        return trunc_data


    def write(self, outfile):
        d = os.path.dirname(outfile)

        utils.make_dir(d)
        if self._data:
            gs.fits.writeCube(self._data, outfile)
        else:
            print("Theory model cube not evaluated!")

        return