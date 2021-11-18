import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import corner
import zeus
from astropy.units import Unit
import sys
sys.path.append('../../kl_tools/')
import basis
from cube import DataCube, ModelCube
import intensity
import likelihood
import mcmc
import parameters
import priors
import utils
from velocity import VelocityMap
from spectrum import Spectrum
#from likelihood_grism_test import _interp2d
import scipy
from scipy import interpolate
import galsim as gs
import galsim.roman as roman
from itertools import repeat


class Mock(object):
    '''
    Mock class object deals with the mock observation image under specific 
    instrument and observation settings. It takes theta and pars dict to 
    generate mock data.
    
    TODO: currently only include grism modelling. 
    '''
    ### settings for the whole class
    
    # default sampled parameters
    _default_theta_pars = {
        'g1':0.0,
        'g2':0.0,
        'theta_int': 0.,# rad
        'sini':0.2,
        'v0':0,# km/s, treated as residual of systematic redshift
        'vcirc':200,# km/s
        'rscale':0.5,# arcsec
    }
    _default_pars = {
        # priors of sampled parameters
        'priors':{
            'g1': priors.GaussPrior(0., 0.1),
            'g2': priors.GaussPrior(0., 0.1),
            'theta_int': priors.UniformPrior(-np.pi/2., np.pi/2.),
            'sini': priors.UniformPrior(0., 1.),
            'v0': priors.GaussPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 10),
            'rscale': priors.UniformPrior(0, 10),
        },
        # dimension parameters for model cube
        # Note that the dimensions and scales in this parameter are irrelevant
        # to the pixel scale and number of pixels in the observed image. The 
        # model cube object is by design independent of the configurations of
        # observations. A parametric surface brightness profile is drawn on 
        # this meta-grid, then observed data will be interpolated from that. 
        'meta':{
            'Nx':256,
            'Ny':256,
            'scale':0.1,# arcsec
            'lambda_range':[100, 1000],
            'lambda_res': 0.5,# nm
        },
        # intensity profile, or surface brightness model for the galaxy
        'intensity':{
            'type':'inclined_exp',
            'hlr':0.5,# arcsec
        },
        # kinematics model for the galaxy
        'velocity':{
            'v_unit':Unit('km / s'),
            'r_unit':Unit('arcsec'),
        },
        # SED model for the galaxy
        'SED':{
            'template':'../../data/Simulation/GSB2.spec',
            'wave_type': 'Ang',
            'flux_type': 'flambda',
            'z': 0.0,
            'wave_range': (50, 50000), # nm
            # obs-frame continuum normalization (nm, erg/s/cm2/nm)
            'obs_cont_norm': (400, 0.),
            # a dict of line names and obs-frame flux values (erg/s/cm2)
            'lines': {'Halpha': 1e-15},
            # intrinsic linewidth in nm
            'line_sigma_int': {'Halpha': 0.5,},
        },
        # observation related parameters
        # Note that from the same galaxy and shear model, we can derive 
        # multiple types of data,
        # e.g. `photometry`, `slit_spectroscopy`, `grism` and/or `IFU`.
        # Thus 'observations' is a list of dictionaries that specify 
        # observation-wise parameters.
        # TODO: maybe think about more flexible way to describe dispersion
        'observations':[
            
            # Roman WFI/GRISM observation, roll angle 1
            {'inst_name':'Roman/WFI',
             'type':'grism',
             'bandpass':'../../data/Bandpass/Roman/WFI_Grism_1stOrder.dat',
             'Nx': 128,# number of pixels
             'Ny': 128,
             'pixel_scale': 0.11,# arcsec
             'R_spec':461,# at 1 micron
             # can be 'airy'/'moffat'/'kolmogorov'/'vonkarman'/'opticalpsf'
             'psf_type':'airy',
             # pass the needed params to build PSF model here
             # in case of airy, we don't need any params
             'psf_kwargs':{'fwhm':0.13},# arcsec
             'disp_ang':0.,# radian
             'diameter':240,# cm
             'exp_time':30.,# seconds
             'gain':1.,
             'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5}
             },
            # Roman WFI/GRISM observation, roll angle 2
            {'inst_name':'Roman/WFI',
             'type':'grism',
             'bandpass':'../../data/Bandpass/Roman/WFI_Grism_1stOrder.dat',
             'Nx': 128,# number of pixels
             'Ny': 128,
             'pixel_scale': 0.11,# arcsec
             'R_spec':461,# at 1 micron
             'psf_type':'airy',
             'psf_kwargs':{'fwhm':0.13},# arcsec
             'disp_ang':np.pi/2.,# radian
             'diameter':240,# cm
             'exp_time':30.,# seconds
             'gain':1.,
             'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5}
             },
            # Roman WFI/Photometry image observation, H band
            {'inst_name':'Roman/WFI',
             'type':'photometry',
             'bandpass':'../../data/Bandpass/Roman/WFI_F129.dat',
             'Nx': 128,# number of pixels
             'Ny': 128,
             'pixel_scale': 0.11,# arcsec
             'psf_type':'airy',
             'psf_kwargs':{'fwhm':0.13},# arcsec
             'diameter':240,# cm
             'exp_time':30.,# seconds
             'gain':1.,
             'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5},
             },
        ],
        'use_numba': False,
    }
    
    def __init__(self, theta_pars, pars):
        # init pars
        self.theta_pars = Mock._default_theta_pars.copy()
        self.pars = Mock._default_pars.copy()
        self.updatePars(theta_pars, pars)
        # build x-y-lambda 3D model cube
        self.V_img, self.imap_img = self.getModeCube()
        
    def updatePars(self, theta_pars, pars):
        for key, val in theta_pars.items():
            self.theta_pars[key] = val
        for key, val in pars.items():
            if isinstance(val, dict):
                for key2, val2 in val.items():
                    self.pars[key][key2] = val2
            else:
                self.pars[key] = val
                
    def getModeCube(self):
        ''' Generate model data cube used for data generation

        To keep the model cube as generic as possible, the model cube should
        not contain any instrument-dependent properties. Thus band-pass and 
        PSF will be applied during observation generation. This may sacrifice
        the speed of the code, but let's see what gonna to happen...
        '''

        # initialize cube grid
        Nx, Ny = self.pars['meta']['Nx'], self.pars['meta']['Ny']
        wave_lim_blue, wave_lim_red = self.pars['meta']['lambda_range']
        wave_res = self.pars['meta']['lambda_res']
        wave_grid = np.arange(wave_lim_blue, wave_lim_red+wave_res, wave_res)
        wave_grid_blue = wave_grid[:-1]
        wave_grid_red = wave_grid[1:]
        wave_grid = (wave_grid_blue+wave_grid_red)/2.
        Nwave = wave_grid.size

        # unit bandpasses, abuse API...
        trivial_bps = [gs.Bandpass(1, 'nm', blue_limit=l, red_limit=l+wave_res,
            zeropoint=0.) for l in wave_grid]
        dc = DataCube(shape=[Nx, Ny, Nwave], bandpasses=trivial_bps)

        # initialize intensity (shear applied)
        self._setup_imap(dc)
        # since we do not convolve PSF here, pass an empty dict to `pars`
        imap_img = self.imap.render(self.theta_pars, dc, {})

        # initialize normalized velocity map V = v/c (shear applied)
        self._setup_vmap()
        # evaluate LoS velocity (v/c) on 2D grid
        # Note: from this step, x-axis is the fast axis
        # Note: GalSim also assume x-axis is the fast axis...
        X, Y = utils.build_map_grid(Nx, Ny, 
                                        scale=self.pars['meta']['scale'])
        vmap_img = self.vmap('obs', X, Y, normalized=True)
        # We use this one for the return map
        V = self.vmap('obs', X, Y)

        # initialize SED
        self._setup_SED()

        # Do Doppler shift mapping
        # map the lambda of 3D model cube grid to new one
        # this step is setting the shape of model cube to [Nwave,Ny,Nx]
        w_mesh = np.outer(wave_grid, 1./(1+vmap_img))
        w_mesh = w_mesh.reshape(wave_grid.shape+vmap_img.shape)
        # get the SED on new lambda grid
        # TODO: use multiprocessing map, or schwimmbad map?
        dc_array = self.SED.spectrum(w_mesh.flatten())
        #dc_array = np.array(list(map(SED.spectrum, w_mesh.flatten())))
        dc_array = dc_array.reshape(w_mesh.shape) * imap_img[np.newaxis,:,:]

        self.modelcube = ModelCube(data=dc_array, bandpasses=trivial_bps)
        self.modelcube_X = X
        self.modelcube_Y = Y
        self.modelcube_wave = wave_grid
        self.modelcube_wave_edge = np.column_stack([wave_grid_blue, 
                                                   wave_grid_red])
        self.modelcube_d = np.sqrt(Nx*Nx+Ny*Ny)*self.pars['meta']['scale']

        return V, imap_img
        
    def _setup_vmap(self, model_name='default'):
        ''' Setup the observer-plane LoS velocity distribution
        Note: the x-axis is the fast axis
        '''
        # collect related pars
        vmap_pars = self.theta_pars.copy()
        vmap_pars['v_unit'] = self.pars['velocity']['v_unit']
        vmap_pars['r_unit'] = self.pars['velocity']['r_unit']
        self.vmap = VelocityMap(model_name, vmap_pars)
        return

    def _setup_imap(self, datacube):
        ''' Setup the observer-plane intensity profile
        Note: the x-axis is the fast axis
        '''
        # collect related pars
        imap_pars = {
            'flux': 1, # SED will deal with physical flux
            'hlr': self.pars['intensity']['hlr'],
            'scale': self.pars['meta']['scale'],
        }
        imap_type = self.pars['intensity']['type']
        self.imap = intensity.build_intensity_map(imap_type, datacube, 
            imap_pars)
        return

    def _setup_SED(self, rel_err=-1):
        self.SED = Spectrum(self.pars['SED'], rel_err=rel_err)
        return


    def getExposure(self):
        datavector = []
        covariance = []

        for obs in self.pars['observations']:
            # check noise setting
            if obs.get('noise', None) is not None:
                _noise_pars = obs.get('noise', None).copy()
                _noise_pars['gain'] = obs.get('gain', 1.0)
            else:
                _noise_pars = None
            # get exposure
            if obs['type'].lower() == 'grism':
                grism_img, noise_img = self._getGrismExposure(obs, 
                    noise_pars=_noise_pars)
                datavector.append(grism_img)
                covariance.append(noise_img)
            elif obs['type'].lower() == 'photometry':
                photometry_img, noise = self._getPhotometryExposure(obs, 
                    noise_pars=_noise_pars)
                datavector.append(photometry_img)
                covariance.append(noise_img)
            else:
                raise ValueError(f'Currently only grism and photometry'+\
                    f' types observations are implemented!')
        return datavector, covariance

    def _getGrismExposure(self, pars, noise_pars=None):
        ''' Generate GRISM observations
            
            {'inst_name':'Roman/WFI',
             'type':'grism',
             'bandpass':'../../data/Bandpass/Roman/WFI_Grism_1stOrder.dat',
             'Nx': 128,# number of pixels
             'Ny': 128,
             'offset':0, # grism field center = galaxy center + offset, in pix
             'pixel_scale': 0.11,# arcsec
             'R_spec':461,# at 1 micron
             'psf_type':'airy',
             'psf_kwargs':{'fwhm':0.13},
             'gain':1.0,
             'disp_ang':0.,# radian
             'diameter':240,# cm
             'exp_time':30.,# seconds
            },
        '''
        # build grism bandpass
        bandpass = gs.Bandpass(pars['bandpass'], wave_type='nm')
        # disperse and project the model cube into grism image
        # wavelength-dependent PSF and bandpass are applied as well
        # Note: modelcube in shape (Nwave, Ny, Nx)
        _grism_list = list(\
            map(self._getGrismDispersed, self.modelcube._data,
                self.modelcube_wave, self.modelcube_wave_edge,
                repeat(bandpass), repeat(pars)))
        grism_img = np.sum(_grism_list, axis=0)

        # apply noise
        if noise_pars is None:
            return grism_img, noise_pars
        else:
            noise = self._getNoise(noise_pars)
            grism_img_withNoise = grism_img.copy()
            grism_img_withNoise.addNoise(noise)
            noise_img = grism_img_withNoise - grism_img
            return grism_img_withNoise, noise_img

    def _getGrismDispersed(self, modelslice, lambdas, wave_edge, 
        bandpass, pars):
        ''' Disperse the model cube into grism image

        modelslice: iterable, 3D numpy array of shape [Nwave, Nx, Ny]
        wave_edge: iterable, 1D numpy array of shape [Nwave,]

        '''
        area = np.pi*(pars['diameter']/2)**2 # cm2

        # build InterpolatedImage from model slice
        assert modelslice.shape == (self.pars['meta']['Ny'], 
            self.pars['meta']['Nx'])
        _img = gs.Image(modelslice, make_const=True, 
                    scale=self.pars['meta']['scale'])
        _gal = gs.InterpolatedImage(_img, scale=self.pars['meta']['scale'])
        # slice bandpass
        slice_bp = bandpass.truncate(blue_limit=wave_edge[0], 
                                     red_limit=wave_edge[1])
        # apply PSF
        # Note that galsim.Airy takes diameter in m and lambda in nm
        PSF = self._build_PSF_model(pars, lam=lambdas, 
                                    diam=pars['diameter']/100.)
        if PSF is not None:
            _gal = gs.Convolve([_gal, PSF])
        # calculate dispersion shift, in units of pixels
        # Note: shift = (dx[pix], dy[pix])
        shift = self._get_dispersion(lambdas, pars)
        # draw slice image
        _grism = _gal.drawImage(nx=pars['Nx'], ny=pars['Ny'], 
                                scale=pars['pixel_scale'], method='auto',
                                area=area, exptime=pars['exp_time'],
                                gain=pars['gain'], offset=shift,
                                bandpass=slice_bp,
                                )
        # noise will be applied after the slices are summed
        return _grism


    def _get_dispersion(self, lam, pars):
        ''' Dispersion relation
        This is a wrapper for instrument dispersion relation
        TODO: consider a LookUpTable for more flexible dispersion?

        For a galaxy at real position (xcen,ycen), and with
        dispersion angle theta, the wavelength lam gets dispersed
        to the new position:
            x = xcen + (lam * dx/dlam + offset) * cos(theta),
            y = ycen + (lam * dx/dlam + offset) * sin(theta)
        '''
        # Note that currently the Roman grism has R = 461*lambda[um],
        # the resulting displacement in number of pixels is
        #   shift = lambda * (2/deltalambda)
        #         = lambda * (2/1um*R) = lambda / 500nm * R
        # where R is the spectral resolution at 1000nm
        # so the dispersion delta(lambda) per pixel is fixed, and
        # the displacement as a function of wavelength is a linear relation
        # But in more general cases, the dispersion relation is non-linear,
        # and interpolation is needed.

        dxdlam = pars['R_spec']/500.
        disp_direction = np.array([np.cos(pars['disp_ang']), 
                                   np.sin(pars['disp_ang'])])
        offset = pars['offset']
        return (lam * dxdlam + offset)*disp_direction

        
    def _build_PSF_model(self, pars, **kwargs):
        psf_type = pars['psf_type'].lower()
        if psf_type == 'airy':
            lam = kwargs.get('lam', 1000) # nm
            diam = kwargs.get('diam', 2.4) # m
            scale_unit = kwargs.get('scale_unit', gs.arcsec)
            return gs.Airy(lam=lam, diam=diam, scale_unit=scale_unit)
        elif psf_type == 'moffat':
            beta = kwargs.get('beta', 2.5)
            fwhm = kwargs.get('fwhm', 0.5)
            return gs.Moffat(beta=beta, fwhm=fwhm)
        elif psf_type == 'none':
            return None
        else:
            raise ValueError(f'{psf_type} has not been implemented yet!')

    def _getPhotometryExposure(self, pars, noise_pars=None):
        ''' Generate Photometry observations
            
            {'inst_name':'Roman/WFI',
             'type':'photometry',
             'bandpass':'../../data/Bandpass/Roman/WFI_F129.dat',
             'Nx': 128,# number of pixels
             'Ny': 128,
             'pixel_scale': 0.11,# arcsec
             'psf_type':'airy',
             'psf_kwargs':{'fwhm':0.13},# arcsec
             'diameter':240,# cm
             'exp_time':30.,# seconds
             'gain':1.,
             },
        '''
        area = np.pi*(pars['diameter']/2)**2 # cm2

        # build bandpass
        bandpass = gs.Bandpass(pars['bandpass'], wave_type='nm')
        # build PSF model, at the center wavelength of bandpass
        PSF = self._build_PSF_model(pars, 
                    lam=bandpass.calculateEffectiveWavelength(),
                    diam=pars['diameter']/100.)
        # build chromatic GSObject
        # use an approximate method here: build a chromatic GSObject
        # More rigorously, we should do grism exposure with zero dispersion
        imap_ary = self.imap.render(self.theta_pars, self.modelcube, {})
        imap_img = gs.Image(imap_ary, scale=self.pars['meta']['scale'],
                            make_const=True)
        _gal = gs.InterpolatedImage(imap_img, 
                                    scale=self.pars['meta']['scale'])
        _gal_chromatic = _gal * self.SED.spectrum
        # apply PSF
        _gal_chromatic = gs.Convolve([_gal_chromatic, PSF])
        # draw photometry image at given bandpass
        photometry_img = _gal_chromatic.drawImage(
                                nx=pars['Nx'], ny=pars['Ny'], 
                                scale=pars['pixel_scale'], method='auto',
                                area=area, exptime=pars['exp_time'],
                                gain=pars['gain'],
                                bandpass=bandpass,)
        # apply noise
        if noise_pars is None:
            return photometry_img, noise_pars
        else:
            noise = self._getNoise(noise_pars)
            photometry_img_withNoise = photometry_img.copy()
            photometry_img_withNoise.addNoise(noise)
            noise_img = photometry_img_withNoise - photometry_img
            return photometry_img_withNoise, noise_img

    def _getNoise(self, noise_pars):
        #'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5}
        noise_type = noise_pars['type'].lower()
        random_seed = noise_pars.get('random_seed', 1314662)
        rng = gs.BaseDeviate(random_seed+1)

        if noise_type == 'ccd':
            sky_level = noise_pars.get('sky_level', 0.65*1.2)
            read_noise = noise_pars.get('read_noise', 0.85)
            gain = noise_pars.get('gain', 1.0)
            noise = gs.CCDNoise(rng=rng, gain=gain, read_noise=read_noise, 
                sky_level=sky_level)
        elif noise_type == 'gauss':
            sigma = noise_pars.get('sigma', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif noise_type == 'poisson':
            sky_level = noise_pars.get('sky_level', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{noise_type} not implemented yet!')
        return noise
