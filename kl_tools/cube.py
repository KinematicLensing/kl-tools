from abc import abstractmethod
import numpy as np
from astropy.io import fits
import galsim
import os
import pickle
from astropy.table import Table
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import galsim as gs

# from . import utils
import utils
from time import time
from parameters import Pars

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class DataVector(object):
    '''
    Light wrapper around things like cube.DataCube to allow for
    uniform interface for things like the intensity map rendering
    '''

    @abstractmethod
    def stack(self, theory_data=None):
        '''
        Each datavector must have a method that defines how to stack it
        into a single (Nx,Ny) image for basis function fitting
        '''
        pass

class ImageGenerator(DataVector):
    ''' Light wrapper around 3D theory model cube, to generate grism data

    Note: 
    - This class object does not save 3D theory model cube. It saves 
    properties related with specific instrument and observations (e.g. PSF, 
    bandpasses, etc.), and provide a realization of the ``stack`` interface
    - The current API design assumes parameters related with instrument and
    observations are not being sampled
    '''
    TYPE = 'photometry'
    def __init__(self, pars):
        ''' Initialize observation-related properties, including:
        - PSF model
        - bandpass
        - etc.

        pars: ``Pars`` object
            the parameters not sampled during MCMC. These parameters will be
            used to initialize the generator
        Let's first write the routine then write the initialization
        '''
        # init theory cube & observed data dimensions
        self.Nx_theory = pars.meta['model_dimension']['Nx']
        self.Ny_theory = pars.meta['model_dimension']['Ny']
        self.scale = pars.meta['model_dimension']['scale']
        self.Nx = pars.meta['observations']['Nx']
        self.Ny = pars.meta['observations']['Ny']
        self.pix_scale = pars.meta['observations']['pixel_scale']
        
        # init bandpass
        self.bandpass_file = pars.meta['observations']['bandpass']
        self.bandpass = gs.Bandpass(self.bandpass_file, wave_type='nm')

        # init psf
        # Note: don't consider chromatic PSF for broadband images so far.
        self.psf_type=pars.meta['observations'].get('psf_type', 'none').lower()
        self.psf_args=pars.meta['observations'].get('psf_kwargs', {})
        self.hasPSF = (self.psf_type != 'none')

        # init noise model
        self.noise_pars = pars.meta['observations'].get('noise', {})
        self.noise_type = self.noise_pars.get('type', 'ccd').lower()
        self.apply_to_data = self.noise_pars.get('apply_to_data', False)

        # others
        self.diameter = pars.meta['observations']['diameter']
        self.area = np.pi * (self.diameter/2.)**2
        self.inst_name = pars.meta['observations']['inst_name']
        self.gain = pars.meta['observations']['gain']
        self.exp_time = pars.meta['observations']['exp_time']
        self.offset = pars.meta['observations'].get('offset', 0.0)
        print("\n[ImageGenerator] Init:")
        print("--- Instrument name = {}".format(self.inst_name))
        print("--- Aperture diameter = {:.2f} cm".format(self.diameter))
        print("--- Aperture area = {:.2f} cm2".format(self.area))
        print("--- Detector gain = {:.2f}".format(self.gain))
        print("--- Exposure time = {:.2f} seconds".format(self.exp_time))
        print("--- Noise type = {}".format(self.noise_type))
        print("--- PSF type = {}".format(self.psf_type))
        print("--- Bandpass file = {}".format(self.bandpass_file))
        print("--- Theory slice dimension = ({:d}, {:d})".format(
            self.Ny_theory, self.Nx_theory))
        print("--- Theory slice scale = {:.2f}".format(self.scale))
        print("--- Observed slice dimension = ({:d}, {:d})".format(
            self.Ny, self.Nx))
        print("--- Observed slice pixel scale = {:.2f}".format(self.pix_scale))

        return

    def stack(self, theory_data, lambdas, force_noise_free=False):
        ''' Generate simulated grism image out of theory 3d model cube

        Inputs:
        =======
        theory_data: IntensityMap object
            the 2d intensity distribution
        lambdas: 1d (Nspec) list of two-elements tuple
            the blue and red limits of each theory cube slice

        Outputs:
        ========
        data: 2d (Ny, Nx) numpy.ndarray
            the 2d stacked image
        noise: 2d (Ny, Nx) numpy.ndarray
            the 2d image noise
        '''
        #if not isinstance(theory_data, gs.GSObject):
        #    t = type(theory_data)
        #    raise TypeError(f'theory_data must be galsim.GSObject object,'+\
        #        f' not {t}!')

        _gal_chromatic = theory_data
        # convolve with PSF
        if self.hasPSF:
            psf = self._build_PSF_model(
                lam=self.bandpass.calculateEffectiveWavelength()
                )
            _gal_chromatic = gs.Convolve([_gal_chromatic, psf])
        photometry_img = _gal_chromatic.drawImage(
                                nx=self.Nx, ny=self.Ny, 
                                scale=self.pix_scale, method='auto',
                                area=self.area, exptime=self.exp_time,
                                gain=self.gain,
                                bandpass=self.bandpass)
        # apply noise
        if force_noise_free:
            return photometry_img.array, None
        else:
            noise = self._getNoise()
            photometry_img_withNoise = photometry_img.copy()
            photometry_img_withNoise.addNoise(noise)
            noise_img = photometry_img_withNoise - photometry_img
            assert (photometry_img_withNoise.array is not None), \
                    "Null photometry data"
            assert (photometry_img.array is not None), "Null photometry data"
            if self.apply_to_data:
                #print("[ImageGenerator][debug]: add noise")
                return photometry_img_withNoise.array, noise_img.array
            else:
                #print("[ImageGenerator][debug]: noise free")
                return photometry_img.array, noise_img.array

    def _build_PSF_model(self, **kwargs):
        ''' Generate PSF model

        Inputs:
        =======
        kwargs: keyword arguments for building psf model
            - lam: wavelength in nm
            - scale: pixel scale

        Outputs:
        ========
        psf_model: GalSim PSF object

        '''
        if self.psf_type is not None:
            if self.psf_type == 'airy':
                lam = kwargs.get('lam', 1000) # nm
                scale = kwargs.get('scale_unit', gs.arcsec)
                return gs.Airy(lam=lam, diam=self.diameter/100,
                                scale_unit=scale)
            elif self.psf_type == 'moffat':
                beta = self.psf_args.get('beta', 2.5)
                fwhm = self.psf_args.get('fwhm', 0.5)
                return gs.Moffat(beta=beta, fwhm=fwhm)
            else:
                raise ValueError(f'{psf_type} has not been implemented yet!')
        else:
            return None

    def _getNoise(self):
        ''' Generate image noise based on parameter settings

        Outputs:
        ========
        noise: GalSim Noise object
        '''
        random_seed = self.noise_pars.get('random_seed', int(time()))
        rng = gs.BaseDeviate(random_seed+1)

        if self.noise_type == 'ccd':
            sky_level = self.noise_pars.get('sky_level', 0.65*1.2)
            read_noise = self.noise_pars.get('read_noise', 8.5)
            # sky_level: electron/s/pix -> ADU/pix
            noise = gs.CCDNoise(rng=rng, gain=self.gain, 
                                read_noise=read_noise, 
                                sky_level=sky_level*self.exp_time/self.gain)
        elif self.noise_type == 'gauss':
            sigma = self.noise_pars.get('sigma', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif self.noise_type == 'poisson':
            sky_level = self.noise_pars.get('sky_level', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{self.noise_type} not implemented yet!')
        return noise

class GrismGenerator(DataVector):
    ''' Light wrapper around 3D theory model cube, to generate grism data

    Note: 
    - This class object does not save 3D theory model cube. It saves 
    properties related with specific instrument and observations (e.g. PSF, 
    bandpasses, etc.), and provide a realization of the ``stack`` interface
    - The current API design assumes parameters related with instrument and
    observations are not being sampled
    '''
    TYPE = 'grism'
    _CHROMATIC_PSF_ = ('airy', 'kolmogorov', 'vonkarman', 'opticalpsf')
    def __init__(self, pars):
        ''' Initialize observation-related properties, including:
        - PSF model
        - bandpass
        - etc.

        pars: ``Pars`` object
            the parameters not sampled during MCMC. These parameters will be
            used to initialize the generator
        Let's first write the routine then write the initialization
        '''
        # init theory cube & observed data dimensions
        self.Nx_theory = pars.meta['model_dimension']['Nx']
        self.Ny_theory = pars.meta['model_dimension']['Ny']
        self.scale = pars.meta['model_dimension']['scale']
        self.Nx = pars.meta['observations']['Nx']
        self.Ny = pars.meta['observations']['Ny']
        self.pix_scale = pars.meta['observations']['pixel_scale']
        
        # init bandpass
        self.bandpass_file = pars.meta['observations']['bandpass']
        self.bandpass = gs.Bandpass(self.bandpass_file, wave_type='nm')

        # init psf
        self.psf_type=pars.meta['observations'].get('psf_type', 'none').lower()
        self.psf_args=pars.meta['observations'].get('psf_kwargs', {})
        self.hasPSF = (self.psf_type != 'none')
        self.hasChromaticPSF = (self.psf_type in 
            GrismGenerator._CHROMATIC_PSF_)

        # init noise model
        self.noise_pars = pars.meta['observations'].get('noise', {})
        self.noise_type = self.noise_pars.get('type', 'ccd').lower()
        self.apply_to_data = self.noise_pars.get('apply_to_data', False)

        # init dispersion
        self.R_spec = pars.meta['observations']['R_spec']
        self.disp_ang = pars.meta['observations']['disp_ang']
        self._init_dispersion_relation()

        # others
        self.diameter = pars.meta['observations']['diameter']
        self.area = np.pi * (self.diameter/2.)**2
        self.inst_name = pars.meta['observations']['inst_name']
        self.gain = pars.meta['observations']['gain']
        self.exp_time = pars.meta['observations']['exp_time']
        self.offset = pars.meta['observations'].get('offset', 0.0)
        print("\n[GrismGenerator] Init:")
        print("--- Instrument name = {}".format(self.inst_name))
        print("--- Aperture diameter = {:.2f} cm".format(self.diameter))
        print("--- Aperture area = {:.2f} cm2".format(self.area))
        print("--- Detector gain = {:.2f}".format(self.gain))
        print("--- Exposure time = {:.2f} seconds".format(self.exp_time))
        print("--- Noise type = {}".format(self.noise_type))
        print("--- PSF type = {}".format(self.psf_type))
        print("--- Bandpass file = {}".format(self.bandpass_file))
        print("--- Theory slice dimension = ({:d}, {:d})".format(
            self.Ny_theory, self.Nx_theory))
        print("--- Theory slice scale = {:.2f}".format(self.scale))
        print("--- Observed slice dimension = ({:d}, {:d})".format(
            self.Ny, self.Nx))
        print("--- Observed slice pixel scale = {:.2f}".format(self.pix_scale))
        print("--- Dispersion angle = {:.2f} deg".format(
            self.disp_ang/np.pi*180))
        print("--- Spectral resolution at 1um = {:.2f}".format(self.R_spec))
        print("--- Grism offset = {} pixels".format(self.offset))

        return

    def stack(self, theory_data, lambdas, force_noise_free = False):
        ''' Generate simulated grism image out of theory 3d model cube

        Inputs:
        =======
        theory_data: 3d (Nspec_theory, Ny_theory, Nx_theory) numpy.ndarray
            the 3d intensity distribution
        lambdas: 1d (Nspec) list of two-elements tuple
            the blue and red limits of each theory cube slice

        Outputs:
        ========
        data: 2d (Ny, Nx) numpy.ndarray
            the 2d stacked image
        noise: 2d (Ny, Nx) numpy.ndarray
            the 2d image noise
        '''
        if not isinstance(theory_data, np.ndarray):
            raise TypeError(f'theory_data must be numpy.ndarray object!')
        if theory_data.shape[0] != lambdas.shape[0]:
            raise ValueError(f'theory_data and lambdas must have the same'+\
                ' dimension in along axis 0')
        # disperse and project the theory 3d model cube into grism image
        _grism_list = list(map(self._disperse, theory_data, lambdas))
        grism_img = np.sum(_grism_list, axis=0)
        # convolve with achromatic psf, if required
        if self.hasPSF and not self.hasChromaticPSF:
            psf = self._build_PSF_model()
            _gal = gs.InterpolatedImage(grism_img, scale=self.pix_scale)
            grism_gal = gs.Convolve([_gal, psf])
            grism_img = grism_gal.drawImage(nx=self.Nx, ny=self.Ny, 
                                            scale=self.pix_scale)

        # apply noise
        if force_noise_free:
            return grism_img.array, None
        else:
            noise = self._getNoise()
            grism_img_withNoise = grism_img.copy()
            grism_img_withNoise.addNoise(noise)
            noise_img = grism_img_withNoise - grism_img
            assert (grism_img_withNoise.array is not None), "Null grism data"
            assert (grism_img.array is not None), "Null grism data"
            if self.apply_to_data:
                #print("[GrismGenerator][debug]: add noise")
                return grism_img_withNoise.array, noise_img.array
            else:
                #print("[GrismGenerator][debug]: noise free")
                return grism_img.array, noise_img.array

    def _disperse(self, theory_slice, lambdas):
        ''' Disperse a single slice of theory 3d model cube

        Inputs:
        =======
        theory_slice: 2d (Ny_theory, Nx_theory) numpy.ndarray
            a single slice of the theory 3d model
        lambdas: two-elements tuple
            the blue and red limits of one theory cube slice

        Outputs:
        ========
        data: 2d (Ny, Nx) numpy.ndarray
            the corresponding grism image if the input slice is dispersed

        '''
        # theory_slice in units of photons/s/cm2
        _img = gs.Image(theory_slice, make_const=True, scale=self.scale)
        _gal = gs.InterpolatedImage(_img, scale=self.scale)
        slice_bandpass = self.bandpass.truncate(blue_limit=lambdas[0], 
                                            red_limit=lambdas[1])
        # if we adopt chromatic PSF, convolve with PSF model here
        if self.hasPSF and self.hasChromaticPSF:
            psf = self._build_PSF_model(lam=np.mean(lambdas))
            _gal = gs.Convolve([_gal, psf]) 
        # get dispersion shift, in units of pixels
        # Note: shift = (dx, dy)
        shift = self.dispersion_relation((lambdas[0]+lambdas[1])/2.)
        # draw slice image
        _grism = _gal.drawImage(nx=self.Nx, ny=self.Ny, scale=self.pix_scale,
                                method='auto', area=self.area, 
                                exptime=self.exp_time, gain=self.gain, 
                                offset=shift,
                                bandpass=slice_bandpass)
        return _grism

    def _init_dispersion_relation(self):
        ''' Initialize Grism dispersion relation

        Note: currently only support linear dispersion relation

        For a galaxy at real position (xcen,ycen), and with
        dispersion angle theta, the wavelength lam gets dispersed
        to the new position:
            x = xcen + (lam * dx/dlam + offset) * cos(theta),
            y = ycen + (lam * dx/dlam + offset) * sin(theta)
        '''
        self.dxdlam = self.R_spec/500.0
        self.disp_vec = np.array([np.cos(self.disp_ang), 
                                   np.sin(self.disp_ang)])
        # lambda expression is not pickleble
        #self.dispersion_relation = \
        #    lambda x: (x * dxdlam + self.offset)*disp_direction

        return

    def dispersion_relation(self, x):
        return (x * self.dxdlam + self.offset)*self.disp_vec

    def _build_PSF_model(self, **kwargs):
        ''' Generate PSF model

        Inputs:
        =======
        kwargs: keyword arguments for building psf model
            - lam: wavelength in nm
            - scale: pixel scale

        Outputs:
        ========
        psf_model: GalSim PSF object

        '''
        if self.psf_type is not None:
            if self.psf_type == 'airy':
                lam = kwargs.get('lam', 1000) # nm
                scale = kwargs.get('scale_unit', gs.arcsec)
                return gs.Airy(lam=lam, diam=self.diameter/100,
                                scale_unit=scale)
            elif self.psf_type == 'moffat':
                beta = self.psf_args.get('beta', 2.5)
                fwhm = self.psf_args.get('fwhm', 0.5)
                return gs.Moffat(beta=beta, fwhm=fwhm)
            else:
                raise ValueError(f'{psf_type} has not been implemented yet!')
        else:
            return None

    def _getNoise(self):
        ''' Generate image noise based on parameter settings

        Outputs:
        ========
        noise: GalSim Noise object
        '''
        random_seed = self.noise_pars.get('random_seed', int(time()))
        rng = gs.BaseDeviate(random_seed+1)

        if self.noise_type == 'ccd':
            sky_level = self.noise_pars.get('sky_level', 0.65*1.2)
            read_noise = self.noise_pars.get('read_noise', 8.5)
            noise = gs.CCDNoise(rng=rng, gain=self.gain, 
                                read_noise=read_noise, 
                                sky_level=sky_level*self.exp_time/self.gain)
        elif self.noise_type == 'gauss':
            sigma = self.noise_pars.get('sigma', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif self.noise_type == 'poisson':
            sky_level = self.noise_pars.get('sky_level', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{self.noise_type} not implemented yet!')
        return noise


class SlitSpecGenerator(DataVector):
    ''' Light wrapper around 3D theory model cube, to generate slit spectrum

    Note: 
    - This class object does not save 3D theory model cube. It saves 
    properties related with specific instrument and observations (e.g. PSF, 
    bandpasses, etc.), and provide a realization of the ``stack`` interface
    - The current API design assumes parameters related with instrument and
    observations are not being sampled
    '''
    TYPE = 'slit'
    _CHROMATIC_PSF_ = ('airy', 'kolmogorov', 'vonkarman', 'opticalpsf')
    def __init__(self, pars):
        ''' Initialize observation-related properties, including:
        - PSF model
        - bandpass
        - etc.

        pars: ``Pars`` object
            the parameters not sampled during MCMC. These parameters will be
            used to initialize the generator
        Let's first write the routine then write the initialization
        '''
        # init theory cube & observed data dimensions
        self.Nx_theory = pars.meta['model_dimension']['Nx']
        self.Ny_theory = pars.meta['model_dimension']['Ny']
        self.scale = pars.meta['model_dimension']['scale']
        self.Nx = pars.meta['observations']['Nx']
        self.Ny = pars.meta['observations']['Ny']
        self.pix_scale = pars.meta['observations']['pixel_scale']
        
        # init bandpass
        self.bandpass_file = pars.meta['observations']['bandpass']
        self.bandpass = gs.Bandpass(self.bandpass_file, wave_type='nm')

        # init psf
        self.psf_type=pars.meta['observations'].get('psf_type', 'none').lower()
        self.psf_args=pars.meta['observations'].get('psf_kwargs', {})
        self.hasPSF = (self.psf_type != 'none')
        self.hasChromaticPSF = (self.psf_type in 
            GrismGenerator._CHROMATIC_PSF_)

        # init noise model
        self.noise_pars = pars.meta['observations'].get('noise', {})
        self.noise_type = self.noise_pars.get('type', 'none').lower()
        self.apply_to_data = self.noise_pars.get('apply_to_data', False)

        # init dispersion
        self.R_spec = pars.meta['observations']['R_spec']
        self.disp_ang = pars.meta['observations']['disp_ang']
        self._init_dispersion_relation()

        # init slit mask
        self.slit_width = pars.meta['observations'].get('slit_width', 2.0)
        self.slit_offset = pars.meta['observations'].get('slit_offset', 0.0)
        self._init_slit_mask()

        # others
        self.diameter = pars.meta['observations']['diameter']
        self.area = np.pi * (self.diameter/2.)**2
        self.inst_name = pars.meta['observations']['inst_name']
        self.gain = pars.meta['observations']['gain']
        self.exp_time = pars.meta['observations']['exp_time']
        self.offset = pars.meta['observations'].get('offset', 0.0)
        print("\n[SlitSpecGenerator] Init:")
        print("--- Instrument name = {}".format(self.inst_name))
        print("--- Aperture diameter = {:.2f} cm".format(self.diameter))
        print("--- Aperture area = {:.2f} cm2".format(self.area))
        print("--- Detector gain = {:.2f}".format(self.gain))
        print("--- Exposure time = {:.2f} seconds".format(self.exp_time))
        print("--- Noise type = {}".format(self.noise_type))
        print("--- PSF type = {}".format(self.psf_type))
        print("--- Bandpass file = {}".format(self.bandpass_file))
        print("--- Theory slice dimension = ({:d}, {:d})".format(
            self.Ny_theory, self.Nx_theory))
        print("--- Theory slice scale = {:.2f}".format(self.scale))
        print("--- Observed slice dimension = ({:d}, {:d})".format(
            self.Ny, self.Nx))
        print("--- Observed slice pixel scale = {:.2f}".format(self.pix_scale))
        print("--- Dispersion angle = {:.2f} deg".format(
            self.disp_ang/np.pi*180))
        print("--- Spectral resolution at 1um = {:.2f}".format(self.R_spec))
        print("--- Spectrum offset = {} pixels".format(self.offset))
        print("--- Slit width (FWHM) = {} arcsec".format(self.slit_width))
        print("--- Slit offset = {} arcsec".format(self.slit_offset))

        return

    def stack(self, theory_data, lambdas, force_noise_free = False):
        ''' Generate simulated grism image out of theory 3d model cube

        Inputs:
        =======
        theory_data: 3d (Nspec_theory, Ny_theory, Nx_theory) numpy.ndarray
            the 3d intensity distribution
        lambdas: 1d (Nspec) list of two-elements tuple
            the blue and red limits of each theory cube slice

        Outputs:
        ========
        data: 2d (Ny, Nx) numpy.ndarray
            the 2d stacked image
        noise: 2d (Ny, Nx) numpy.ndarray
            the 2d image noise
        '''
        if not isinstance(theory_data, np.ndarray):
            raise TypeError(f'theory_data must be numpy.ndarray object!')
        if theory_data.shape[0] != lambdas.shape[0]:
            raise ValueError(f'theory_data and lambdas must have the same'+\
                ' dimension in along axis 0')
        # disperse and project the theory 3d model cube into grism image
        _grism_list = list(map(self._disperse, theory_data, lambdas))
        grism_img = np.sum(_grism_list, axis=0)
        # convolve with achromatic psf, if required
        if self.hasPSF and not self.hasChromaticPSF:
            psf = self._build_PSF_model()
            grism_img = gs.Convolve([grism_img, psf])

        # apply noise
        if force_noise_free:
            return grism_img.array, None
        else:
            noise = self._getNoise()
            grism_img_withNoise = grism_img.copy()
            grism_img_withNoise.addNoise(noise)
            noise_img = grism_img_withNoise - grism_img
            assert (grism_img_withNoise.array is not None), "Null grism data"
            assert (grism_img.array is not None), "Null grism data"
            if self.apply_to_data:
                #print("[SlitSpecGenerator][debug]: add noise")
                return grism_img_withNoise.array, noise_img.array
            else:
                #print("[SlitSpecGenerator][debug]: noise free")
                return grism_img.array, noise_img.array

    def _disperse(self, theory_slice, lambdas):
        ''' Disperse a single slice of theory 3d model cube

        Inputs:
        =======
        theory_slice: 2d (Ny_theory, Nx_theory) numpy.ndarray
            a single slice of the theory 3d model
        lambdas: two-elements tuple
            the blue and red limits of one theory cube slice

        Outputs:
        ========
        data: 2d (Ny, Nx) numpy.ndarray
            the corresponding grism image if the input slice is dispersed

        '''
        #theory_slice *= self.slit_mask
        _img = gs.Image(theory_slice * self.slit_mask, 
            make_const=True, scale=self.scale)
        _gal = gs.InterpolatedImage(_img, scale=self.scale)
        slice_bandpass = self.bandpass.truncate(blue_limit=lambdas[0], 
                                            red_limit=lambdas[1])
        # if we adopt chromatic PSF, convolve with PSF model here
        if self.hasPSF and self.hasChromaticPSF:
            psf = self._build_PSF_model(lam=np.mean(lambdas))
            _gal = gs.Convolve([_gal, psf]) 
        # get dispersion shift, in units of pixels
        # Note: shift = (dx, dy)
        shift = self.dispersion_relation((lambdas[0]+lambdas[1])/2.)
        # draw slice image
        _grism = _gal.drawImage(nx=self.Nx, ny=self.Ny, scale=self.pix_scale,
                                method='auto', area=self.area, 
                                exptime=self.exp_time, gain=self.gain, 
                                offset=shift,
                                bandpass=slice_bandpass)
        return _grism

    def _init_dispersion_relation(self):
        ''' Initialize Grism dispersion relation

        Note: currently only support linear dispersion relation

        For a galaxy at real position (xcen,ycen), and with
        dispersion angle theta, the wavelength lam gets dispersed
        to the new position:
            x = xcen + (lam * dx/dlam + offset) * cos(theta),
            y = ycen + (lam * dx/dlam + offset) * sin(theta)
        '''
        self.dxdlam = self.R_spec/500.0
        self.disp_vec = np.array([np.cos(self.disp_ang), 
                                   np.sin(self.disp_ang)])
        # lambda expression is not pickleble
        #self.dispersion_relation = \
        #    lambda x: (x * dxdlam + self.offset)*disp_direction

        return

    def dispersion_relation(self, x):
        return (x * self.dxdlam + self.offset)*self.disp_vec

    def _build_PSF_model(self, **kwargs):
        ''' Generate PSF model

        Inputs:
        =======
        kwargs: keyword arguments for building psf model
            - lam: wavelength in nm
            - scale: pixel scale

        Outputs:
        ========
        psf_model: GalSim PSF object

        '''
        if self.psf_type is not None:
            if self.psf_type == 'airy':
                lam = kwargs.get('lam', 1000) # nm
                scale = kwargs.get('scale_unit', gs.arcsec)
                return gs.Airy(lam=lam, diam=self.diameter/100,
                                scale_unit=scale)
            elif self.psf_type == 'moffat':
                beta = self.psf_args.get('beta', 2.5)
                fwhm = self.psf_args.get('fwhm', 0.5)
                return gs.Moffat(beta=beta, fwhm=fwhm)
            else:
                raise ValueError(f'{psf_type} has not been implemented yet!')
        else:
            return None

    def _getNoise(self):
        ''' Generate image noise based on parameter settings

        Outputs:
        ========
        noise: GalSim Noise object
        '''
        random_seed = self.noise_pars.get('random_seed', int(time()))
        rng = gs.BaseDeviate(random_seed+1)

        if self.noise_type == 'ccd':
            sky_level = self.noise_pars.get('sky_level', 0.65*1.2)
            read_noise = self.noise_pars.get('read_noise', 8.5)
            noise = gs.CCDNoise(rng=rng, gain=self.gain, 
                                read_noise=read_noise, 
                                sky_level=sky_level*self.exp_time/self.gain)
        elif self.noise_type == 'gauss':
            sigma = self.noise_pars.get('sigma', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif self.noise_type == 'poisson':
            sky_level = self.noise_pars.get('sky_level', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{self.noise_type} not implemented yet!')
        return noise

    def _init_slit_mask(self):
        X, Y = utils.build_map_grid(self.Nx_theory, self.Ny_theory,
            scale=self.scale)

        _r = X*self.disp_vec[0] + Y*self.disp_vec[1] - self.slit_offset
        _slit_sigma = self.slit_width / np.sqrt(8.*np.log(2))
        _mask = np.exp(-0.5 * (_r/_slit_sigma)**2 )
        self.slit_mask = _mask #/ np.sum(_mask)

class DataCube(DataVector):
    '''
    Base class for an abstract data cube.
    Contains astronomical images of a source
    at various wavelength slices
    '''

    def __init__(self, data=None, shape=None, bandpasses=None, pix_scale=None,
                 pars=None):
        '''
        Initialize either a filled DataCube from an existing numpy
        array or an empty one from a given shape

        data: np.array
            A numpy array containing all image slice data.
            For now, assumed to be the shape format given below.
        shape: tuple
            A 3-tuple in the format of (Nspec, Nx, Ny)
            where (Nx, Ny) are the shapes of the image slices
            and Nspec is the Number of spectral slices.
        bandpasses: list
            A list of galsim.Bandpass objects containing
            throughput function, lambda window, etc.
        pix_scale: float
            the pixel scale of the datacube slices
        pars: dict
            A dictionary that holds any additional metadata
        '''

        if data is None:
            if shape is None:
                raise ValueError('Must instantiate a DataCube with either ' + \
                                 'a data array or a shape tuple!')

            self.Nspec = shape[0]
            self.Nx = shape[1]
            self.Ny = shape[2]
            self.shape = shape

            self._check_shape_params()
            self._data = np.zeros(self.shape)

        else:
            if bandpasses is None:
                raise ValueError('Must pass bandpasses if data is not None!')

            if len(data.shape) != 3:
                # Handle the case of 1 slice
                assert len(data.shape) == 2
                data = data.reshape(1, data.shape[0], data.shape[1])

            self.shape = data.shape

            self.Nspec = self.shape[0]
            self.Nx = self.shape[1]
            self.Ny = self.shape[2]

            self._data = data

            if self.shape[0] != len(bandpasses):
                raise ValueError('The length of the bandpasses must ' + \
                                 'equal the length of the third data dimension!')

        # a bit awkward, but this allows flexible setup for other params
        if bandpasses is None:
            raise ValueError('Must pass a list of bandpasses!')
        self.bandpasses = bandpasses

        d = {'pix_scale': (pix_scale, (int, float)), 'pars': (pars, dict)}
        for name, (val, t) in d.items():
            if val is not None:
                if not isinstance(val, t):
                    raise TypeError(f'{name} must be a dict!')

        self.pix_scale = pix_scale
        self.pars = pars

        # Not necessarily needed, but could help ease of access
        self.lambda_unit = self.bandpasses[0].wave_type
        self.lambdas = [] # Tuples of bandpass bounds in unit of bandpass
        for bp in bandpasses:
            li = bp.blue_limit
            le = bp.red_limit
            self.lambdas.append((li, le))

            # Make sure units are consistent
            # (could generalize, but not necessary)
            assert bp.wave_type == self.lambda_unit

        self._construct_slice_list()

        return

    def _check_shape_params(self):
        Nzip = zip(['Nspec', 'Nx', 'Ny'], [self.Nspec, self.Nx, self.Ny])
        for name, val in Nzip:
            if val < 1:
                raise ValueError(f'{name} must be greater than 0!')

        if len(self.shape) != 3:
            raise ValueError('DataCube.shape must be len 3!')

        return

    def _construct_slice_list(self):
        self.slices = SliceList()

        for i in range(self.Nspec):
            bp = self.bandpasses[i]
            self.slices.append(Slice(self._data[i,:,:], bp))

        return

    @property
    def data(self):
        return self._data

    def set_data(self, data):
        if self.shape != data.shape:
            raise ValueError('Wrong data shape! '\
                'Should be {} but get {}'.format(self.shape, data.shape))
        self._data = data 
        return

    def slice(self, indx):
        return self.slices[indx].data

    def stack(self, method='default'):
        ''' Stack the 
        '''
        return np.sum(self._data, axis=0)

    def compute_aperture_spectrum(self, radius, offset=(0,0), plot_mask=False):
        '''
        radius: aperture radius in pixels
        offset: aperture center offset tuple in pixels about slice center
        '''

        mask = np.zeros((self.Nx, self.Ny), dtype=np.dtype(bool))

        im_center = (self.Nx/2, self.Ny/2)
        center = np.array(im_center) + np.array(offset)

        aper_spec = np.zeros(self.Nspec)

        for x in range(self.Nx):
            for y in range(self.Ny):
                dist = np.sqrt((x-center[0])**2+(y-center[1])**2)
                if dist < radius:
                    aper_spec += self._get_pixel_spectrum(x,y)
                    mask[x,y] = True

        if plot_mask is True:
            plt.imshow(mask, origin='lower')

            cx, cy = center[0], center[1]
            circle = plt.Circle((cx,cy), radius, color='r', fill=False,
                                lw=3, label='Aperture')

            ax = plt.gca()
            ax.add_patch(circle)
            plt.legend()
            plt.show()

        return aper_spec

    def plot_aperture_spectrum(self, radius, offset=(0,0), size=None,
                               title=None, outfile=None, show=True,
                               close=True):

        aper_spec = self.compute_aperture_spectrum(radius, offset=offset)
        lambda_means = np.mean(self.lambdas, axis=1)

        plt.plot(lambda_means, aper_spec)
        plt.xlabel(f'Lambda ({self.lambda_unit})')
        plt.ylabel(f'Flux (ADUs)')

        if title is not None:
            plt.title(title)
        else:
            plt.title(f'Aperture spectrum for radius={radius} pixels; ' +\
                      f'offset={offset}')

        if size is not None:
            plt.gcf().set_size_inches(size)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()
        elif close is True:
            plt.close()

        return

    def compute_pixel_spectrum(self, i, j):
        '''
        Compute the spectrum of the pixel (i,j) across
        all slices

        # TODO: Work out units!
        '''

        pix_spec = self._get_pixel_spectrum(i,j)

        # presumably some unit conversion...

        # ...

        return pix_spec

    def _get_pixel_spectrum(self, i, j):
        '''
        Return the raw spectrum of the pixel (i,j) across
        all slices
        '''

        return self._data[:,i,j]

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

        # Have to do it this way as lists cannot be indexed by np arrays
        # trunc_bandpasses = self.bandpasses[cut]
        trunc_bandpasses = [self.bandpasses[i]
                            for i in range(self.Nspec)
                            if cut[i] == True]

        return DataCube(data=trunc_data, bandpasses=trunc_bandpasses)

    def plot_slice(self, slice_index, plot_kwargs):
        self.slices[slice_index].plot(**plot_kwargs)

        return

    def plot_pixel_spectrum(self, i, j, show=True, close=True, outfile=None):
        '''
        Plot the spectrum for pixel (i,j) across
        all slices

        # TODO: Work out units!
        '''

        pix_spec = self.compute_pixel_spectrum(i,j)

        lambda_means = np.mean(self.lambdas, axis=1)
        unit = self.lambda_unit

        plt.plot(lambda_means, pix_spec)
        plt.xlabel(f'Lambda ({unit})')
        plt.ylabel('Flux (ADU)')
        plt.title(f'Spectrum for pixel ({i}, {j})')

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()
        elif close is True:
            plt.close()

        return

    def write(self, outfile):
        d = os.path.dirname(outfile)

        utils.make_dir(d)

        im_list = []
        for s in self.slices:
            im_list.append(s._data)

        gs.fits.writeCube(im_list, outfile)

        return

class FitsDataCube(DataCube):
    '''
    Same as Datacube, but instantiated from a fitscube file
    and associated file containing bandpass list

    We assume the same structure as galsim.fits.writeCube()

    cubefile: location of fits cube
    bandpasses: either a filename of a bandpass list or the list
    '''

    def __init__(self, cubefile, bandpasses, dir=None):
        if dir is not None:
            cubefile = os.path.join(dir, cubefile)

        self.cubefile = cubefile

        fits_cube = galsim.fits.readCube(cubefile)
        Nimages = len(fits_cube)
        im_shape = fits_cube[0].array.shape
        data = np.zeros((Nimages, im_shape[0], im_shape[1],))

        for i, im in enumerate(fits_cube):
            data[i,:,:] = im.array

        if isinstance(bandpasses, str):
            bandpass_file = bandpasses
            if '.pkl' in bandpass_file:
                with open(bandpass_file, 'rb') as f:
                    bandpasses = pickle.load(f)
            else:
                raise Exception('For now, only pickled lists of ' +\
                                'galsim.Bandpass objects are accepted')
        else:
            if not isinstance(bandpasses, list):
                raise Exception('For now, must pass bandpasses as either filename or list!')

        super(FitsDataCube, self).__init__(data=data, bandpasses=bandpasses)

        return

class SliceList(list):
    '''
    A list of Slice objects
    '''
    pass

class Slice(object):
    '''
    Base class of an abstract DataCube slice,
    corresponding to a source observation in a given
    bandpass
    '''
    def __init__(self, data, bandpass):
        self._data = data
        self.bandpass = bandpass

        self.red_limit = bandpass.red_limit
        self.blue_limit = bandpass.blue_limit
        self.central = (self.red_limit+self.blue_limit)/2. # can use eff lam
        self.dlamda = self.red_limit - self.blue_limit
        self.lambda_unit = bandpass.wave_type

        return

    @property
    def data(self):
        return self._data

    def plot(self, show=True, close=True, outfile=None, size=9, title=None,
             imshow_kwargs=None):

        if imshow_kwargs is None:
            im = plt.imshow(self._data)
        else:
            im = plt.imshow(self._data, **imshow_kwargs)

        plt.colorbar(im)

        if title is not None:
            plt.title(title)
        else:
            li, le = self.blue_limit, self.red_limit
            unit = self.lambda_unit
            plt.title(f'DataCube Slice; {li} {unit} < ' +\
                      f'lambda < {le} {unit}')

        plt.gcf().set_size_inches(size, size)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()
        elif close is True:
            plt.close()

        return

def setup_simple_bandpasses(lambda_blue, lambda_red, dlambda,
                            throughput=1., zp=30., unit='nm'):
    '''
    Setup list of bandpasses needed to instantiate a DataCube
    given the simple case of constant spectral resolution, throughput,
    and image zeropoints for all slices

    Useful for quick setup of tests and simulated datavectors

    lambda_blue: float
        Blue-end of datacube wavelength range
    lambda_red: float
        Rd-end of datacube wavelength range
    dlambda: float
        Constant wavelength range per slice
    throughput: float
        Throughput of filter of data slices
    unit: str
        The wavelength unit
    zeropoint: float
        Image zeropoint for all data slices
    '''

    li, lf = lambda_blue, lambda_red
    lambdas = [(l, l+dlambda) for l in np.arange(li, lf, dlambda)]

    bandpasses = []
    for l1, l2 in lambdas:
        bandpasses.append(gs.Bandpass(
            throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
            ))
    bandpasses = [gs.Bandpass(
        throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
        ) for l1,l2 in lambdas]

    return bandpasses

def get_datavector_types():
    return DATAVECTOR_TYPES

# NOTE: This is where you must register a new model
DATAVECTOR_TYPES = {
    'default': DataCube,
    'datacube': DataCube,
    }

def build_datavector(name, kwargs):
    '''
    name: str
        Name of datavector
    kwargs: dict
        Keyword args to pass to datavector constructor
    '''

    name = name.lower()

    if name in DATAVECTOR_TYPES.keys():
        # User-defined input construction
        datavector = DATAVECTOR_TYPES[name](**kwargs)
    else:
        raise ValueError(f'{name} is not a registered datavector!')

    return datavector


# Used for testing
def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'cube')
    utils.make_dir(outdir)

    li, le, dl = 500, 600, 1
    lambdas = np.arange(li, le, dl)

    throughput = '0.85'
    unit = 'nm'
    zp = 30
    bandpasses = []

    print('Building test bandpasses')
    for l1, l2 in zip(lambdas, lambdas+1):
        bandpasses.append(galsim.Bandpass(
            throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
            ))

    print('Testing bandpass helper func')
    bandpasses_alt = setup_simple_bandpasses(
        li, le, dl, throughput=throughput, unit=unit, zp=zp
        )
    assert bandpasses == bandpasses_alt

    Nspec = len(bandpasses)

    print('Building empty test data')
    shape = (100, 100, Nspec)
    data = np.zeros(shape)

    print('Building Slice object')
    n = 50 # slice num
    s = Slice(data[n,:,:], bandpasses[n])

    print('Testing slice plots')
    s.plot(show=False)

    print('Building SliceList object')
    sl = SliceList()
    sl.append(s)

    print('Building DataCube object from array')
    cube = DataCube(data=data, bandpasses=bandpasses)

    print('Testing DataCube truncation on slice centers')
    lambda_range = le - li
    blue_cut = li + 0.25*lambda_range + 0.5
    red_cut  = li + 0.75*lambda_range - 0.5
    truncated = cube.truncate(blue_cut, red_cut, trunc_type='center')
    nslices_cen = len(truncated.slices)
    print(f'----Truncation resulted in {nslices_cen} slices')

    print('Testing DataCube truncation on slice edges')
    truncated = cube.truncate(blue_cut, red_cut, trunc_type='edge')
    nslices_edg = len(truncated.slices)
    print(f'----Truncation resulted in {nslices_edg} slices')

    if nslices_edg != (nslices_cen-2):
        return 1

    print('Building DataCube from simulated fitscube file')
    mock_dir = os.path.join(utils.TEST_DIR,
                            'mocks',
                            'COSMOS')
    test_cubefile = os.path.join(mock_dir,
                                 'kl-mocks-COSMOS-001.fits')
    bandpass_file = os.path.join(mock_dir,
                                 'bandpass_list.pkl')
    if (os.path.exists(test_cubefile)) and (os.path.exists(bandpass_file)):
        print('Building from pickled bandpass list file')
        fits_cube = FitsDataCube(test_cubefile, bandpass_file)

        print('Building from bandpass list directly')
        with open(bandpass_file, 'rb') as f:
            bandpasses = pickle.load(f)
        fits_cube = FitsDataCube(test_cubefile, bandpasses)

        print('Making slice plot from DataCube')
        indx = fits_cube.Nspec // 2
        outfile = os.path.join(outdir, 'slice-plot.png')
        plot_kwargs = {
            'show': show,
            'outfile': outfile
        }
        fits_cube.plot_slice(indx, plot_kwargs)

        print('Making pixel spectrum plot from DataCube')
        box_size = fits_cube.slices[indx]._data.shape[0]
        i, j = box_size // 2, box_size // 2
        outfile = os.path.join(outdir, 'pixel-spec-plot.png')
        fits_cube.plot_pixel_spectrum(i, j, show=show, outfile=outfile)

        truth_file = os.path.join(mock_dir, 'truth.fits')
        if os.path.exists(truth_file):
            print('Loading truth catalog')
            truth = Table.read(truth_file)
            z = truth['zphot'][0]
            ha = 656.28 # nm
            ha_shift = (1+z) * ha

            print('Making pixel spectrum plot with true z')
            fits_cube.plot_pixel_spectrum(i, j, show=False, close=False)
            plt.axvline(
                ha_shift, lw=2, ls='--', c='k', label=f'(1+{z:.2})*H_alpha'
                )
            plt.legend()
            outfile = os.path.join(outdir, 'pix-spec-z.png')
            plt.savefig(outfile, bbox_inches='tight')
            if show is True:
                plt.show()

            print('Making aperture spectrum plot')
            radius = 4 # pixels
            offset = (0,0) # pixels
            fits_cube.plot_aperture_spectrum(radius, offset=offset,
                                             show=False, close=False)
            plt.axvline(
                ha_shift, lw=2, ls='--', c='k', label=f'(1+{z:.2})*H_alpha'
                )
            plt.legend()
            outfile = os.path.join(outdir, 'apt-spec-plot.png')
            plt.savefig(outfile, bbox_inches='tight')
            if show is True:
                plt.show()

    else:
        print('Files missing - skipping tests')

    print('Done!')

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test is True:
        print('Starting tests')
        rc = main(args)

        if rc == 0:
            print('All tests ran succesfully')
        else:
            print(f'Tests failed with return code of {rc}')
