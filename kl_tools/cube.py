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
    def stack(self):
        '''
        Each datavector must have a method that defines how to stack it
        into a single (Nx,Ny) image for basis function fitting
        '''
        pass

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

class DataSimulator():
    ''' Base class for a grism + image data.
    
        This class works as a "data simulator": it accepts `Pars` object as 
        input arguments and generates model vector. 

        Contains astronomical images of a source at various wavelength slices.
        Records dispersion relation given specific instrument capabilities.
    '''

    #def __init__(self, data=None, shape=None, bandpasses=None, pix_scale=None,
    #             pars=None):
        ''' Initialize either a filled DataCube from an existing numpy
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
            A dictionary that holds any additional metadata.
            In the case of grism, pars include
            {
                'diameter': diameter of telescope aperture in meter,
                'scale': pixel scale of SliceList (high-res),
                'Nx', 'Ny': grism image dimension,
                'pixel_scale': grism image pixel scale,
                'gain': detector gain,
                'exp_time': exposure time in units of second,
                'R_spec': grism resolution,
                'disp_angle': dispersion angle in units of radian,
                'offset': offset of the grism image,
                'psf_type': PSF model type, 
                'noise': noise properties
                {
                    'type': noise model type, 
                    'sky_level': sky level,
                    'read_noise': read noise,
                    'gain': detector gain, 
                }
            }
        '''
    def __init__(self, pars):
        ''' Initialize Grism data with Pars object
        
        The initialization will save the Pars object, set data dimension,
        and construct SliceList object
        Note that since we separate bandpass object into observation related
        parameters, the bandpasses during initialization is only a trivial one 
        '''
        # check and copy Pars
        utils.check_type(pars, 'pars', Pars)
        self.pars = pars

        
        # set model dimension
        self.Nx = self.pars.meta['model_dimension']['Nx']
        self.Ny = self.pars.meta['model_dimension']['Ny']
        self.scale = self.pars.meta['model_dimension']['scale']
        
        lam = self.pars.meta['model_dimension']['lambda_range']
        dlam = self.pars.meta['model_dimension']['lambda_res']
        blue_edge = np.arange(lam[0], lam[1], dlam)
        self.lambdas = np.array([(l, l+dlam) for l in blue_edge])
        self.lambda_unit = self.pars.meta['model_dimension']['lambda_unit']
        self.lambda_cen = np.mean(self.lambdas, axis=1)
        self.Nspec = self.lambdas.shape[0]
        # check convention
        self.shape = (self.Nspec, self.Ny, self.Nx)
        
        # set trivial modelcube and bandpasses
        self.bandpasses = [ gs.Bandpass(1.0, self.lambda_unit, 
            blue_limit=l[0], red_limit=l[1], 
            zeropoint=zp) for l in self.lambdas]
        # set theory model cube (trivial)
        self.modelcube = DataCube(shape=self.shape, 
            bandpasses=self.bandpasses, pix_scale=self.scale)
        
        # number of observations
        self.nobs = self.pars.meta['observations']['number_of_observations']

        return

    def evaluateTheoryModel(self, theta, rel_err=-1):
        ''' Evaluate the theory model cube based on theta and Pars
    
        theta: 1d array
            parameters being sampled

        '''
        # evaluate intensity profile, velocity distribution and SED
        self._setup_imap()
        imap_img = self.imap.render(self.pars.theta2pars(theta), 
            self.modelcube, {})
        self._setup_vmap(theta)
        X, Y = utils.build_map_grid(self.Nx, self.Ny, scale=self.scale)
        vmap_img = self.vmap('obs', X, Y, normalized=True)
        V = self.vmap('obs', X, Y)
        self._setup_SED(theta)

        # build Doppler-shifted datacube
        w_mesh = np.outer(self.lambda_cen, 1./(1.+vmap_img))
        w_mesh = w_mesh.reshape(self.lambda_cen.shape+vmap_img.shape)
        dc_array = self.sed.spectrum(w_mesh.flatten())
        dc_array = dc_array.reshape(w_mesh.shape)*imap_img[np.newaxis, :, :]
        self.modelcube.set_data(dc_array)

        return

    def getExposure(self):
        datavector = []
        covariance = []

        for obs in self.pars.meta['observations']:
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

    def _setup_vmap(self, theta):
        ''' Setup the observer-plane LoS velocity distribution
        
        theta: list of real
            sampled parameters
        '''
        # collect related pars
        vmap_pars = self.pars.theta2pars(theta)
        vmap_pars['v_unit'] = self.pars.meta['velocity']['v_unit']
        vmap_pars['r_unit'] = self.pars.meta['velocity']['r_unit']
        model_name = self.pars.meta['velocity']['model_name'] #default/centered
        self.vmap = VelocityMap(model_name, vmap_pars)
        return

    def _setup_imap(self):
        ''' Setup the observer-plane intensity profile
        
        '''
        # collect related pars
        imap_pars = {
            'flux': 1, # SED will deal with physical flux
            'hlr': self.pars.meta['intensity']['hlr'],
            'scale': self.pars.meta['model_dimension']['scale'],
        }
        imap_type = self.pars.meta['intensity']['type'] # default/basis/inclined_exp
        self.imap = intensity.build_intensity_map(imap_type, self.datacube, 
            imap_pars)
        return

    def _setup_SED(self, theta, rel_err=-1):
        ''' Initialize the SED interpolation

        theta: 1d array
            sampled parameters. Currently the parameters being sampled
            (g1, g2, theta_int, sini, v0, vcirc, rscale) will not affect
            sed. But in general, we may consider update the SED during 
            MCMC sampling.

        '''
        self.sed = Spectrum(self.pars.meta['sed'], rel_err=rel_err)
        return


    def stack(self):
        ''' Get the dispersed 2d grism spectrum

            The function will calculate offset for each Slice caused by grism
            dispersion, and stack them to get grism spectrum

            Note: slice object structure
                    self._data = data
                    self.bandpass = bandpass
                    self.red_limit = bandpass.red_limit
                    self.blue_limit = bandpass.blue_limit
                    self.dlamda = self.red_limit - self.blue_limit
                    self.lambda_unit = bandpass.wave_type

        '''
        # apply dispersion and stack
        _grism_list = list(map(self._disperse_mapper ,self.slices))
        grism_img = np.sum(_grism_list, axis=0)
        # add noise
        noise = self._getNoise()
        if noise is not None:
            grism_img_withNoise = grism_img.copy()
            grism_img_withNoise.addNoise(noise)
            noise_img = grism_img_withNoise - grism_img
            return grism_img_withNoise, noise_img
        else:
            return grism_img, None 


    def _disperse_mapper(self, model_slice):

        area = np.pi*(self.pars['diameter']/2.)**2

        # build InterpolatedImage from model slice
        
        _img = gs.Image(model_slice.data(), make_const=True, 
                    scale=self.pars['scale'])
        _gal = gs.InterpolatedImage(_img, scale=self.pars['scale'])
        # apply PSF
        # Note that galsim.Airy takes diameter in m and lambda in nm
        PSF = self._build_PSF_model(lam=model_slice.central, 
                                    diam=self.pars['diameter']/100.)
        if PSF is not None:
            _gal = gs.Convolve([_gal, PSF])
        # calculate dispersion shift, in units of pixels
        # Note: shift = (dx[pix], dy[pix])
        shift = self._disperse(model_slice.central)
        # draw slice image
        _grism = _gal.drawImage(nx=self.pars['Nx'], ny=self.pars['Ny'], 
                                scale=self.pars['pixel_scale'], 
                                method='auto',
                                area=area, exptime=self.pars['exp_time'],
                                gain=self.pars['gain'], offset=shift,
                                bandpass=model_slice.bandpass,
                                )
        # noise will be applied after the slices are summed
        return _grism


    def _disperse(self, lam):
        ''' Get the offset due to dispersion for a given Slice

            Inputs:
                lam:    center wavelength (nm) of the slice
            Outputs:
                disp:   dispersion vector (dx, dy) in units of pixel

            This is a wrapper for instrument dispersion relation
            TODO: consider a LookUpTable for more flexible dispersion?

            For a galaxy at real position (xcen, ycen), with dispersion angle theta, the wavelength lam gets dispersed to the new position:
            
                x = xcen + (lam * dx/dlam + offset) * cos(theta),
                y = ycen + (lam * dx/dlam + offset) * sin(theta)
            
            # Note that currently the Roman grism has R = 461*lambda[um],
            # the resulting displacement in number of pixels is
            #   shift = lambda * (2/deltalambda)
            #         = lambda * (2/1um*R) = lambda / 500nm * R
            # where R is the spectral resolution at 1000nm
            # so the dispersion delta(lambda) per pixel is fixed, and
            # the displacement as a function of wavelength is a linear relation
            # But in more general cases, the dispersion relation is non-linear,
            # and interpolation is needed.
        '''
        dxdlam = self.pars['R_spec']/500. # 1/nm
        disp_direction = np.array([np.cos(self.pars['disp_ang']), 
                                   np.sin(self.pars['disp_ang'])])
        offset = self.pars['offset']
        return (lam * dxdlam + offset)*disp_direction

    def _build_PSF_model(self, **kwargs):
        psf_type = self.pars['psf_type'].lower()
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


    def _getNoise(self):
        #'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5}
        noise_pars = self.pars['noise']
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
        elif noise_type == 'none':
            noise = None
        else:
            raise ValueError(f'{noise_type} not implemented yet!')
        return noise

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


    def write(self, outfile):
        d = os.path.dirname(outfile)

        utils.make_dir(d)

        im_list = []
        for s in self.slices:
            im_list.append(s._data)

        gs.fits.writeCube(im_list, outfile)

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
