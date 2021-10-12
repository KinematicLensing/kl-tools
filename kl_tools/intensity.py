import numpy as np
import os
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians

import utils
import basis
import likelihood
from transformation import TransformableImage

import pudb

'''
This file contains a mix of IntensityMap classes for explicit definitions
(e.g. an inclined exponential) and IntensityMapFitter + Basis classes for
fitting a a chosen set of arbitrary basis functions to the stacked datacube
image
'''

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class IntensityMap(object):
    '''
    If needed, could have this be the return class of
    IntensityMapFitter.fit()
    '''

    def __init__(self, name, nx, ny):
        '''
        name: str
            Name of intensity map type
        nx: int
            Size of image on x-axis
        ny: int
            Size of image on y-axis
        '''

        if not isinstance(name, str):
            raise TypeError('IntensityMap name must be a str!')
        self.name = name

        for n in [nx, ny]:
            if not isinstance(n, int):
                raise TypeError('IntensityMap image size params must be ints!')
        self.nx = nx
        self.ny = ny

        self.image = None

        return

    def render(self, theta_pars, pars, redo=False):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map
        redo: bool
            Set to remake rendered image regardless of whether
            it is already internally stored

        return: np.ndarray
            The rendered intensity map
        '''

        # only render if it has not been computed yet, or if
        # explicitly asked
        if self.image is not None:
            if redo is False:
                return self.image

        else:
            return self._render(theta_pars, pars)

    @abstractmethod
    def _render(self, theta_pars, pars):
        '''
        Each subclass should define how to render

        Most will need theta and pars, but not all
        '''
        pass

    def plot(self, show=True, close=True, outfile=None, size=(7,7)):
        if self.image is None:
            raise Exception('Must render profile first! This can be ' +\
                            'done by calling render() with relevant params')

        ax = plt.gca()
        im = ax.imshow(self.image, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title('BasisIntensityMap.render() call')

        plt.gcf().set_size_inches(size)
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

class InclinedExponential(IntensityMap):
    '''
    This class is mostly for testing purposes. Can give the
    true flux, hlr for a simulated datacube to ensure perfect
    intensity map modeling for validation tests.

    We explicitly use an exponential over a general InclinedSersic
    as it is far more efficient to render, and is only used for
    testing anyway
    '''

    def __init__(self, datacube, flux=None, hlr=None):
        '''
        datacube: DataCube
            While this implementation will not use the datacube
            image explicitly (other than shape info), most will
        flux: float
            Object flux
        hlr: float
            Object half-light radius (in pixels)
        '''

        nx, ny = datacube.Nx, datacube.Ny
        super(InclinedExponential, self).__init__('inclined_exp', nx, ny)

        pars = {'flux': flux, 'hlr': hlr}
        for name, val in pars.items():
            if val is None:
                pars[name] = 1.
            else:
                if not isinstance(val, (float, int)):
                    raise TypeError(f'{name} must be a float or int!')

        self.flux = pars['flux']
        self.hlr = pars['hlr']

        return

    def _render(self, theta_pars, pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map
        redo: bool
            Set to remake rendered image regardless of whether
            it is already internally stored

        return: np.ndarray
            The rendered intensity map
        '''

        # A = theta_pars['A']
        inc = Angle(np.arcsin(theta_pars['sini']), radians)

        gal = gs.InclinedExponential(
            inc, flux=self.flux, half_light_radius=self.hlr
        )

        # Only add knots if a psf is provided
        if 'psf' in pars:
            if 'knots' in pars:
                knot_pars = pars['knots']
                knots = gs.RandomKnots(**knot_pars)
                gal = gal + knots

        rot_angle = Angle(theta_pars['theta_int'], radians)
        gal = gal.rotate(rot_angle)

        # TODO: still don't understand why this sometimes randomly fails
        try:
            g1 = theta_pars['g1']
            g2 = theta_pars['g2']
            gal = gal.shear(g1=g1, g2=g2)
        except Exception as e:
            print('imap generation failed!')
            print(f'Shear values used: g=({g1}, {g2})')
            raise e

        # TODO: could generalize in future, but for now assume
        #       a constant PSF for exposures
        if 'psf' in pars:
            psf = pars['psf']
            gal = gs.Convolve([gal, psf])

        self.image = gal.drawImage(nx=self.nx, ny=self.ny).array

        return self.image

class BasisIntensityMap(IntensityMap):
    '''
    This is a catch-all class for Intensity Maps made by
    fitting arbitrary basis functions to the stacked
    datacube image

    TODO: Need a better name!
    '''

    def __init__(self, datacube, basis_type='default', basis_kwargs=None,
                 fit_now=True):
        '''
        basis_type: str
            Name of basis type to use
        datacube: DataCube
            A truncated datacube whose stacked slices will be fit to
        basis_kwargs: dict
            Dictionary of kwargs needed to construct basis
        fit_now: bool
            Turn on to fit the basis to the datacube now. This should be
            turned off for subclasses that require the fitting to be done
            later, such as when the image transforms are defined
        '''

        nx, ny = datacube.Nx, datacube.Ny
        super(BasisIntensityMap, self).__init__('basis', nx, ny)

        self.fitter = IntensityMapFitter(
            basis_type, self.nx, self.ny, basis_kwargs=basis_kwargs
            )

        if fit_now is True:
            self._fit_to_datacube(datacube)
        else:
            self.image = None

        return

    def _fit_to_datacube(self, datacube):
        self.image = self.fitter.fit(datacube)

        return

    def get_basis(self):
        return self.fitter.basis

    def render(self, theta_pars=None, pars=None):
        '''
        These args exist to allow for a uniform API for
        intensity maps
        '''

        super(BasisIntensityMap, self).render(theta_pars, pars)

        return

    def _render(self, theta_pars, pars):
        return self.image

class TransformedBasis(BasisIntensityMap, TransformableImage):
    '''
    This class fits basis functions to the stacked datacube as in
    BasisIntensityMap, but instead uses the chosen basis tranformed
    into the obs plane from the disk plane
    '''

    def __init__(self, datacube, basis_type='default', basis_kwargs=None):
        '''
        Setup the basis, but can't setup the transforms until we have
        access to the sampled transform parameters
        '''

        super(TransformedBasis, self).__init__(
            datacube,
            basis_type=basis_type,
            basis_kwargs=basis_kwargs,
            fit_now=False
            )

        # we can't render the image until the transformation params are
        # defined, so we have to store the datacube for this subclass
        self.datacube = datacube

        # will be set once the transforms are defined
        self.transform_pars = None
        self._planes = None
        self.obs2source = None
        self.source2gal = None
        self.gal2disk = None

        return

    def render(self, theta_pars, pars):

        # Now we can initialize the transforms
        super(TransformableImage, self).__init__(theta_pars)

        super(TransformedBasis, self).render(theta_pars, pars)

        return

    def _render(self, theta_pars, pars):
        # now we can finally fit the transformed basis funcs
        # to the stored datacube
        self._fit_to_datacube()

        super(TransformedBasis, self)._render()

        return

    def _fit_to_datacube(self):
        self.image = self.fitter.fit(self.datacube)

        return

    # def _setup_data_im(self, datacube):
    #     self.data_image = np.sum(datacube._data, axis=2)

    #     return

def get_intensity_types():
    return INTENSITY_TYPES

# NOTE: This is where you must register a new model
INTENSITY_TYPES = {
    'default': TransformedBasis,
    'basis': BasisIntensityMap,
    'inclined_exp': InclinedExponential,
    'transformed_basis': TransformedBasis,
    }

def build_intensity_map(name, datacube, kwargs):
    '''
    name: str
        Name of intensity map type
    datacube: DataCube
        The datacube whose stacked image the intensity map
        will represent
    kwargs: dict
        Keyword args to pass to intensity constructor
    '''

    name = name.lower()

    if name in INTENSITY_TYPES.keys():
        # User-defined input construction
        intensity = INTENSITY_TYPES[name](datacube, **kwargs)
    else:
        raise ValueError(f'{name} is not a registered intensity!')

    return intensity

class IntensityMapFitter(object):
    '''
    This base class represents an intensity map defined
    by some set of basis functions {phi_i}.
    '''
    def __init__(self, basis_type, nx, ny, basis_kwargs=None):
        '''
        basis_type: str
            The name of the basis_type type used
        nx: int
            The number of pixels in the x-axis
        ny: int
            The number of pixels in the y-ayis
        basis_kwargs: dict
            Keyword args needed to build given basis type
        '''

        for name, n in {'nx':nx, 'ny':ny}.items():
            if name in basis_kwargs:
                if n != basis_kwargs[name]:
                    raise ValueError(f'{name} must be consistent if ' +\
                                       'also passed in basis_kwargs!')

        self.basis_type = basis_type
        self.nx = nx
        self.ny = ny

        self.grid = utils.build_map_grid(nx, ny)

        self._initialize_basis(basis_kwargs)

        return

    def _initialize_basis(self, basis_kwargs):
        if basis_kwargs is not None:
            basis_kwargs['nx'] = self.nx
            basis_kwargs['ny'] = self.ny
        else:
            basis_kwargs = {
                'nx': self.nx,
                'ny': self.ny,
                }

        self.basis = basis.build_basis(self.basis_type, basis_kwargs)
        self.Nbasis = self.basis.N
        self.mle_coefficients = np.zeros(self.Nbasis)

        self._initialize_pseudo_inv()

        return

    # TODO: Add @njit when ready
    def _initialize_pseudo_inv(self):
        '''
        Setup Moore-Penrose pseudo inverse given basis

        data: np.ndarray
            The sum of datacube slices corresponding to the emission
            line. This is what we will fit the intensity map to
        basis: list
            A list of functions that evaluate the ith basis function
            at the given location(s)
        '''


        Ndata = self.nx * self.ny
        Nbasis = self.Nbasis

        # build image grid vectors
        # X, Y = utils.build_map_grid(nx, ny)
        X, Y = self.grid
        x = X.reshape(Ndata)
        y = Y.reshape(Ndata)

        # the design matrix for a given basis and datacube
        M = np.zeros((Ndata, Nbasis))
        for n in range(Nbasis):
            func, func_args = self.basis.get_basis_func(n)
            args = [x, y, *func_args]
            M[:,n] = func(*args)

        # now compute the pseudo-inverse:
        self.pseudo_inv = np.linalg.pinv(M)

        # for n, b in enumerate(basis):
        #     X[:, n] = b(data_vec)

        # for n in range(Ndata):
        #     for m in range(Ndata):
        #         self.pseudo_inv[n,m] = self.basis

        return

    def fit(self, datacube, cov=None):
        '''
        Fit MLE of the intensity map for a given set of datacube
        slices

        NOTE: This assumes that the datacube is truncated to be only be
              around the relevant emission line region, though not
              necessarily with the continuum emission subtracted

        cov: np.ndarray
            The covariance matrix of the datacube images.
            NOTE: If the covariance matrix is of the form sigma*I for a
                  constant sigma, it does not contribute to the MLE. Thus
                  you should only pass cov if it is non-trivial
        '''

        nx, ny = self.nx, self.ny
        if (datacube.Nx, datacube.Ny) != (nx, ny):
            raise ValueError('DataCube must have same dimensions ' +\
                             'as intensity map!')

        # We will fit to the sum of all slices
        data = np.sum(datacube._data, axis=2).reshape(nx*ny)

        # Find MLE basis coefficients
        mle_coeff = self._fit_mle_coeff(data, cov=cov)

        assert len(mle_coeff) == self.Nbasis
        self.mle_coefficients = mle_coeff

        # Now create MLE intensity map
        mle_im = self.basis.render_im(mle_coeff)

        assert mle_im.shape == (nx, ny)
        self.mle_im = mle_im

        return mle_im

    def _fit_mle_coeff(self, data, cov=None):
        '''
        data: np.array
            The (nx*ny) data vector
        '''

        if cov is None:
            # The solution is simply the Moore-Penrose pseudo inverse
            # acting on the data vector
            mle_coeff = self.pseudo_inv.dot(data)

        else:
            # If cov is diagonal but not constant sigma, then see
            # SPECTRO-PERFECTIONISM (bolton, schlegel et al. 2009)
            raise NotImplementedError(
                'The MLE for intensity maps with non-trivial ' +\
                'covariance matrices is not yet implemented!'
                )

        return mle_coeff

    def plot_mle_fit(self, datacube, show=True, close=True, outfile=None,
                     size=(9,9)):
        '''
        datacube: DataCube
            Datacube that MLE fit was done on
        '''

        # fit was done on stacked datacube
        data = np.sum(datacube._data, axis=2)
        mle = self.mle_im

        fig, axes = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=True, figsize=size
            )

        image = [data, mle, data-mle, 100.*(data-mle)/mle]
        titles = ['Data', 'MLE', 'Residual', '% Residual']

        for i in range(len(image)):
            ax = axes[i//2, i%2]
            if '%' in titles[i]:
                vmin, vmax = -100., 100.
            else:
                vmin, vmax = None, None
            im = ax.imshow(
                image[i], origin='lower', vmin=vmin, vmax=vmax
                )
            ax.set_title(titles[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.suptitle(f'MLE comparison for {self.Nbasis} basis functions')
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

def main(args):
    '''
    For now, just used for testing the classes
    '''

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'intensity')
    utils.make_dir(outdir)

    print('Creating IntensityMapFitter w/ shapelet basis')
    nmax = 10
    nx, ny = 30,30
    basis_kwargs = {'Nmax': nmax}
    fitter = IntensityMapFitter(
        'shapelets', nx, ny, basis_kwargs=basis_kwargs
        )

    print('Setting up test datacube and true Halpha image')
    true_pars, pars = likelihood.setup_test_pars(nx, ny)

    # add some knot features
    knot_frac = 0.5 # not really correct, but close enough for tests
    pars['psf'] = gs.Gaussian(fwhm=3) # pixels w/o pix_scale defined
    pars['knots'] = {
        'npoints': 25,
        'half_light_radius': pars['true_hlr'],
        'flux': knot_frac * pars['true_flux'],
    }

    li, le, dl = 655.8, 656.8, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nspec = len(lambdas)
    shape = (nx, ny, Nspec)
    datacube, sed, vmap, true_im = likelihood.setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    print('test')

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)

    outfile = os.path.join(outdir, 'datacube-slices.png')
    print(f'Saving example datacube slice images to {outfile}')
    # if Nspec < 10:
    sqrt = int(np.ceil(np.sqrt(Nspec)))
    slice_indices = range(Nspec)

    k = 1
    for i in slice_indices:
        plt.subplot(sqrt, sqrt, k)
        plt.imshow(datacube.slices[i]._data, origin='lower')
        plt.colorbar()
        l, r = lambdas[i]
        plt.title(f'lambda=({l:.1f}, {r:.1f})')
        k += 1
    plt.gcf().set_size_inches(12,12)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    print('Fitting simulated datacube with shapelet basis')
    start = time.time()
    mle_im = fitter.fit(datacube)
    t = time.time() - start
    print(f'Total fit time took {1000*t:.2f} ms for {fitter.Nbasis} basis funcs')

    outfile = os.path.join(outdir, 'compare-mle-to-data.png')
    print(f'Plotting MLE fit compared to stacked data to {outfile}')
    fitter.plot_mle_fit(datacube, outfile=outfile, show=show)

    print('Initializing a BasisIntensityMap for shapelets')
    imap = BasisIntensityMap(
        datacube, basis_type='shapelets', basis_kwargs={'Nmax':nmax}
        )
    imap.render()

    outfile = os.path.join(outdir, 'shapelet-imap-render.png')
    print(f'Saving render for shapelet basis to {outfile}')
    imap.plot(outfile=outfile, show=show)

    print('Initializing a TransformedBasis with shapelets')
    basis_kwargs = {'Nmax':nmax}
    imap = TransformedBasis(
        datacube, basis_type='shapelets', basis_kwargs=basis_kwargs
        )

    print('Initializing a TransformedBasis with the builder')
    imap_pars = {
        'basis_kwargs': basis_kwargs
    }
    imap = build_intensity_map('transformed_basis', datacube, imap_pars)

    true_pars = {
        'g1': 0.05,
        'g2': -0.025,
        'theta_int': np.pi / 3,
        'sini': 0.8,
        'v0': 10.,
        'vcirc': 200,
        'rscale': 5,
    }
    pars = None
    print('Rendering transformed shapelet fit')
    pudb.set_trace()
    imap.render(true_pars, pars)

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