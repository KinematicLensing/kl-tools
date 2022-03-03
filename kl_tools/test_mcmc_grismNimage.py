import os
import sys
sys.path.append('/Users/jiachuanxu/Workspace/KL_measurement/kl-tools/kl_tools')
import numpy as np
from astropy.units import Unit
import matplotlib.pyplot as plt
import galsim
import galsim.roman as roman

import basis
import cube
import intensity
import likelihood
import mcmc
import parameters
import priors
import utils
import velocity
from spectrum import Spectrum
from MockObs import Mock
import emcee
from numpy.random import default_rng
import time
from schwimmbad import MPIPool
os.environ["OMP_NUM_THREADS"] = "1"

theta_pars_keys = ['g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale']

def log_prior(theta, pars):
    theta_pars = {}
    for i,key in enumerate(theta_pars_keys):
        theta_pars[key] = theta[i]
    logprior = 0
    for i in theta_pars.keys():
        pobj = pars['priors'][i]
        logprior += pobj(theta_pars[i], log=True)
    return logprior

def log_likelihood(theta, pars, dvs, covmats):
    
    theta_pars = {}
    for i,key in enumerate(theta_pars_keys):
        theta_pars[key] = theta[i]
    m = Mock(theta_pars, pars)
    obs_data, covmats = m.getExposure()
    # calculate chi2
    diff = [dv.array-mv.array for dv,mv in zip(dvs, obs_data)]
    chi2 = 0
    dof = 0
    for _diff, cov in zip(diff, covmats):
        noise_average = np.std(cov.array)
        chi2 += np.sum( (_diff/noise_average)**2 )
        dof += _diff.flatten().shape[0]
    return -0.5*chi2 + log_prior(theta, pars)


theta_pars = {'g1':0.0, 'g2':0.0, 'theta_int': 0., 'sini':0.5,'v0':0,'vcirc':200,'rscale':0.5}
pars = {
    'priors':{
        'g1': priors.GaussPrior(0., 0.1),
        'g2': priors.GaussPrior(0., 0.1),
        'theta_int': priors.UniformPrior(-np.pi/2., np.pi/2.),
        'sini': priors.UniformPrior(0., 1.),
        'v0': priors.GaussPrior(0, 20),
        'vcirc': priors.GaussPrior(200, 10),
        'rscale': priors.UniformPrior(0, 10),
    },
    'meta':{
        'Nx':64,
        'Ny':64,
        'scale':0.1,# arcsec
        'lambda_range':[1200, 1400],
        'lambda_res': 1,# nm
    },
    'intensity':{
        'type':'inclined_exp',
        'hlr':0.5,# arcsec
    },
    'velocity':{
        'v_unit':Unit('km / s'),
        'r_unit':Unit('arcsec'),
    },
    'SED':{
        'template':'../data/Simulation/GSB2.spec',
        'wave_type': 'Ang',
        'flux_type': 'flambda',
        'z': 1.0,
        'wave_range': (500, 3000), # nm
        # obs-frame continuum normalization (nm, erg/s/cm2/nm)
        'obs_cont_norm': (1400, 1e-17),
        # a dict of line names and obs-frame flux values (erg/s/cm2)
        'lines': {'Halpha': 1e-15, 'OII':[1e-15, 1.2e-15], 'OIII':[1e-15, 1.2e-15]},
        # intrinsic linewidth in nm
        'line_sigma_int': {'Halpha': 4, 'OII':[2,2], 'OIII':[3,3]},
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
         'bandpass':'../data/Bandpass/Roman/WFI_Grism_1stOrder.dat',
         'Nx': 64,# number of pixels
         'Ny': 64,
         'pixel_scale': 0.11,# arcsec
         'R_spec':461,# at 1 micron
         'offset':-1210.22,# pix
         # can be 'none'/'airy'/'moffat'
         # 'kolmogorov'/'vonkarman'/'opticalpsf' to be implemented.
         'psf_type':'airy',
         # pass the needed params to build PSF model here
         # in case of airy, we don't need any params
         'psf_kwargs':{'fwhm':0.13},# arcsec
         'disp_ang':0.,# radian
         'diameter':240,# cm
         'exp_time':600.,# seconds
         'gain':1.,
         'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5},
         },
        # Roman WFI/GRISM observation, roll angle 2
        {'inst_name':'Roman/WFI',
         'type':'grism',
         'bandpass':'../data/Bandpass/Roman/WFI_Grism_1stOrder.dat',
         'Nx': 64,# number of pixels
         'Ny': 64,
         'pixel_scale': 0.11,# arcsec
         'R_spec':461,# at 1 micron
         'offset':-1210.22,
         'psf_type':'airy',
         'psf_kwargs':{'fwhm':0.13},# arcsec
         'disp_ang':np.pi/2.,# radian
         'diameter':240,# cm
         'exp_time':600.,# seconds
         'gain':1.,
         'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5},
         },
        # Roman WFI/Photometry image observation, H band
        {'inst_name':'Roman/WFI',
         'type':'photometry',
         'bandpass':'../data/Bandpass/Roman/WFI_F129.dat',
         'Nx': 32,# number of pixels
         'Ny': 32,
         'pixel_scale': 0.11,# arcsec
         'psf_type':'airy',
         'psf_kwargs':{'fwhm':0.13},# arcsec
         'diameter':240,# cm
         'exp_time':600.,# seconds
         'gain':1.,
         'noise':{'type':'ccd','sky_level':0.65*1.2,'read_noise':8.5},
         },
    ],
    'use_numba': False,
}

start = time.time()

m = Mock(theta_pars, pars)
obs_data, covmats = m.getExposure()


theta_pars = {'g1':0.0, 'g2':0.0, 'theta_int': 0., 'sini':0.5,'v0':0,'vcirc':200,'rscale':0.5}
theta_test = [0., 0., 0., 0.5, 0., 200., 0.5]
loglike = log_likelihood(theta_test, pars, obs_data, covmats)
logprior = log_prior(theta_test, pars)
print(loglike, logprior)

end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds [1 evaluation]".format(serial_time))

theta_test = [0., 0., 0., 0.5, 0., 200., 0.5]
balls = [0.1, 0.1, 0.1, 0.05, 5, 10, 0.05, ]

ndim, nwalkers = 7, 35
p0 = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    for j in range(ndim):
        p0[i,j] = theta_test[j] + np.random.normal(0.0, balls[j])
print("start sampling")

pool = MPIPool()
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=[pars, obs_data, covmats], pool=pool)
#state = sampler.run_mcmc(p0, 100, progress=True)
#sampler.reset()
start = time.time()

f = open("test_chain_grism.dat", 'w')
if f:
    print("Writing output file...")
else:
    print("Fail to open %s"%filename)
#write header here
f.write('# ' + '    '.join(theta_pars_keys)+" log_like\n")

for (p, loglike, state) in sampler.sample(p0,iterations=1000):
    for row,logl in zip(p,loglike):
        p_text = '  '.join(str(r) for r in row)
        f.write('%s %e\n' % (p_text,logl))
    f.flush()
f.close()

#sampler.run_mcmc(state, 10000, progress=True)
end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds".format(serial_time))
pool.close()
