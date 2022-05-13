# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys
import pickle
import schwimmbad
import mpi4py
from schwimmbad import MPIPool
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import zeus

import utils
from mcmc import KLensZeusRunner, KLensEmceeRunner
import priors
import cube
import likelihood
from parameters import Pars
from likelihood import LogPosterior_Roman
from velocity import VelocityMap

import pudb

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--file', type=str, default='',
                    help='Name of the parameters YAML file')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')

def main(args, pool):

    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    show = args.show
    pars_file = args.file

    outdir = os.path.join(
        utils.TEST_DIR, 'test-mcmc-run', run_name
        )
    utils.make_dir(outdir)

    ### Initialize Log-Posterior object
    ### ===============================
    # initialize parameters
    pars_from_yaml = Pars(yaml_file=pars_file)
    truth = pars_from_yaml.dynamic_fid
    true_pars = pars_from_yaml.theta2pars(truth)
    # initialize likelihood
    # the data vector will be generated from `fid_pars`
    logprob = LogPosterior_Roman(pars_from_yaml, 
        fid_pars=pars_from_yaml.dynamic_fid)

    ### Print fiducial data for debug
    ### =============================
    ds = logprob.log_likelihood.dsim
    # >>> intensity profile
    outfile = os.path.join(outdir, 'true-im.png')
    print(f'Saving true intensity profile in obs plane to {outfile}')
    plt.imshow(ds.imap_img, origin='lower')
    plt.colorbar()
    plt.title('True intensity profile in obs plane')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()
    # >>> velocity map
    outfile = os.path.join(outdir, 'vmap.png')
    print(f'Saving true vamp in obs plane to {outfile}')
    plt.imshow(ds.vmap_img*2.99e5, origin='lower')
    plt.colorbar(label='v')
    plt.title('True velocity map in obs plane')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()
    # >>> 3D theory model
    #outfile = os.path.join(outdir, 'datacube.fits')
    #print(f'Saving test datacube to {outfile}')
    #datacube.write(outfile)
    # >>> Observed data
    outfile = os.path.join(outdir, 'observed_datevectors.png')
    print(f'Saving observed images (a.k.a. data vectors) to {outfile}')

    fig, axes = plt.subplots(1, ds.Nobs, figsize=(3*ds.Nobs+1,3))
    for i in range(ds.Nobs):
        ax = axes[i]
        cb = ax.imshow(logprob.log_likelihood.datavector[i], 
            origin='lower', cmap='hot')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cb, cax=cax)
        ax.set(xticks=[], yticks=[])
    #plt.gcf().set_size_inches(12,12)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = logprob.ndims
    nwalkers = 2*ndims

    if sampler == 'zeus':
        print('Setting up KLensZeusRunner')

        runner = KLensZeusRunner(
            nwalkers, ndims, logprob, None, 
            logprob.log_likelihood.parameters
            )

    elif sampler == 'emcee':
        print('Setting up KLensEmceeRunner')

        runner = KLensEmceeRunner(
            nwalkers, ndims, logprob, None,
            logprob.log_likelihood.parameters
            )
    #-----------------------------------------------------------------
    # Run chains
    print('Starting mcmc run')
    # try:
    runner.run(nsteps, pool)
    # except Exception as e:
    #     g1 = runner.start[:,0]
    #     g2 = runner.start[:,1]
    #     print('Starting ball for (g1, g2):')
    #     print(f'g1: {g1}')
    #     print(f'g2: {g2}')
    #     val = np.sqrt(g1**2+g2**2)
    #     print(f' |g1+ig2| = {val}')
    #     raise e
    runner.burn_in = nsteps // 2

    #-----------------------------------------------------------------
    # Sanity check and post-processing
    print('Ending mcmc run')
    # >>> sampler
    if (sampler == 'zeus') and ((ncores > 1) or (mpi == True)):
        # The sampler isn't pickleable for some reason in this scenario
        # Save whole chain
        outfile = os.path.join(outdir, 'test-mcmc-chain.pkl')
        chain = runner.sampler.get_chain(flat=True)
        print(f'pickling chain to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(chain, f)
    else:
        outfile = os.path.join(outdir, 'test-mcmc-sampler.pkl')
        print(f'Pickling sampler to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner.sampler, f)

        outfile = os.path.join(outdir, 'test-mcmc-runner.pkl')
        print(f'Pickling runner to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner, f)
    # >>> fiducial params
    outfile = os.path.join(outdir, 'test-mcmc-truth.pkl')
    print(f'Pickling truth to {outfile}')
    with open(outfile, 'wb') as f:
        pickle.dump(pars_from_yaml.dynamic_fid, f)
    
    # >>> chain traces
    outfile = os.path.join(outdir, 'chains.png')
    print(f'Saving chain plots to {outfile}')
    #reference = pars.pars2theta(pars_from_yaml.dynamic_fid)
    reference = pars_from_yaml.sampled_fid
    runner.plot_chains(
        outfile=outfile, reference=reference, show=show
        )
    
    # >>> posterior
    outfile = os.path.join(outdir, 'corner-truth.png')
    print(f'Saving corner plot to {outfile}')
    title = 'Reference lines are param truth values'
    runner.plot_corner(
        outfile=outfile, reference=reference, title=title, show=show
        )
    
    # >>> Sanity check on MAP
    runner.compute_MAP()
    map_medians = runner.MAP_medians
    map_medians_pars = pars_from_yaml.sampled_theta_2_dynamic_pars(map_medians)
    print('(median) MAP values:')
    for name, val in map_medians_pars.items():
        print(f'{name}: {val:.4f}')
    #outfile = os.path.join(outdir, 'compare-data-to-map.png')
    #print(f'Plotting MAP comparison to data in {outfile}')
    #runner.compare_MAP_to_data(outfile=outfile, show=show)

    #outfile = os.path.join(outdir, 'compare-vmap-to-map.png')
    #print(f'Plotting MAP comparison to velocity map in {outfile}')
    #vmap_pars = true_pars
    #vmap_pars['r_unit'] = meta_pars['r_unit']
    #vmap_pars['v_unit'] = meta_pars['v_unit']
    #vmap_true = VelocityMap('default', vmap_pars)
    #runner.compare_MAP_to_truth(vmap_true, outfile=outfile, show=show)

    outfile = os.path.join(outdir, 'corner-map.png')
    print(f'Saving corner plot compare to MAP in {outfile}')
    title = 'Reference lines are param MAP values'
    runner.plot_corner(
        outfile=outfile, reference=runner.MAP_medians, title=title, show=show
        )
    
    # >>> priors & likelihoods
    if sampler == 'emcee':
        blobs = runner.sampler.blobs
    elif sampler == 'zeus':
        blobs = runner.sampler.get_blobs()

    outfile = os.path.join(outdir, 'chain-probabilities.pkl')
    print(f'Saving prior & likelihood values to {outfile}')
    data = {
        'prior': blobs[:,:,0],
        'likelihood': blobs[:,:,1]
    }
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

    # >>> likelihood and prior trace of a random walker
    outfile = os.path.join(outdir, 'chain-probabilities.png')
    print(f'Saving prior & likelihood value plot to {outfile}')
    indx = np.random.randint(0, high=nwalkers)
    prior = blobs[:,indx,0]
    like = blobs[:,indx,1]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    plt.subplot(131)
    plt.plot(prior, label='prior', c='tab:blue')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()
    plt.subplot(132)
    plt.plot(like, label='likelihood', c='tab:orange')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()
    plt.subplot(133)
    plt.plot(prior, label='prior', c='tab:blue')
    plt.plot(like, label='likelihood', c='tab:orange')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()
    return 0

if __name__ == '__main__':
    '''
    To run with serial pool (`SerialPool`):
    $ python test_mcmc_roman.py 10000 -sampler=emcee -run_name=test_mcmc_roman --file=../yaml/params_test.yaml

    To run with `multiprocessing` utilities (`MultiPool`):
    $ python test_mcmc_roman.py 10000 -sampler=emcee -run_name=test_mcmc_roman --file=../yaml/params_test.yaml -ncores=28

    To run with mpi pool (`MPIPool`):
    $ mpiexec -n 16 --mca btl tcp,self python test_mcmc_roman.py 10000 -sampler=emcee -run_name=test_mcmc_roman --file=../yaml/params_test.yaml --mpi
    '''
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    if isinstance(pool, MPIPool):
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    print('Starting tests')
    rc = main(args, pool)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
