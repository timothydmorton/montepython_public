import numpy as np
import emcee

import sampler

import pdb

default_nwalkers = 20

def lnprob(theta, cosmo, data):
    """ln posterior to be used by emcee

    theta is parameter vector

    cosmo is Class() instance
    """
    parameter_names = data.get_mcmc_parameters(['varying'])

    # Update arguments according to parameter vector
    pars = {k:v for k,v in zip(parameter_names, theta)}
    data.cosmo_arguments.update(pars)
    cosmo.set(pars)

    logl = 0
    pdb.set_trace()
    for likelihood in data.lkl.itervalues():
        logl += likelihood.loglkl(cosmo, data)

    return logl

def run(cosmo, data, command_line):

    nwalkers = default_nwalkers

    parameter_names = data.get_mcmc_parameters(['varying'])
    ndim = len(parameter_names)

    # pdb.set_trace()
    # Initialize walkers
    fiducial_values = [data.mcmc_parameters[p]['initial'][0]*\
                        data.mcmc_parameters[p]['scale'] for p in parameter_names]
    p0 = [(1 + np.random.randn(ndim)*0.01)*fiducial_values for i in range(nwalkers)]

    s = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[cosmo, data])

    nsteps = command_line.N
    for i, result in enumerate(s.sample(p0, iterations=nsteps)):
        if (i+1) % 1 == 0:
            print("{0:5.1%}".format(float(i) / nsteps))

    print('Acceptance fraction: {0}'.format(s.acceptance_fraction))
    
    np.save('emcee_chain.npy', s.flatchain)
