import numpy as np
import emcee

import sampler

default_nwalkers = 200

def lnprob(theta, cosmo, data):
    """ln posterior to be used by emcee

    theta is parameter vector

    cosmo is Class() instance
    """
    parameter_names = data.get_mcmc_parameters(['varying'])

    # Update arguments according to parameter vector
    for k,v in zip(parameter_names, theta):
        data.cosmo_arguments[k] = v

    logl = 0
    for likelihood in data.lkl.itervalues():
        logl += likelihood.loglkl(cosmo, data)

    return logl

def run(cosmo, data, command_line):

    nwalkers = default_nwalkers

    parameter_names = data.get_mcmc_parameters(['varying'])
    ndim = len(parameter_names)

    # Initialize walkers
    fiducial_values = [data.mcmc_parameters[p]['initial'][0] for p in parameter_names]
    p0 = [(1 + np.random.randn(ndim)*0.05)*fiducial_values for i in range(nwalkers)]

    s = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[cosmo, data])
    s.run_mcmc(p0, command_line.N)

    print('Acceptance fraction: {0}'.format(s.acceptance_fraction))
    
    np.save('emcee_chain.npy', s.flatchain)
