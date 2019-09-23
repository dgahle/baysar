
'''
demo of the building a posteror for fitting two spectra

has no prior
'''''

import numpy as np
from numpy import random

import scipy.io as sio
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import os, sys, io

import time as clock

sys.path.append(os.path.expanduser('~/baysar'))

from tulasa import general

# from tulasa.plotting_functions import plot_guess, mini_matrix, stark_pdf, impurity_pdf
from tulasa import plotting_functions as pf

from tulasa.data_processing import wave_cal, add_noise, time_posterior, \
                                   add_chain_bounds

from tulasa import general # , fitting
# from tulasa.plotting_functions import plot_fit

from baysar.lineshapes import GaussianNorm
from baysar.input_functions import make_input_dict
from baysar.posterior import BaysarPosterior, BaysarPosteriorFilterWrapper

def blackouts(data, regions, level):
    for r in regions:
        reduce_index = general.in_between(data[0], r)
        data[1][reduce_index] = level

    return data[1]


if __name__=='__main__':

    # Load ASDEX data
    file = './AUG_ROV012_32244.txt'

    with open (file, "r") as myfile:
        tmp_data = myfile.read().replace('\n', '') # readlines()

    tmp_data = np.array([ float(t) for t in tmp_data.split() ])

    boundary = 1e5
    tmp_ems = tmp_data[np.where( np.array(tmp_data) > boundary)] * 1e-1 * 1e-4 # /nm/m2 to /A/cm2
    tmp_wave = tmp_data[np.where( np.array(tmp_data) < boundary)] * 10 # nm to A

    # Blacking out regions that are not being fitted
    regions = [ [4072, 4074], [4076.75, 4077.25], [4081, 4083] ]
    tmp_ems = blackouts([tmp_wave, tmp_ems], regions, 0.9e12)


    # Structuring input for BaySAR
    a_cal = 1e11 # fake-calibration constant

    experimental_emission = [tmp_ems]
    wavelength_axis = [tmp_wave]

    num_chords = 1
    emission_constant = [a_cal]
    noise_region = [ [3975, 3990] ] # the wavelength region that the noise is calculated from

    intfun = GaussianNorm(x=np.arange(31), cwl=15)
    instrument_function_fwhm = 0.4
    instrument_function = [intfun([instrument_function_fwhm/np.mean(np.diff(tmp_wave)), 1])]

    # Emitting plasma species
    species = ['D', 'N']
    ions = [ ['0'], ['1', '2', '3'] ]

    # Contaiminant lines from unknown species
    mystery_lines = [ [ [4024.78, 4025.33, 4026.33] ], [ [0.541, 0.373, 0.086] ] ]
    #                           wavelengths                 branching ratio


    # Creating input dictionary to create BaySAR posterior object
    indict = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                            instrument_function=instrument_function, emission_constant=emission_constant,
                            noise_region=noise_region, species=species, ions=ions,
                            mystery_lines=mystery_lines, refine=[0.05],
                            ion_resolved_temperatures=False, ion_resolved_tau=True)

    from baysar.plasmas import MeshLine, arb_obj

    num_points = 9
    x = np.linspace(1, 9, num_points)
    profile_function = MeshLine(x=x, zero_bounds=-2, bounds=[0, 10], log=True)
    profile_function = arb_obj(electron_density=profile_function,
                               electron_temperature=profile_function,
                               number_of_variables_ne=len(x),
                               number_of_variables_te=len(x),
                               bounds_ne=[11, 16], bounds_te=[-1, 2])

    posterior = BaysarPosterior(input_dict=indict, check_bounds=True,
                                curvature=1e3, print_errors=False,
                                profile_function=profile_function)

    posterior.plasma.theta_bounds[posterior.plasma.slices['cal0']] = [-1, 1]
    posterior.plasma.theta_bounds[posterior.plasma.slices['background0']] = [11.7, 12.5]
    posterior.plasma.theta_bounds[posterior.plasma.slices['electron_temperature']] = [-1, 1.5]
    posterior.plasma.theta_bounds[posterior.plasma.slices['N_1_dens']] = [13, 14]
    posterior.plasma.theta_bounds[posterior.plasma.slices['N_1_tau']] = [-1, 2]

    from tulasa.plotting_functions import plot_fit

    # make and plot a starting sample
    plot_start_samaple = False
    if plot_start_samaple:
        from tulasa.general import plot

        sample_num = 20
        sample = posterior.stormbreaker_start(sample_num, min_logp=-1000)

        plot([posterior(s) for s in sample])
        plot_fit(posterior, sample, size=int(sample_num / 2), alpha=0.2, ylim=(1e10, 1e16),
                 error_norm=True, plasma_ref=None)

    # Sampling using ParallelTempering (from inference.mcmc)
    sample_posterior = False
    if sample_posterior:
        from inference.mcmc import GibbsChain, PcaChain, ParallelTempering
        from scipy.optimize import fmin_l_bfgs_b

        start = posterior.stormbreaker_start(1, min_logp=-500).flatten()
        opt = fmin_l_bfgs_b(posterior.cost, start, approx_grad=True,
                            bounds=posterior.plasma.theta_bounds.tolist())

        chain = PcaChain(posterior=posterior, start=opt[0])
        chain.advance(100)

        try:
            chain.plot_diagnostics()
        except:
            pass
        chain.trace_plot()
        # chain.matrix_plot(plot_style='scatter')

        plot_fit(posterior, sample=chain.get_sample()[-50:], size=20, alpha=0.1,
                 ylim=(1e11, 1e16), error_norm=True)