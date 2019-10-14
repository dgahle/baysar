

'''
demo of the building a posteror for fitting two spectra

has no prior
'''''

import os, sys, io
import time as clock

import numpy as np
from numpy import random

import scipy.io as sio
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from tulasa.general import plot, in_between, histagramm
from tulasa.plotting_functions import plot_fit, histmd

from baysar.lineshapes import Gaussian
from baysar.input_functions import make_input_dict
from baysar.plasmas import MeshLine, arb_obj
from baysar.posterior import BaysarPosterior, BaysarPosteriorFilterWrapper

def blackouts(data, regions, level):
    for r in regions:
        reduce_index=in_between(data[0], r)
        data[1][reduce_index]=level
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

    # #load tcv
    # file = '/home/dgahle/Downloads/64070_NConcTest.mat'
    # # file='/home/dgahle/baysar_work/data/tcv_shots/oliviers_shots_12_2017/59422_9_DSS.mat'
    # from scipy.io import loadmat
    # data = loadmat(file)
    #
    # spec_range = slice(480, 800)
    # tmp_ems = data['Spec'][spec_range].flatten()*1e-4*0.18
    # tmp_wave = data['WL'][spec_range].flatten()
    # # chord=15
    # # tmp_ems = data['shot']['LoadRaw'][0][0]['ACQ'][0][0]['Spectra'][0][0][spec_range, chord, :].mean(1)*1e-4*0.18
    # # tmp_wave = data['shot']['LoadCal'][0][0]['lambda'][0][0][spec_range, chord]

    # Blacking out regions that are not being fitted
    regions = [ [0, 3990], [3997, 4007], [4065, 4200] ]
    tmp_ems = blackouts([tmp_wave, tmp_ems], regions, 0.9e12)


    # Structuring input for BaySAR
    a_cal = 1e11 # fake-calibration constant

    experimental_emission = [tmp_ems]
    wavelength_axis = [tmp_wave]

    num_chords = 1
    emission_constant = [a_cal]
    noise_region = [ [4008, 4020] ] # the wavelength region that the noise is calculated from

    # # tcv instrument_function
    # file_dss = '/home/dgahle/baysar_work/DSS/InstrFunCharac_404nm_HOR.mat'
    # dss = loadmat(file_dss)
    # chord = 15
    # intfunl = dss['InstrFunL'][chord][::-1]
    # intfunr = dss['InstrFunR'][chord][1:]
    # intfun_old = np.concatenate((intfunl, intfunr))
    # instrument_function = [intfun_old]

    intfun = Gaussian(x=np.arange(31), cwl=15)
    instrument_function_fwhm = 0.8
    instrument_function = [intfun([instrument_function_fwhm/np.mean(np.diff(tmp_wave)), 1])]

    # Emitting plasma species
    species = ['N']
    ions = [ ['1'] ]

    # Contaiminant lines from unknown species
    mystery_lines = [ [ [4024.78, 4025.33, 4026.33] ], [ [0.541, 0.373, 0.086] ] ]
    #                           wavelengths                 branching ratio


    # Creating input dictionary to create BaySAR posterior object
    indict = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                            instrument_function=instrument_function, emission_constant=emission_constant,
                            noise_region=noise_region, species=species, ions=ions,
                            mystery_lines=mystery_lines, refine=[0.05],
                            ion_resolved_temperatures=False, ion_resolved_tau=True)

    num_points = 10
    x = np.linspace(0.5, 9.5, num_points)
    profile_function = MeshLine(x=x, zero_bounds=-2, bounds=[0, 10], log=True)
    profile_function = arb_obj(electron_density=profile_function,
                               electron_temperature=profile_function,
                               number_of_variables_ne=len(x),
                               number_of_variables_te=len(x),
                               bounds_ne=[11, 16], bounds_te=[-1, 2])

    posterior = BaysarPosterior(input_dict=indict, check_bounds=True,
                                curvature=1e2, print_errors=False,
                                profile_function=profile_function)

    rsample=[posterior.random_start(order=3, flat=True) for n in np.arange(100)]
    rsample=sorted(rsample, key=posterior.cost)
    # sample=posterior.sample_start(number=30, order=3, flat=True)
    # plot_fit(posterior, sample, alpha=1)

    from inference.mcmc import GibbsChain, PcaChain, ParallelTempering
    chain=PcaChain(posterior=posterior, start=rsample[0], parameter_boundaries=posterior.plasma.theta_bounds.tolist())
    chain.advance(100)
    # chain.plot_diagnostics()
    # chain.burn=500
    # chain.matrix_plot(reference=chain.mode(), plot_style='scatter')
    # sample = chain.get_sample(burn=500, thin=5)
    # plot_fit(posterior, sample, ylim=(1e9, 1e13), alpha=0.02)
    # conc=[]
    # n_te=[]
    # n_ne=[]
    # for t in sample:
    #     posterior(t)
    #     conc.append(posterior.posterior_components[0].lines[0].ems_conc)
    #     n_te.append(posterior.posterior_components[0].lines[0].ems_te)
    #     n_ne.append(posterior.posterior_components[0].lines[0].ems_ne)
    # # histmd(data=[conc, n_te, n_ne], params=[0, 1, 2], labels=None, burn=0, bins=20, save=None)
    # from tulasa.general import histagramm
    # histagramm(100*np.array(conc).flatten(), xlabel=r'$c_{N}(\%)$')
    # histagramm(np.array(n_te).flatten(), xlabel=r'$T_{e} \ / \ eV$')
    # histagramm(np.array(n_ne).flatten(), xlabel=r'$n_{e} \ / \ cm^{-3}$')
    # chain.plot_diagnostics()
