
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

from tulasa import general, fitting
# from tulasa.plotting_functions import plot_fit

from baysar.lineshapes import GaussianNorm, Eich
from baysar.input_functions import make_input_dict
from baysar.posterior import BaysarPosterior, BaysarPosteriorFilterWrapper

import emcee as thor

# from inference.mcmc import GibbsChain, HamiltonianChain, PcaChain


def blackouts(data, regions, level):
    for r in regions:
        reduce_index = general.in_between(data[0], r)
        data[1][reduce_index] = level

    return data[1]


from tulasa.data_processing import hdi_estimator, calc_dl

def postproccess(posterior, sample, hdi_pc=0.85, print_counter=False, save=None, thin=1, lnprobs=None):

    output = {}

    output['sample'] = []
    output['logP'] = []

    num = int(len(sample) / thin)

    for counter0 in np.linspace(0, len(sample) - 1, num=num, dtype=int):

        theta = sample[counter0]

        output['sample'].append(theta)
        output['logP'].append(posterior(theta))

        # l_keys = ['n_ii_1p', 'n_ii_1g', 'n_ii_3g', 'n_iii_2f', 'n_iii_2p', 'n_iv_3g', 'delta', 'epsilon']
        l_keys = ['delta', 'epsilon', 'n_ii_3g', 'n_ii_1g', 'n_ii_1p', 'n_iii_2p', 'n_iii_2f', 'n_iv_3g']

        for counter1, l in enumerate(posterior.posterior_components[0].lines[:-1]):

            tmp_key = l_keys[counter1]

            if counter0 == 0:

                try:
                    l.n_upper

                    output[tmp_key] = \
                        {'ne_exc': [l.exc_ne], 'ne_rec': [l.rec_ne], 'te_exc': [l.exc_te], 'te_rec': [l.rec_te],
                         'f_rec': [l.f_rec], 'neutral_density': [posterior.plasma.plasma_state['D']['0']['conc']],
                         'dl': {'exc': [calc_dl(l.exc_profile, hdi_pc, posterior.plasma.plasma_state['los'])],
                                'rec': [calc_dl(l.rec_profile, hdi_pc, posterior.plasma.plasma_state['los'])],
                                'total': [calc_dl(l.ems_profile, hdi_pc, posterior.plasma.plasma_state['los'])]}}
                except AttributeError:
                    output[tmp_key] = \
                        {'ems_te': [l.ems_te], 'ems_ne': [l.ems_ne], 'ems_conc': [l.ems_conc],
                         'dl': [calc_dl(l.emission_profile, hdi_pc, posterior.plasma.plasma_state['los'])]}
                except:
                    raise

            else:

                if lnprobs is None:

                    try:
                        l.n_upper

                        output[tmp_key]['ne_exc'].append(l.exc_ne)
                        output[tmp_key]['ne_rec'].append(l.rec_ne)

                        output[tmp_key]['te_exc'].append(l.exc_te)
                        output[tmp_key]['te_rec'].append(l.rec_te)

                        output[tmp_key]['f_rec'].append(l.f_rec)

                        output[tmp_key]['neutral_density'].append(posterior.plasma.plasma_state['D']['0']['conc'])

                        try:
                            dl_exc = calc_dl(l.exc_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                        except:
                            dl_exc = np.nan

                        try:
                            dl_rec = calc_dl(l.rec_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                        except:
                            dl_rec = np.nan

                        try:
                            dl_ems = calc_dl(l.ems_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                        except:
                            dl_ems = np.nan

                        output[tmp_key]['dl']['exc'].append(dl_exc)
                        output[tmp_key]['dl']['rec'].append(dl_rec)
                        output[tmp_key]['dl']['total'].append(dl_ems)

                    except AttributeError:

                        output[tmp_key]['ems_te'].append(l.ems_te)
                        output[tmp_key]['ems_ne'].append(l.ems_ne)
                        output[tmp_key]['ems_conc'].append(l.ems_conc)

                        try:
                            tmp_dl = calc_dl(l.emission_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                        except ValueError:
                            tmp_dl = np.nan

                        output[tmp_key]['dl'].append(tmp_dl)

                    except:
                        raise

                else:

                    if lnprobs[counter0] < -150:
                        pass
                    else:
                        try:
                            l.n_upper

                            output[tmp_key]['ne_exc'].append(l.exc_ne)
                            output[tmp_key]['ne_rec'].append(l.rec_ne)

                            output[tmp_key]['te_exc'].append(l.exc_te)
                            output[tmp_key]['te_rec'].append(l.rec_te)

                            output[tmp_key]['f_rec'].append(l.f_rec)

                            output[tmp_key]['neutral_density'].append(posterior.plasma.plasma_state['D']['0']['conc'])

                            try:
                                dl_exc = calc_dl(l.exc_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                            except:
                                dl_exc = np.nan

                            try:
                                dl_rec = calc_dl(l.rec_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                            except:
                                dl_rec = np.nan

                            try:
                                dl_ems = calc_dl(l.ems_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                            except:
                                dl_ems = np.nan

                            output[tmp_key]['dl']['exc'].append(dl_exc)
                            output[tmp_key]['dl']['rec'].append(dl_rec)
                            output[tmp_key]['dl']['total'].append(dl_ems)

                        except AttributeError:

                            output[tmp_key]['ems_te'].append(l.ems_te)
                            output[tmp_key]['ems_ne'].append(l.ems_ne)
                            output[tmp_key]['ems_conc'].append(l.ems_conc)

                            try:
                                tmp_dl = calc_dl(l.emission_profile, hdi_pc, posterior.plasma.plasma_state['los'])
                            except ValueError:
                                tmp_dl = np.nan

                            output[tmp_key]['dl'].append(tmp_dl)

                        except:
                            raise



        if print_counter:
            print(counter0 + 1, len(sample), np.round(100 * (counter0 + 1) / len(sample), 3), ' %')

    if save is not None:
        np.savez(save, **output)
    else:
        return output


def new_bounds(sample):

    bounds = []

    for s in sample:

        bounds.append([min(s), max(s)])

    return bounds


if __name__=='__main__':

    # Load ASDEX data
    # file = os.path.expanduser('~/BaySAR/data/asdex_shots/AUG_ROV012_32244.txt')
    file = os.path.expanduser('~/baysar/demo/AUG_ROV012_32244.txt')

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
    wavelength_axes = [tmp_wave]

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
    indict = make_input_dict(num_chords=num_chords,
                             wavelength_axes=wavelength_axes, experimental_emission=experimental_emission,
                             instrument_function=instrument_function, emission_constant=emission_constant,
                             noise_region=noise_region, species=species, ions=ions,
                             mystery_lines=mystery_lines, refine=[0.01],
                             ion_resolved_temperatures=False, ion_resolved_tau=True)

    # Creating plamsa profile model
    from baysar.lineshapes import MeshLine

    x = np.linspace(1, 11, 7)
    profile_function = MeshLine(x=x, zero_bounds=-1, bounds=[0, 12])

    profile_function.number_of_varriables = len(x)
    profile_function.dr = False

    # Creating posterior objects
    posterior = BaysarPosterior(input_dict=indict, profile_function=profile_function,
                                check_bounds=True, curvature=1e4, print_errors=True)

    # Updating parameterisation
    posterior.plasma.conc = False # impurity/neutral density not concentration is sampled
    posterior.plasma.logte = True # log10(Te) not Te is sampled

    # Updating bounds for the updates parameterisation
    posterior.plasma.chain_bounds()

    # Manual updating bounds - as the Te and ne automated bounds function needs to be updated to work with the MeshLine
    posterior.plasma.theta_bounds[
    posterior.plasma.all_the_indicies['ne_index']:posterior.plasma.all_the_indicies['te_index']] = [12, 15]
    posterior.plasma.theta_bounds[
    posterior.plasma.all_the_indicies['te_index']:posterior.plasma.all_the_indicies['upper_te_index']] = [-1, 2]

    # Makes initial guess
    start = np.zeros(len(posterior.plasma.theta_bounds))

    start[0] = 12 # log10(background)
    start[posterior.plasma.all_the_indicies['ne_index']:posterior.plasma.all_the_indicies['te_index']] = \
        np.array([13.7, 14.5, 14.8, 14.5, 14.3, 14, 13.7]) # log10(ne) at each knot point in the MeshLine
    start[posterior.plasma.all_the_indicies['te_index']:posterior.plasma.all_the_indicies['upper_te_index']] = \
        np.array([-0.5, 0.1, 0.3, 0.5, 0.7, 1., .2]) # log10(Te) at each knot point in the MeshLine
    start[-12:-10] = 1 # B-field/los angle & B-field (Zeeman)
    start[-10:-6] = [13, 12.7, 11.5, 11] # log10(neutral, N II, N III, N IV)
    start[-6:-4] = 0.1 # log10(Ti) - D and N
    start[-4:-1] = [-1, -4, -4] # log10(tau) N II-IV
    start[-1] = 12.3 # X line spectral radiance (contaiminant lines)

    print(posterior(start)) # evaluation of the posterior (fit probability)

    pf.plot_fit(posterior, [start], alpha=1, size=1) # plots fit and plasma profiles from a given sample


    """
    Useful places
    
    posterior.posterior_components # list of spectrometer chord objects
    posterior.posterior_components() # returns the likelihood of fitting the spectra of this chord (there can be multiple)
    posterior.posterior_components[0].forward_model()  # returns the forward modelled spectra of this chord 
    posterior.posterior_components[0].lines # list of linemodels in each spectrometer chord objects
    [l() for l in posterior.posterior_components[0].lines] # list of all the evaluated lineshapes (before instrument convolution)
    """
