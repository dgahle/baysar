"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""



import numpy as np
from numpy import random

import os, sys, io, copy

from baysar.line_data import line_data_multiplet

def within(sample, array):

    """
    This function checks is a value or list/array of values (the sample) are 'within' an array or values.

    :param sample: a list of strings, ints or floats
    :param array: Array of values to check the sample(s) against
    :return: True or False if all the samples are in the array
    """

    acceptable_types = (int, float)

    if type(sample) == str:

        if sample[0] == '[':
            sample = sample[1:-1].replace(', ', ' ').split()
        else:
            sample = sample.replace(', ', ' ').split()

    elif type(sample) == list:

        try:
            assert any([type(sample) == t for t in acceptable_types])
        except AssertionError:
            if type(sample[0]) == str:
                sample = [float(s) for s in sample]
            else:
                print('Ahh, poop.', AssertionError)
                print('Sample: ', sample)
                raise
        except:
            print("Unexpected error:", sys.exc_info())  # [0])
            raise
    else:
        sample = [sample]

    if array[0] == acceptable_types:
        return any([ (float(s) > min(array) and float(s) < max(array)) for s in sample])
    else:
        return any([ any([(float(s) > min(a) and float(s) < max(a)) for s in sample]) for a in array ])



def make_input_dict(num_chords,
                    wavelength_axes, experimental_emission, instrument_function,
                    emission_constant, noise_region,
                    species, ions, line_data=line_data_multiplet,
                    mystery_lines=None, refine=None,
                    ion_resolved_temperatures=False, ion_resolved_tau=False):

    """
    This function takes in information about the spectrometer settings, experimental data, and plamsa contents
    and returns a dictionary needed to instantiate the BaySARPosterior class for spectral fitting.

    :param num_chords: Number (int) of spectrometer chords sampling the same/symetrical plamsma that are wanted to be fit
    :param wavelength_axes / A: List of wavelength arrays for each spectrometer chord
    :param experimental_emission / ph cm^-2 A^-1 sr^-1 s^-1: List of spectra arrays for each spectrometer chord
    :param instrument_function: List of instrument functions for each spectrometer chord
    :param emission_constant: List of calibration constants to convert from counts to spectral radiance (ph cm^-2 A^-1 sr^-1 s^-1)
    :param noise_region: List of 2 element list(s) which have the upper and lower bound of the wavelength region to have the noise calculated
    :param species: List of strings of the elements in the plasma to be forward modeled
    :param ions: List of list of strings of the ionisation stages of the ions of the elements in the plasma to be forward modeled
    :param line_data: Dictionary of line data which contains relavent atomic information such as atomic mass and infomation on LS
                      transitions such as the rate data and multiplet splitting. This is set by default but can be changed by the
                      user if desired
    :param mystery_lines: List of central wavelengths of unknown lines which need to be fited by emmissivities. The structure is a list of two lists.
                          The first list (mystery_lines[0]) is of the central wavelengths of the lines. For multiplets mystery_lines[0][i] is a list
                          of the lines in the multiplet to be fitted. The second list (mystery_lines[1]) is a list of list of the multiplet fractions
    :return: dictionary needed to instantiate the BaySARPosterior class
    """

    input_dict = {}

    input_dict['number_of_chords'] = num_chords

    input_dict['inference_resolution'] = \
        {'ion_resolved_temperatures': ion_resolved_temperatures, 'ion_resolved_tau': ion_resolved_tau}

    input_dict['chords'] = {}

    for counter0, chord in enumerate(np.arange(num_chords)):

        input_dict['chords'][counter0] = {}
        input_dict['chords'][counter0]['data'] = {}

        input_dict['chords'][counter0]['data']['wavelength_axis'] = wavelength_axes[counter0]
        input_dict['chords'][counter0]['data']['experimental_emission'] = experimental_emission[counter0]

        input_dict['chords'][counter0]['data']['instrument_function'] = instrument_function[counter0]
        input_dict['chords'][counter0]['data']['emission_constant'] = emission_constant[counter0]
        input_dict['chords'][counter0]['data']['noise_region'] = noise_region[counter0]

        if refine is not None:
            input_dict['chords'][counter0]['data']['refine'] = refine[counter0]



    input_dict['physics'] = line_data

    # do we want to keep the species?
    input_dict['physics'] = {key: value for key, value in input_dict['physics'].items() if key in species}

    # do we want to keep the ion?
    for tmp_species in input_dict['physics'].keys():

        # what are the ions
        tmp_boul_list = [s == tmp_species for s in species]

        tmp_ion_index = np.where(tmp_boul_list)

        bad_types = [tuple, list, np.ndarray]

        try:
            while any([t == type(tmp_ion_index) for t in bad_types]):
                tmp_ion_index = tmp_ion_index[0]
        except IndexError:
            print(tmp_species)
            print(tmp_boul_list)
            print(input_dict['physics'].keys())
            raise
        except:
            raise

        ion_str = [str(i) for i in ions[tmp_ion_index]]

        tmp_dict = input_dict['physics'][tmp_species]
        keep_ion_keys = ['atomic_mass', 'atomic_charge', 'effective_charge_406']

        input_dict['physics'][tmp_species] = {key: value for key, value in tmp_dict.items() if key in ion_str+keep_ion_keys}

        input_dict['physics'][tmp_species]['ions'] = ion_str

        # what lines do we want to keep
        for ion in input_dict['physics'][tmp_species]['ions']:

            # input_dict['physics'][tmp_species][ion]['no_data_lines'] = 0

            # filter for cwl from list of lists wavelength axis
            all_lines = list( input_dict['physics'][tmp_species][ion].keys() )
            for line in all_lines:

                # print(tmp_species, ion, line)

                input_dict['physics'][tmp_species][ion]['no_data_lines'] = 0

                # if line != 'lines':
                if not any([line == l for l in ('lines', 'no_data_lines')]):
                    if within(line, wavelength_axes):

                        try:
                            input_dict['physics'][tmp_species][ion][line]['tec']
                        except KeyError:
                            input_dict['physics'][tmp_species][ion]['no_data_lines'] += 1
                        except:
                            raise

                        pass

                    else:
                        try:
                            input_dict['physics'][tmp_species][ion].pop(line)
                            input_dict['physics'][tmp_species][ion]['lines'].remove(line)
                        except:
                            print(tmp_species, ion, line)
                            raise
        pass

    # add mystery lines
    if mystery_lines is not None:

        input_dict['physics']['X'] = {}

        xmultiplet_couter = 0

        for tmp_line in mystery_lines[0]:

            if type(tmp_line) == list:

                tmp = str(tmp_line).replace(',', '')

                input_dict['physics']['X'][tmp[1:len(tmp)-1]] = {'wavelength': tmp_line,
                                                                 'fractions': mystery_lines[1][xmultiplet_couter]}
                xmultiplet_couter += 1
            else:
                input_dict['physics']['X'][str(tmp_line)] = {'wavelength': tmp_line}


    else: pass

    return input_dict



if __name__=='__main__':

    pass