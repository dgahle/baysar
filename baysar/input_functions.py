"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""



import numpy as np
from numpy import random

import os, sys, io, copy

from baysar.line_data import line_data_multiplet, adas_line_data

def within(stuff, boxes):

    """
    This function checks is a value or list/array of values (the sample) are 'within' an array or values.

    :param stuff: a list of strings, ints or floats
    :param box: Array of values to check the sample(s) against
    :return: True or False if all the samples are in the array
    """

    if type(stuff) in (tuple, list):
        return any([any([(box.min() < float(s)) and (float(s) < box.max()) for s in stuff])
                    for box in boxes])
    else:
        if type(stuff) is str:
            stuff = float(stuff)

        return any((box.min() < stuff) and (stuff < box.max()) for box in boxes)

def make_input_dict(num_chords,
                    wavelength_axis, experimental_emission, instrument_function,
                    emission_constant, noise_region,
                    species, ions, line_data=line_data_multiplet,
                    mystery_lines=None, refine=None,
                    ion_resolved_temperatures=False, ion_resolved_tau=False):

    """
    This function takes in information about the spectrometer settings, experimental data, and plamsa contents
    and returns a dictionary needed to instantiate the BaySARPosterior class for spectral fitting.

    :param num_chords: Number (int) of spectrometer chords sampling the same/symetrical plamsma that are wanted to be fit
    :param wavelength_axis / A: List of wavelength arrays for each spectrometer chord
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

        input_dict['chords'][counter0]['data']['wavelength_axis'] = wavelength_axis[counter0]
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
                    if within(line, wavelength_axis):

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
                            # input_dict['physics'][tmp_species][ion]['lines'].remove(line)
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

def new_input_dict(wavelength_axis, experimental_emission, instrument_function,
                   emission_constant, noise_region,
                   species, ions, line_data=adas_line_data, mystery_lines=None,
                   refine=0.01, ion_resolved_temperatures=False, ion_resolved_tau=False):

    experimental_data_checks = [len(wavelength_axis)==len(experimental_emission),
                                len(wavelength_axis)==len(instrument_function),
                                len(wavelength_axis)==len(emission_constant),
                                len(wavelength_axis)==len(noise_region)]

    experimental_data_checks_fail_statement = 'Not all (wavelength_axis, experimental_emission, instrument_function, noise_region) are the same length'

    assert all(experimental_data_checks), experimental_data_checks_fail_statement
    assert len(species)==len(ions), 'len(species)!=len(ions)'
    assert len(mystery_lines[0])==len(mystery_lines[1]), 'len(mystery_lines[0])!=len(mystery_lines[1]) each X line needs to be given multiplet fractions'

    if type(refine) in (float, int):
        refine = [refine for wa in wavelength_axis]
    elif type(refine) in (list, tuple, np.ndarray):
        assert len(refine)==len(wavelength_axis), 'len(refine)!=len(wavelength_axis)'
    else:
        raise TypeError('type(refine) must be in (float, int, list, tuple, np.ndarray).' +
                        ' If type(refine) in (list, tuple, np.ndarray) then len(refine)==len(wavelength_axis)')

    input_dict = {}

    input_dict['species'] = []

    for s, i in zip(species, ions):
        if s in line_data:
            for ion in i:
                sion = s+'_'+ion
                if ion in line_data[s]:
                    input_dict[sion] = line_data[s][ion]
                    if 'default_pecs' in input_dict[sion]:
                        input_dict[sion].pop('default_pecs')
                    input_dict['species'].append(sion)
                    for line in list(input_dict[sion].keys()):
                        if not within(line, wavelength_axis):
                            input_dict[sion].pop(line)
                else:
                    print(sion + ' is not in adas_line_data')
        else:
            print(s + ' is not in adas_line_data')

    data = [wavelength_axis, experimental_emission, instrument_function,
            emission_constant, noise_region, refine]
    data_string = ['wavelength_axis', 'experimental_emission', 'instrument_function',
                   'emission_constant', 'noise_region', 'refine']

    for d, d_str in zip(data, data_string):
        input_dict[d_str] = d

    input_dict['ion_resolved_temperatures'] = ion_resolved_temperatures
    input_dict['ion_resolved_tau'] = ion_resolved_tau

    if mystery_lines is not None:
        input_dict['X_lines'] = mystery_lines[0]
        input_dict['X_fractions'] = mystery_lines[1]

    return input_dict

if __name__=='__main__':

    num_chords = 1
    wavelength_axis = [np.linspace(3900, 4200, 512)]
    experimental_emission = [np.array([1e12*np.random.rand() for w in wavelength_axis[0]])]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ['D', 'N']
    ions = [ ['0'], ['1', '2', '3'] ]
    noise_region = [[4040, 4050]]
    mystery_lines = [ [ [4070], [4001, 4002] ],
                      [    [1],    [0.4, 0.6]]]

    input_dict = make_input_dict(num_chords=num_chords,
                                 wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                                 instrument_function=instrument_function, emission_constant=emission_constant,
                                 noise_region=noise_region, species=species, ions=ions,
                                 mystery_lines=mystery_lines, refine=[0.01],
                                 ion_resolved_temperatures=False, ion_resolved_tau=True)

    new_id = new_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                            instrument_function=instrument_function, emission_constant=emission_constant,
                            noise_region=noise_region, species=species, ions=ions,
                            mystery_lines=mystery_lines, refine=0.01,
                            ion_resolved_temperatures=False, ion_resolved_tau=True)

    pass