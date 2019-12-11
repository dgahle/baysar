"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""



import numpy as np
from numpy import random

import os, sys, io, copy

from baysar.line_data import adas_line_data

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
        elif type(stuff) in (int, float, np.float):
            pass
        else:
            raise TypeError("Input stuff is not of an appropriate type.")

        return any((box.min() < stuff) and (stuff < box.max()) for box in boxes)

def check_input_dict_input(wavelength_axis, experimental_emission, instrument_function,
                           emission_constant, noise_region,
                           species, ions, line_data=adas_line_data, mystery_lines=None,
                           refine=0.01, ion_resolved_temperatures=False, ion_resolved_tau=False):

    '''
    Checks all the input for make_input_dict.

    :param (list, array of arrays) wavelength_axis: Wavelength axis of each spectrometer chord.
    :param (list, array of arrays) experimental_emission: Spectra of each spectrometer chord.
    :param (list, array of arrays) instrument_function: Instrumental function for each spectrometer.
    :param (int, float or list, array of arrays) emission_constant: Calibration constant from counts to spectral radiance.
    :param (list, array of list, arrays) noise_region: Regions of background/continuum that can be used to estimate the measurement noise.
    :param (list of str) species: The elements in the plasma and experimental regions.
    :param (list of list of str) ions: The ions of the elements in the plasma.
    :param (dict) line_data: Line information such as central wavelengths, multiplet info and references to atomic data.
    :param (list of int, float) mystery_lines: Wavelengths and branching ratios of mystery lines in the experimental data.
    :param (int, float or list of int, float) refine: Resolution that the spectral forward model is evaluated on.
    :param (bool) ion_resolved_temperatures: If False there is an single ion temperature per element not per ion.
    :param (bool) ion_resolved_tau: If False there is an single particle density and ion confinement time (tau) per element not per ion.
    '''

    '''
    The checks that need to be done
    '''

    # Spectrometer data/input all needs to have the same lengths and be lists or arrays of lists or arrays
    exp_stuff = [wavelength_axis, experimental_emission, instrument_function, noise_region, emission_constant]
    # len check
    exp_len_check = [len(e)==len(exp_stuff[0]) for e in exp_stuff]
    if not all(exp_len_check):
        raise ValueError("[wavelength_axis, experimental_emission, instrument_function, noise_region, emission_constant] are not all the same length")

    # type check
    exp_type_check0 = [type(e) in (list, np.ndarray) for e in exp_stuff]
    if not all(exp_type_check0):
        raise TypeError("[wavelength_axis, experimental_emission, instrument_function, noise_region, emission_constant] are not all lists or arrays")
    exp_type_check1 = [type(e) in (list, np.ndarray) for exp in exp_stuff[:-1] for e in exp]
    if not all(exp_type_check1):
        raise TypeError("Contents of [wavelength_axis, experimental_emission, instrument_function, "
                        "noise_region] are not all lists or arrays")
    exp_type_check2 = all([np.isreal(exp).all() for exp in exp_stuff[:-1]])
    if not exp_type_check2:
        raise TypeError("Data given in [wavelength_axis, experimental_emission, instrument_function, noise_region] are not all ints or floats")
    if not all([np.isreal(e) for e in emission_constant]):
        raise TypeError("Emission constants type must be in (int, float, np.float64, np.int64)")

    # check that the noise_regions are within the wavelength_axes
    for region, axis in zip(noise_region, wavelength_axis):
        if any([within(r, axis) for r in region]):
            raise ValueError("Noise regions are not in the wavelength axes!")
        if np.diff(region)<np.diff(axis).min():
            raise ValueError("Noise regions is subpixel!")


    # species and ions must be the same lengths and only conatain strings
    if len(species)!=len(ions):
        raise ValueError("len(species)!=len(ions)")
    if not all([type(species)==list, all([type(s)==str for s in species])]):
        raise TypeError("species must be a list strings")
    if not all([ type(ions)==list,
                 all([type(i)==list for i in ions]),
                 all([type(i)==str for s in ions for i in s]) ]):
        raise TypeError("All ions must be a list of list of strings")

    # line_data must be a dictionary
    if type(line_data)!=dict:
        raise TypeError('type(line_data)!=dict. Input line_data must be a dict')

    instrument_function=[np.array(i, dtype=float) for i in instrument_function]

    # All species and ions must be in the line_data dict
    check_emitters_in_line_data = []
    emitter_fails = []
    for counter, s in enumerate(species):
        for i in ions[counter]:
            s_true = s in line_data
            i_true = i in line_data[s]
            check_emitters_in_line_data.append(s_true and i_true)
            if not check_emitters_in_line_data[-1]:
                emitter_fails.append(s+'_'+i)

    if not all(check_emitters_in_line_data):
        raise KeyError(emitter_fails, "not in given line_data")

    # mystery_lines must be a list of two lists of the same length which only contains lists
    if mystery_lines is not None:
        if type(mystery_lines)!=list:
            raise TypeError("type(mystery_lines)!=list")
        if len(mystery_lines[0])!=len(mystery_lines[1]):
            raise ValueError("len(mystery_lines[0])!=len(mystery_lines[1]) each mystery line needs a branching ratio even is it is a singlet ([1])")
        if not all([ len(mystery_lines)==2,
                     all([type(ml)==list for ml in mystery_lines]),
                     all([type(l)==list for ml in mystery_lines for l in ml]) ]):
            raise TypeError("mystery_lines must be a list of two lists than only contain lists")
        if not all([type(m) in (int, float) for ml in mystery_lines for l in ml for m in l]):
            raise TypeError("Some of `the give mystery line central wavelengths and/or branching ratios are not ints or floats")

    # refine must either a float or a list or array of the same length of all the other spectrometer input
    if type(refine) in (float, int):
        refine = [refine for wa in wavelength_axis]
    elif type(refine) in (list, tuple, np.ndarray):
        if len(refine)!=len(wavelength_axis):
            raise ValueError('len(refine)!=len(wavelength_axis) ' +
                             'type(refine) must be in (float, int, list, tuple, np.ndarray).' +
                             ' If type(refine) in (list, tuple, np.ndarray) then len(refine)==len(wavelength_axis)')
    else:
        raise TypeError('type(refine) must be in (float, int, list, tuple, np.ndarray).' +
                        ' If type(refine) in (list, tuple, np.ndarray) then len(refine)==len(wavelength_axis)')

    # The resolution parameters must be bools
    if not all([type(v)==bool for v in [ion_resolved_temperatures, ion_resolved_tau]]):
        raise TypeError('Not all ion resolutions are booleans')



def make_input_dict(wavelength_axis, experimental_emission, instrument_function,
                    emission_constant, noise_region,
                    species, ions, line_data=adas_line_data, mystery_lines=None,
                    refine=0.01, ion_resolved_temperatures=False, ion_resolved_tau=False):

    '''
    Returns a dict that is needed to instantiate both BaysarPosterior and SpectrometerChord classes.

    :param (list, array of arrays) wavelength_axis: Wavelength axis of each spectrometer chord.
    :param (list, array of arrays) experimental_emission: Spectra of each spectrometer chord.
    :param (list, array of arrays) instrument_function: Instrumental function for each spectrometer.
    :param (int, float or list, array of arrays) emission_constant: Calibration constant from counts to spectral radiance.
    :param (list, array of list, arrays) noise_region: Regions of background/continuum that can be used to estimate the measurement noise.
    :param (list of str) species: The elements in the plasma and experimental regions.
    :param (list of list of str) ions: The ions of the elements in the plasma.
    :param (dict) line_data: Line information such as central wavelengths, multiplet info and references to atomic data.
    :param (list of int, float) mystery_lines: Wavelengths and branching ratios of mystery lines in the experimental data.
    :param (int, float or list of int, float) refine: Resolution that the spectral forward model is evaluated on.
    :param (bool) ion_resolved_temperatures: If False there is an single ion temperature per element not per ion.
    :param (bool) ion_resolved_tau: If False there is an single particle density and ion confinement time (tau) per element not per ion.
    :return (dict) input_dict: Contains all the needed info instantiate both BaysarPosterior and SpectrometerChord classes.
    '''

    check_input_dict_input(wavelength_axis, experimental_emission, instrument_function,
                           emission_constant, noise_region, species, ions, line_data, mystery_lines,
                           refine, ion_resolved_temperatures, ion_resolved_tau)

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

    new_id = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                             instrument_function=instrument_function, emission_constant=emission_constant,
                             noise_region=noise_region, species=species, ions=ions,
                             mystery_lines=mystery_lines, refine=0.01,
                             ion_resolved_temperatures=False, ion_resolved_tau=True)

    pass
