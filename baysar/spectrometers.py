from numpy import square, sqrt, mean, linspace, nan, log, diff, arange, zeros, concatenate, where

from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import sparse
import numpy as np
import time as clock

import os, sys, io, warnings


from baysar.linemodels import XLine, ADAS406Lines, BalmerHydrogenLine
from baysar.lineshapes import Gaussian

def clip_data(x_data, y_data, x_range):

    '''
    returns sections of x and y data that is within the desired x range

    :param x_data:
    :param y_data:
    :param x_range:
    :return:
    '''

    assert len(x_data)==len(y_data), 'len(x_data)!=len(y_data)'

    x_index = where((min(x_range) < x_data) & (x_data < max(x_range)))

    return x_data[x_index], y_data[x_index]

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

from baysar.input_functions import within

class SpectrometerChord(object):

    """
    The SpectrometerChord object is meant to be a soft simulation of a spectrometer with wa single
    line of sight. From the input dictionary the error (Guassian noise and Poisson) of the emission
    is calculated and allows for the calculation of the likelihood of a give set of plasma parameters

    """

    def __init__(self, plasma, refine=None, chord_number=None):

        self.chord_number = chord_number
        self.plasma = plasma
        self.input_dict = self.plasma.input_dict
        self.refine = refine
        self.y_data = self.input_dict['experimental_emission'][self.chord_number]

        if self.refine is not None:
            assert type(self.refine) in (int, float), 'self.refine is not a int or float'
            self.x_data_exp = self.input_dict['wavelength_axis'][self.chord_number]
            low, up = min(self.x_data_exp), max(self.x_data_exp)
            self.x_data = linspace(low, up, int( abs(low - up) / self.refine ))
        else:
            self.x_data = self.input_dict['wavelength_axis'][self.chord_number]

        self.instrument_function = self.input_dict['instrument_function'][self.chord_number]
        self.noise_region = self.input_dict['noise_region'][self.chord_number]
        self.a_cal = self.input_dict['emission_constant'][self.chord_number]

        self.int_func_sparce_matrix()
        self.get_error()
        self.get_lines()

    def __call__(self, *args, **kwargs):
        return self.likelihood()

    def get_noise_spectra(self):
        if self.refine is None:
            self.x_data_continuum, self.y_data_continuum = clip_data(self.x_data, self.y_data, self.noise_region)
        else:
            self.x_data_continuum, self.y_data_continuum = clip_data(self.x_data_exp, self.y_data, self.noise_region)

    def get_error(self, fake_cal=False):
        self.get_noise_spectra()
        self.error = sqrt(square(mean(self.y_data_continuum)) +  # noise
                                  self.y_data * self.a_cal) # poisson

    def likelihood(self, cauchy=True):
        fm = self.forward_model() # * calibration_fractor
        assert len(fm) == len(self.y_data), 'len(fm) != len(self.y_data)'

        if cauchy:
            likeli_in = 1 + ( (self.y_data - fm) / self.error ) ** 2
            likeli_in = 1 / likeli_in
            likeli = sum( log(likeli_in) )
        else:
            likeli = - 0.5 * sum( ( (self.y_data - fm) / self.error) ** 2 )

        assert likeli != nan, 'likeli == nan'
        return likeli

    def forward_model(self, dont_interpolate=False):
        spectra = sum([ l().flatten() for l in self.lines ]) # TODO: This is what takes all the time
        continuum = self.plasma.plasma_state['background'+str(self.chord_number)]
        tmp_a_cal = self.plasma.plasma_state['cal'+str(self.chord_number)]
        spectra = (spectra / tmp_a_cal) + continuum

        if self.instrument_function_matrix is not None:
            spectra = self.instrument_function_matrix.dot(spectra)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectra = fftconvolve(spectra, self.instrument_function, mode='same')

        if dont_interpolate:
            return spectra

        spectra_interp=interp1d(self.x_data, spectra)
        if self.refine is not None:
            return spectra_interp(self.x_data_exp)
        else:
            return spectra_interp(self.x_data)

    def get_lines(self):
        self.lines = []
        for species in self.input_dict['species']:
            index = np.where([s=='_' for s in species])[0][0]
            elem = species[:index]
            for l in self.input_dict[species]:
                if any([elem==h for h in self.plasma.hydrogen_isotopes]):
                    line = BalmerHydrogenLine(self.plasma, species, l, self.x_data)
                else:
                    line = ADAS406Lines(self.plasma, species, l, self.x_data)
                self.lines.append(line)

        if 'X_lines' in self.input_dict:
            for l, f in zip(self.input_dict['X_lines'], self.input_dict['X_fractions']):
                line = XLine(plasma=self.plasma, species='X', cwl=l, fwhm=diff(self.x_data)[0],
                             wavelengths=self.x_data, fractions=f)
                self.lines.append(line)

    def int_func_sparce_matrix(self):
        if self.refine is not None:
            self.instrument_function_x = arange( len(self.instrument_function) )
            int_func_interp = interp1d(self.instrument_function_x, self.instrument_function)

            k = abs( mean( diff(self.x_data) ) /
                     mean( diff(self.x_data_exp)) )
            self.instrument_function_inter_x = arange(self.instrument_function_x.min(),
                                                      self.instrument_function_x.max(), k)

            int_func = int_func_interp(self.instrument_function_inter_x)
            res = len(self.x_data)
        else:
            int_func, res = (self.instrument_function, len(self.y_data))

        shape = (res, res)
        matrix = zeros(shape)
        buffer = zeros(res)

        long_int_func = concatenate( (buffer, int_func[::-1], buffer) )
        rang = arange(res)[::-1] + int( (len(int_func) + 1) / 2 ) # 16

        for i in progressbar(np.arange(len(rang)), 'Building convolution matrix: ', 30):
            key_i = rang[i]
            matrix_i = long_int_func[key_i:key_i + res]  # matrix_i # [::-1]
            k = sum(matrix_i)
            if k == 0:
                k = 1
            matrix[i, :] = matrix_i / k

        self.instrument_function_matrix = sparse.csc_matrix(matrix)


if __name__=='__main__':

    import numpy as np

    from baysar.input_functions import make_input_dict
    from baysar.plasmas import PlasmaLine

    num_chords = 1
    wavelength_axis = [np.linspace(3900, 4200, 512)]
    experimental_emission = [np.array([1e12*np.random.rand() for w in wavelength_axis[0]])]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ['D', 'N']
    ions = [ ['0'], ['1'] ] # , '2', '3'] ]
    noise_region = [[4040, 4050]]
    mystery_lines = [[[4070], [4001, 4002]], [[1], [0.4, 0.6]]]

    input_dict = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                                instrument_function=instrument_function, emission_constant=emission_constant,
                                noise_region=noise_region, species=species, ions=ions,
                                mystery_lines=mystery_lines, refine=[0.05],
                                ion_resolved_temperatures=False, ion_resolved_tau=True)

    plasma = PlasmaLine(input_dict)

    theta = np.array([min(b)+np.random.rand()*np.diff(b) for b in plasma.theta_bounds])

    theta[plasma.slices['electron_density']] = 14
    theta[plasma.slices['electron_temperature']] = 0.5
    theta[plasma.slices['N_1_dens']] = 12
    theta[plasma.slices['N_1_tau']] = 1

    plasma(theta)

    chord = SpectrometerChord(plasma, refine=0.05, chord_number=0)
    spectra = (sum([ l().flatten() for l in chord.lines ]) /
               chord.plasma.plasma_state['cal'+str(chord.chord_number)]) + chord.plasma.plasma_state['background'+str(chord.chord_number)]
    spectra = chord.instrument_function_matrix.dot(spectra)

    # for l in chord.lines:
    #     print(l, l.species, type(l()), l().flatten())

    print(chord())

    pass
