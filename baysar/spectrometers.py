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

    if not np.isreal(x_data[x_index]).all():
        print('not isreal(x_data[x_index])')
        print(x_data)
    if not np.isreal(y_data[x_index]).all():
        print('not isreal(Y_data[x_index])')
        print(y_data)

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
from scipy.ndimage.measurements import center_of_mass

def centre_peak0(peak, places=7):
    com=center_of_mass(peak)
    x=np.arange(len(peak))
    centre=np.mean(x)
    shift=com-centre
    interp=interp1d(x, peak, bounds_error=False, fill_value=0.)
    return interp(x+shift)

def centre_peak(peak, places=7):
    x=np.arange(len(peak))
    centre=np.mean(x)
    while not np.round(abs(centre-center_of_mass(peak))[0], places)==0:
        peak=centre_peak0(peak)
    return peak

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
        self.x_data = self.input_dict['wavelength_axis'][self.chord_number]
        if self.refine is not None:
            assert type(self.refine) in (int, float), 'self.refine is not a int or float'
            low, up = min(self.x_data), max(self.x_data)
            self.x_data_fm=linspace(low, up, int( abs(low - up) / self.refine ))
        else:
            self.x_data_fm=self.x_data

        self.instrument_function=centre_peak(self.input_dict['instrument_function'][self.chord_number])
        self.instrument_function=np.true_divide(self.instrument_function, self.instrument_function.sum())
        self.noise_region = self.input_dict['noise_region'][self.chord_number]
        self.a_cal = self.input_dict['emission_constant'][self.chord_number]

        self.int_func_sparce_matrix()
        self.get_error()
        self.get_lines()

    def __call__(self, *args, **kwargs):
        return self.likelihood()

    def get_noise_spectra(self):
        self.x_data_continuum, self.y_data_continuum = clip_data(self.x_data, self.y_data, self.noise_region)

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
            likeli=-0.5*sum(((self.y_data-fm)/self.error)**2)
        assert likeli != nan, 'likeli == nan'
        return likeli

    def forward_model(self, dont_interpolate=False):
        wave_cal = self.plasma.plasma_state['calwave'+str(self.chord_number)]
        self.wavelength_scaling(wave_cal)
        spectra = sum([l().flatten() for l in self.lines]) # TODO: This is what takes all the time?
        self.wavelength_scaling(1/wave_cal)
        continuum=self.plasma.plasma_state['background'+str(self.chord_number)]
        tmp_a_cal=self.plasma.plasma_state['cal'+str(self.chord_number)]
        spectra=(spectra/tmp_a_cal)+continuum

        if self.instrument_function_matrix is not None:
            spectra = self.instrument_function_matrix.dot(spectra)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectra = fftconvolve(spectra, self.instrument_function, mode='same')

        if dont_interpolate:
            return spectra

        if self.refine is not None:
            spectra_interp=interp1d(self.x_data_fm, spectra, bounds_error=False, fill_value='extrapolate')
            spectra=spectra_interp(self.x_data)

        return spectra

    def wavelength_scaling(self, wave_cal):
        for line in self.lines:
            if type(line)==XLine:
                line.line.x*=wave_cal
            if type(line)==ADAS406Lines:
                line.linefunction.line.x*=wave_cal
            if type(line)==BalmerHydrogenLine:
                line.lineshape.wavelengths*=wave_cal
                line.lineshape.doppler_function.line.x*=wave_cal

    def get_lines(self):
        self.lines = []
        for species in self.input_dict['species']:
            index = np.where([s=='_' for s in species])[0][0]
            elem = species[:index]
            for l in self.input_dict[species]:
                if any([elem==h for h in self.plasma.hydrogen_isotopes]):
                    line = BalmerHydrogenLine(self.plasma, species, l, self.x_data_fm)
                else:
                    line = ADAS406Lines(self.plasma, species, l, self.x_data_fm)
                self.lines.append(line)

        if 'X_lines' in self.input_dict:
            for l, f in zip(self.input_dict['X_lines'], self.input_dict['X_fractions']):
                line = XLine(plasma=self.plasma, species='X', cwl=l, fwhm=diff(self.x_data)[0],
                             wavelengths=self.x_data_fm, fractions=f)
                self.lines.append(line)

    def int_func_sparce_matrix(self):
        # self.centre_instrument_function()
        if self.refine is not None:
            len_instrument_function=len(self.instrument_function)
            self.instrument_function_x=arange(len_instrument_function)
            int_func_interp = interp1d(self.instrument_function_x, self.instrument_function)
            self.instrument_function_inter_x=np.linspace(self.instrument_function_x.min(),
                                                         self.instrument_function_x.max(),
                                                         int(len_instrument_function*len(self.x_data_fm)/len(self.x_data)))
            int_func = int_func_interp(self.instrument_function_inter_x)
        else:
            int_func = self.instrument_function

        res=len(self.x_data_fm)
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
    instrument_function = [np.array([  2.98900783e-37,   1.32796352e-32,   2.82044103e-28,   2.86364837e-24,
                                       1.38993388e-20,   3.22507917e-17,   3.57732507e-14,   1.89691652e-11,
                                       4.80850218e-09,   5.82697565e-07,   3.37557977e-05,   9.34814275e-04,
                                       1.23758228e-02,   7.83239560e-02,   2.36966490e-01,   3.42729148e-01,
                                       2.36966490e-01,   7.83239560e-02,   1.23758228e-02,   9.34814275e-04,
                                       3.37557977e-05,   5.82697565e-07,   4.80850218e-09,   1.89691652e-11,
                                       3.57732507e-14,   3.22507917e-17,   1.38993388e-20,   2.86364837e-24,
                                       2.82044103e-28,   1.32796352e-32,   2.98900783e-37])]
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
    chord=SpectrometerChord(plasma, refine=0.05, chord_number=0)
    print(chord())

    print(chord.instrument_function)
    print(chord.instrument_function_matrix.todense())
