from numpy import square, sqrt, mean, linspace, nan, log, diff, arange, zeros, concatenate, where

from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import sparse
import numpy as np
import time as clock

import os, sys, io, warnings


from baysar.linemodels import XLine, ADAS406Lines, BalmerHydrogenLine
from baysar.lineshapes import Gaussian
from baysar.tools import clip_data, progressbar, centre_peak

class SpectrometerChord(object):

    """
    The SpectrometerChord object is meant to be a soft simulation of a spectrometer with wa single
    line of sight. From the input dictionary the error (Guassian noise and Poisson) of the emission
    is calculated and allows for the calculation of the likelihood of a give set of plasma parameters

    """

    def __init__(self, plasma, refine=None, chord_number=None):
        print("Building SpectrometerChord no. %d"%(chord_number))
        self.chord_number = chord_number
        self.plasma = plasma
        self.input_dict = self.plasma.input_dict
        self.refine = refine
        self.y_data = self.input_dict['experimental_emission'][self.chord_number]
        self.x_data = self.input_dict['wavelength_axis'][self.chord_number]
        if self.refine is not None:
            if not np.isreal(self.refine):
                raise TypeError('self.refine is not a real number')
            low, up = self.x_data.min(), self.x_data.max()+self.refine
            self.x_data_fm=arange(low, up, self.refine)
        else:
            self.x_data_fm=self.x_data

        self.dispersion=diff(self.x_data)[0]
        self.dispersion_fm=diff(self.x_data_fm)[0]
        self.dispersion_ratios=self.dispersion_fm/self.dispersion
        self.instrument_function=centre_peak(self.input_dict['instrument_function'][self.chord_number])
        self.instrument_function=np.true_divide(self.instrument_function, self.instrument_function.sum())
        self.noise_region=self.input_dict['noise_region'][self.chord_number]
        self.a_cal=self.input_dict['emission_constant'][self.chord_number]
        self.wavelength_calibrator=self.plasma.calwave_functions[self.chord_number]
        self.radiance_calibrator=self.plasma.cal_functions[self.chord_number]
        self.background_function=self.plasma.background_functions[self.chord_number]

        self.get_error()
        self.get_lines()
        self.int_func_sparce_matrix()
        print("Built SpectrometerChord no. %d"%(chord_number))

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

    def forward_model(self):
        self.wavelength_scaling()
        spectra = sum([l().flatten() for l in self.lines]) # TODO: This is what takes all the time?
        self.wavelength_scaling(inverse=True)

        background_theta=self.plasma.plasma_state['background'+str(self.chord_number)]
        continuum=self.background_function.calculate_background(background_theta)

        ems_cal_theta=self.plasma.plasma_state['cal'+str(self.chord_number)]
        spectra0=self.radiance_calibrator.inverse_calibrate(spectra, ems_cal_theta)

        # TODO - BIG SAVINGS BOGOF
        # TODO - centre of mass check?
        return self.instrument_function_matrix.dot(spectra)+continuum

        # spectra=fftconvolve(spectra0, self.instrument_function_fm, mode='same') # needs to be the interpolated int_func ?
        # sinterp=interp1d(self.x_data_fm, spectra)
        # return sinterp(self.x_data)

    def wavelength_scaling(self, inverse=False):
        cal_theta=self.plasma.plasma_state['calwave'+str(self.chord_number)]
        if inverse:
            calibrate=self.wavelength_calibrator.inverse_calibrate
        else:
            calibrate=self.wavelength_calibrator.calibrate

        for line in self.lines:
            if type(line)==XLine:
                line.line.cwl=calibrate(line.line.cwl, cal_theta)
            if type(line)==ADAS406Lines:
                line.linefunction.line.cwl=calibrate(line.linefunction.line.cwl, cal_theta)
            if type(line)==BalmerHydrogenLine:
                line.lineshape.cwl=calibrate(line.lineshape.cwl, cal_theta)

    def get_lines(self):
        # print("Getting line objects")
        self.lines = []
        for species in self.input_dict['species']:
            index=np.where([s=='_' for s in species])[0][0]
            elem=species[:index]
            for l in self.input_dict[species]:
                # sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
                sys.stdout.write("Getting line objects: {0} {1}".format(species, l, end="\r"))
                sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
                if any([elem==h for h in self.plasma.hydrogen_isotopes]):
                    line = BalmerHydrogenLine(self.plasma, species, l, self.x_data_fm)
                else:
                    line = ADAS406Lines(self.plasma, species, l, self.x_data_fm)
                self.lines.append(line)

        if 'X_lines' in self.input_dict:
            for l, f in zip(self.input_dict['X_lines'], self.input_dict['X_fractions']):
                sys.stdout.write("Getting XLine objects: {0}".format(l, end="\r"))
                sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
                line = XLine(plasma=self.plasma, species='X', cwl=l, fwhm=diff(self.x_data)[0],
                             wavelengths=self.x_data_fm, fractions=f)
                self.lines.append(line)

    def int_func_sparce_matrix(self):
        # self.centre_instrument_function()
        # todo - set default to refine=1 not None
        len_instrument_function = len(self.instrument_function)

        if len_instrument_function % 2 == 0:
            self.instrument_function_x = linspace(-(len_instrument_function // 2 - 0.5),
                                                  len_instrument_function // 2 - 0.5, len_instrument_function)
        else:
            self.instrument_function_x = linspace(-len_instrument_function // 2, len_instrument_function // 2,
                                                  len_instrument_function)

        int_func_interp = interp1d(self.instrument_function_x, self.instrument_function, bounds_error=False, fill_value=0.)

        if_x_fm_res=self.dispersion_ratios*np.diff(self.instrument_function_x)[0]
        self.instrument_function_x_fm=arange(self.instrument_function_x.min(),
                                            self.instrument_function_x.max(), if_x_fm_res)
        self.instrument_function_fm=int_func_interp(self.instrument_function_x_fm)
        self.instrument_function_fm/=self.instrument_function_fm.sum() # /self.refine)

        fine_axis = linspace(0, len(self.x_data), len(self.x_data_fm))
        shape = (len(self.x_data), len(self.x_data_fm))
        matrix = zeros(shape)
        for i in progressbar(np.arange(len(self.x_data)), 'Building convolution matrix: ', 30):
            matrix_i = int_func_interp(fine_axis - i)
            matrix[i, :] = matrix_i / sum(matrix_i)
            # k = sum(matrix_i)
            # if k == 0:
            #     matrix[i, :] = matrix_i
            # else:
            #     matrix[i, :] = matrix_i / k

        self.instrument_function_matrix=sparse.csc_matrix(self.dispersion_ratios*matrix)
        self.instrument_function_matrix.eliminate_zeros()
        self.check_instrument_function_matrix(self.instrument_function_matrix)

    @classmethod
    def check_instrument_function_matrix(self, matrix, accuracy=7):
        tmp0=np.ones(matrix.todense().shape[0])
        tmp1=np.ones(matrix.todense().shape[1])
        target=center_of_mass(tmp0)[0]
        shot=center_of_mass(matrix.dot(tmp1))[0]
        shift=target-shot
        if np.round(abs(shift), accuracy):
            raise ValueError("Instrumental function is not centred", shot, target)





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
