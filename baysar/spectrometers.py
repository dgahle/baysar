from numpy import square, sqrt, mean, std, linspace, nan, log, diff, arange, zeros, concatenate, where

from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import sparse
import numpy as np
import time as clock

import os, sys, io, warnings

from baysar.linemodels import XLine, ADAS406Lines, BalmerHydrogenLine
from baysar.lineshapes import Gaussian
from baysar.tools import clip_data, progressbar, centre_peak, within

from adas import continuo

type_checking={'is_nan'} #

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
        if "continuo" in self.input_dict:
            self.continuo=self.input_dict["continuo"]

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
        self.instrument_function=self.input_dict['instrument_function'][self.chord_number]

        self.noise_region=self.input_dict['noise_region'][self.chord_number]
        self.a_cal=self.input_dict['emission_constant'][self.chord_number]
        self.anomalous_error=0.
        self.instrument_function_calibrator=self.plasma.calintfun_functions[self.chord_number]
        self.wavelength_calibrator=self.plasma.calwave_functions[self.chord_number]
        self.radiance_calibrator=self.plasma.cal_functions[self.chord_number]
        self.background_function=self.plasma.background_functions[self.chord_number]

        self.dot=False

        if not self.plasma.calibrate_intensity:
            self.plasma.plasma_state['cal_'+str(self.chord_number)]=np.array([1.])

        if not self.plasma.calibrate_wavelength:
            estimate_cwl=self.x_data.mean()
            estimate_disp=np.diff( self.x_data ).mean()
            self.plasma.plasma_state['calwave_'+str(self.chord_number)]=np.array([estimate_cwl, estimate_disp])

        self.get_error()
        self.get_lines()
        self.int_func_sparce_matrix()
        print("Built SpectrometerChord no. %d"%(chord_number))

    def __call__(self, *args, **kwargs):
        """
        Return the likelihood (logP) of the spectral fit.
        """

        return self.likelihood()

    def get_noise_spectra(self):
        """
        Clips out the spectral region to calculate the random noise.
        """

        self.x_data_continuum, self.y_data_continuum=clip_data(self.x_data, self.y_data, self.noise_region)

    def get_error(self, fake_cal=False):
        """
        Calculates the error on the spectra, accounting for random and shot noise
        as well as "anomalous" error which can be used to "blackout" areas of
        spectra that the user does not want to fit.
        """

        self.get_noise_spectra()
        self.error = sqrt(   square(std(self.y_data_continuum)) +  # noise
                                        self.y_data * self.a_cal +  # poisson
                              self.anomalous_error*self.y_data    ) # annomolus

    def likelihood(self, cauchy=True):
        """
        Calculates the likelihood (logP) of the spectral fit being forward
        modelled.
        """

        fm = self.forward_model() # * calibration_fractor
        if len(fm) != len(self.y_data):
            raise ValueError('len(fm) != len(self.y_data)')

        self.chi_squared=( (self.y_data-fm)**2/self.y_data ).sum()
        self.square_normalised_residuals=( (self.y_data-fm)/self.error )**2
        if any([sn_res<0 for sn_res in self.square_normalised_residuals]):
            raise ValueError('Some points of square_normalised_residuals are negative')

        if any([sn_res==0 for sn_res in self.square_normalised_residuals]):
            raise ValueError('Some points of square_normalised_residuals are zero')

        if any(np.isinf(self.square_normalised_residuals)):
            # for func in self.plasma.theta_functions:
            #     print(func, self.plasma.theta_functions[func])
            raise ValueError('Some points of square_normalised_residuals are inf')

        if any(np.isnan(self.square_normalised_residuals)):
            print(fm)
            print('nan indicies', np.where(np.isnan(self.square_normalised_residuals)))
            raise ValueError('Some points of square_normalised_residuals are nan')

        if cauchy:
            likeli = - log( (1 + self.square_normalised_residuals) ).sum( )
        else:
            likeli=-0.5*self.square_normalised_residuals.sum()

        if np.isnan(likeli):
            raise ValueError('likeli == nan')
        if any(np.isinf(self.square_normalised_residuals)):
            raise ValueError('Some points of abs prob are inf')

        return likeli

    def forward_model(self):
        """
        The forward_model evaluates the lines models of the SpecrtometerChord and
        applies the synthetic diagnostic applied in the diagnostic.

        The is synthetic diagnostic is defined by a wavelength and intensity
        calibration as well as the background level and instrument function.

        Spectra outputs in units of  ph/cm2/sr/A/s.
        """
        spectra = sum([l().flatten() for l in self.lines]) # TODO: This is what takes all the time?

        background_theta=self.plasma.plasma_state['background_'+str(self.chord_number)]
        background=self.background_function.calculate_background(background_theta)

        ems_cal_theta=self.plasma.plasma_state['cal_'+str(self.chord_number)]
        spectra=self.radiance_calibrator.inverse_calibrate(spectra, ems_cal_theta)

        if "continuo" in dir(self):
            if self.continuo:
                te=self.plasma.plasma_state['electron_temperature']
                ne=self.plasma.plasma_state['electron_density']
                n1=self.plasma.plasma_state['main_ion_density'].clip(1)
                dl=np.diff(self.plasma.los).sum()
                # shape wave, te
                ff_rates, total_rates=continuo(iz0=1, iz1=1, tev=te, wave=self.x_data)
                continuum=dl*total_rates*ne*n1
                spectra+=continuum.sum(1)/(4*np.pi)

        cif0='calint_func_'+str(self.chord_number) in self.plasma.plasma_state
        cif1='instrument_function_calibrator' in dir(self)
        calibrating_instrument_function=cif0 and cif1
        if calibrating_instrument_function:
            int_func_cal_theta=self.plasma.plasma_state['calint_func_'+str(self.chord_number)]
            pixels=np.arange( len(self.x_data_fm) )
            instrument_function_last_used=self.instrument_function_calibrator.calibrate(pixels, *int_func_cal_theta)
        else:
            instrument_function_last_used=self.instrument_function_fm

        # TODO - BIG SAVINGS BOGOF
        self.preconv_integral=np.trapz(spectra, self.x_data_fm)
        if self.dot:
            spectra=self.instrument_function_matrix.dot(spectra)

            if len(self.x_data)!=len(spectra):
                raise ValueError("len(self.x_data_fm)!=len(spectra). Lengths are {} and {}".format(len(self.x_data), len(spectra)))

            self.x_data_wavecal_interp=self.x_data
        else:
            spectra=fftconvolve(spectra, instrument_function_last_used, mode='same')

            if len(self.x_data_fm)!=len(spectra):
                raise ValueError("len(self.x_data_fm)!=len(spectra). Lengths are {} and {}".format(len(self.x_data_fm), len(spectra)))
            elif np.isinf(spectra).any():
                raise TypeError("spectra contains infs")
            elif np.isnan(spectra).any():
                raise TypeError("spectra contains NaNs")

            self.x_data_wavecal_interp=self.x_data_fm

        self.postconv_integral=np.trapz(spectra, self.x_data_wavecal_interp)
        self.conv_integral_ratio=self.postconv_integral/self.preconv_integral
        if not np.isclose(self.conv_integral_ratio, 1):
            raise ValueError(f"Area not conserved in convolution! ({self.conv_integral_ratio}!=1), {self.dispersion_ratios}, {instrument_function_last_used.sum()}")

        spectra+=background
        self.prewavecal_spectra=spectra
        # wave calibration
        self.wavecal_interp=interp1d(self.x_data_wavecal_interp, spectra, bounds_error=False, fill_value="extrapolate")
        cal_theta=self.plasma.plasma_state['calwave_'+str(self.chord_number)]
        self.cal_wave=self.wavelength_calibrator.calibrate(self.x_data, cal_theta)
        spectra=self.wavecal_interp(self.cal_wave).clip(background)

        # output checks
        if len(self.y_data)!=len(spectra):
            raise ValueError("len(self.y_data)!=len(spectra). Lengths are {} and {}".format(len(self.y_data), len(spectra)))
        elif np.isnan(spectra).any():
            raise TypeError("spectra contains NaNs")
        elif np.isinf(spectra).any():
                raise TypeError("spectra contains infs")

        return spectra


    def get_lines(self):
        """
        Builds and collects line objects that make up the atomic transitions and
        mystery lines being modelling in the spectral region.
        """

        # print("Getting line objects")
        self.lines = []
        for species in self.input_dict['species']:
            index=np.where([s=='_' for s in species])[0][0]
            elem=species[:index]
            for l in self.input_dict[species]:
                # check if the line is in the wavelength region
                if within(l, [self.x_data]):
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
                # check if the line is in the wavelength region
                if within(l, [self.x_data]):
                    sys.stdout.write("Getting XLine objects: {0}".format(l, end="\r"))
                    sys.stdout.flush()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    line = XLine(plasma=self.plasma, species='X', cwl=l, fwhm=diff(self.x_data)[0],
                                 wavelengths=self.x_data_fm, fractions=f)
                    self.lines.append(line)

    def int_func_sparce_matrix(self):
        """
        Pixel normalises and centres the instrument function and makes copies in
        the resolution of the synthetic diagnostic, forward model and sparse
        matrix form of the instrument function.
        """

        # check that the instrument_function is centred and normalised
        self.instrument_function=centre_peak(self.instrument_function)
        self.instrument_function=np.true_divide(self.instrument_function, self.instrument_function.sum())
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
        # self.instrument_function_fm/=np.trapz(self.instrument_function_fm, self.instrument_function_x_fm)
        self.instrument_function_fm=np.true_divide(self.instrument_function_fm, self.instrument_function_fm.sum())
        self.instrument_function_fm=centre_peak(self.instrument_function_fm)

        fine_axis = linspace(0, len(self.x_data), len(self.x_data_fm))
        shape = (len(self.x_data), len(self.x_data_fm))
        matrix = zeros(shape)
        for i in progressbar(np.arange(len(self.x_data)), 'Building convolution matrix: ', 30):
            matrix_i = int_func_interp(fine_axis - i)
            matrix[i, :] = matrix_i / sum(matrix_i)

        self.instrument_function_matrix=sparse.csc_matrix(matrix) # *self.dispersion_ratios)
        self.instrument_function_matrix.eliminate_zeros()
        self.check_instrument_function_matrix(self.instrument_function_matrix)

    @classmethod
    def check_instrument_function_matrix(self, matrix, accuracy=7):
        """
        Checks that the dot prduct with the instrument function matrix does
        not change the centre of mass of an arrray. Default to an accuracy
        of 7 decimal places.
        """

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
