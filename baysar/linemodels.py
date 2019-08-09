
"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""


from scipy.io import readsav
from scipy.constants import pi
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

from scipy.signal import fftconvolve
import scipy.constants

from numpy import diff, log10

import copy
import numpy as np
import warnings

import sys
import time as clock

def reduce_wavelength(wavelengths, cwl, half_range, return_indicies=False):

    """
    This function returns an array which contains a subsection of input array ('wavelengths') which is
    between and inclucing the points 'cwl' +- 'half_range'. If the end of this range is outside of the
    'wavelength' array then the end of the reduced array is the end of the 'wavelength'.

    :param wavelengths: Input array to be reduced
    :param cwl: Point in the array which will be the centre of the new reduced array
    :param half_range: Half the range of the new array.
    :param return_indicies: Boulean (False by default) which when True the function returns the indicies
                            of 'wavelengths' that match the beginning and end of the reduced array
    :return: Returns an subset of 'wavelengths' which is centred around 'cwl' with a range of 'cwl' +-
             'half_range'
    """

    upper_cwl = cwl + half_range
    lower_cwl = cwl - half_range

    if upper_cwl > max(wavelengths):
        upper_index = len(wavelengths) - 1
    else:
        tmp_wave = abs(wavelengths - upper_cwl)
        upper_index = np.where(tmp_wave == min(tmp_wave))[0][0]

    if lower_cwl < max(wavelengths):
        lower_index = 0
    else:
        tmp_wave = abs(wavelengths - lower_cwl)
        lower_index = np.where(tmp_wave == min(tmp_wave))[0][0]

    # print(lower_index, upper_index)

    if return_indicies:
        return wavelengths[lower_index:upper_index+1], [lower_index, upper_index]
    else:
        return wavelengths[lower_index:upper_index + 1]

def build_tec406(filename):

    """
    Takes idl .sav file and builds a 3D interpolator (using RegularGridInterpolator) for the Total
    Emission Coefficent as a function of ion confinement time, electron density and temperature

    :param filename: string of the .sav location
    :return: Returns a 3D RegularGridInterpolator for ne*TEC(tau, ne, Te)
    """

    try:
        tmp_data = readsav(filename)
    except FileNotFoundError:
        tmp_data = readsav(filename.replace(' ', ''))
    except:
        raise

    tmp_te = np.log10(tmp_data['te']).tolist()
    tmp_ne = np.log10(tmp_data['edens']).tolist()
    tmp_tau = np.log10(tmp_data['tau']).tolist()

    tmp_tec_grid0 = tmp_data['tec_grid']
    tmp_tec_grid = np.log10(copy.copy(tmp_tec_grid0)).tolist()

    tmp_3d = (tmp_tau, tmp_ne, tmp_te)

    bounds_error = False

    try:
        return RegularGridInterpolator(tmp_3d, tmp_tec_grid, bounds_error=bounds_error)
    except ValueError:
        tmp_tec_grid = reshape_tec_grid(tmp_tec_grid)
        return RegularGridInterpolator(tmp_3d, tmp_tec_grid, bounds_error=bounds_error)
    except:
        raise

# TODO: This function actually needs writing
def build_pec(filename, reaction_key=0):

    """
    Takes idl .sav file and builds a 2D interpolator (using RectBivariateSpline) for the
    Photon Emissivity Coeffiecient (PEC) as a function of electron temperature and density

    :param filename: string of the .sav location
    :param reaction_key: 0 by default which indexes for the excitation PEC data, 1 for the
                         recombination PEC data
    :return: Returns a 2D RectBivariateSpline for PEC(ne, Te)
    """

    try:
        tmp_data = readsav(filename)
    except FileNotFoundError:
        tmp_data = readsav(filename.replace(' ', ''))
    except:
        raise

    tmp_te = tmp_data['te']
    tmp_ne = tmp_data['edens']

    tmp_tec_grid = tmp_data['pecs'][:, :, reaction_key]

    # print( tmp_ne.shape, tmp_te.shape, tmp_tec_grid.shape )

    return RectBivariateSpline( tmp_ne, tmp_te, tmp_tec_grid.T )

def reshape_tec_grid(tec_grid):

    """
    Takes the 3D ne*TEC Tensor and rearranges the dimentions into an appropriate structure for
    making the 3D interpolator using build_pec()

    :param tec_grid: 3D ne*TEC Tensor
    :return: restructured 3D ne*TEC Tensor
    """

    old_shape = tec_grid.shape
    new_shape = old_shape[::-1]

    new_tec_grid = np.zeros(new_shape)

    for counter0 in np.arange(new_shape[0]):

        for counter1 in np.arange(new_shape[1]):


            try:
                new_tec_grid[counter0, counter1, :] = tec_grid[:, counter1, counter0]
            except (ValueError, IndexError):
                print(counter0, counter1)
                print(old_shape, new_shape)
                raise
            except:
                raise

    return new_tec_grid

class BasicLine(object):

    """
    This class is a manager object which takes the input of higher level classes such as
    NoADASLines, Xline and ADAS406Lines and the lineshape object such as Guassian which
    are purely statistical and return line shapes.
    """

    def __init__(self, cwl, wavelengths, lineshape, vectorise, fractions=[]):

        self.cwl = cwl
        self.wavelengths = wavelengths

        self.vectorise = vectorise

        try:
            self.lineshape = lineshape(wavelengths, cwl, vectorise=self.vectorise,
                                       fractions=fractions)
        except TypeError:
            self.lineshape = lineshape(wavelengths, cwl, vectorise=self.vectorise)
        except: raise

    def __call__(self, fwhm, ems):

        return self.lineshape([ [fwhm], [ems] ])


class DopplerLine(BasicLine):

    """
    This class is used to calculate Guassian lineshapes as a function of area and ion
    temperature. It is inherited by NoADASLine, Xline and ADAS406Line and used by
    NoADASLines and ADAS406Lines.
    """

    def __init__(self, cwl, wavelengths, lineshape, atomic_mass, vectorise=1, half_range=5):

        self.wavelengths = wavelengths
        self.long_wavelengths = wavelengths

        self.reduced_wavelength, self.reduced_wavelength_indicies = \
            reduce_wavelength(wavelengths, cwl, half_range, return_indicies=True)

        self.zeros_peak = np.zeros(len(self.wavelengths))

        super(DopplerLine, self).__init__(cwl, self.reduced_wavelength, lineshape, vectorise=vectorise)

        self.atomic_mass = atomic_mass

    def __call__(self, ti, ems):

        fwhm = self.ti_to_fwhm(ti)

        peak = self.zeros_peak  # np.zeros(len(self.wavelengths))
        peak[min(self.reduced_wavelength_indicies):max(self.reduced_wavelength_indicies) + 1] = \
            super(DopplerLine, self).__call__(fwhm, ems)

        assert len(peak) == len(self.long_wavelengths), \
            'len(peak) ' + str(len(peak)) + ' != len(self.wavelengths) ' \
            + str(len(self.long_wavelengths))

        return peak

    def ti_to_fwhm(self, ti):

        '''
        wavelengths are in A
        temp in eV
        atomic weights in atomic units

        ion_temp = 1.67e8 * atomic_weight * (fwhm / cwl) ** 2
        '''''

        # tmp = ti / (1.67e8 * self.atomic_mass)
        # tmp = 5.988e-9 / (ti * self.atomic_mass)
        # tmp = 2 * np.sqrt(tmp)

        tmp = 7.715e-5 * np.sqrt(ti / self.atomic_mass)

        return self.cwl * tmp # = fwhm


class ADAS405Line(DopplerLine):

    """
    This class is used to calculate Gaussian lineshapes as a function of impurity concentration,
    ion temperature, emission length, electron temperature and density. The units returned are
    in spectra radiance (ph cm^-3 A^-1 s^-1).
    """

    def __init__(self, cwl, wavelengths, lineshape, atomic_mass, tec405, length):

        super(ADAS405Line, self).__init__(cwl, wavelengths, lineshape, atomic_mass)

        self.tec405 = tec405

        self.length_per_sr = length / (4 * pi)

    def __call__(self):

        n0 = self.plasma[self.isotope]['conc']
        ti = self.plasma[self.isotope]['ti']

        ne = self.plasma['electron_density']
        te = self.plasma['electron_temperature']

        ems = n0 * self.tec405(ne, te) / self.length_per_sr

        return super(ADAS405Line, self).__call__(ti, ems)

# TODO: Need to update to inherit from ADAS405Line
class ADAS406Line(DopplerLine):

    """
    This class is used to calculate Gaussian lineshapes as a function of impurity concentration,
    ion temperature, emission length, ion confinement time, electron temperature and density. The
    units returned are in spectra radiance (ph cm^-3 A^-1 s^-1).
    """

    def __init__(self, cwl, wavelengths, lineshape, atomic_mass, tec406,
                 species, ion, plasma, jj_frac, half_range=10, fractions=[]):

        '''
        tmp_line = ADAS406Line(cwl=tmp_cwl, wavelengths=tmp_wavelengths, lineshape=tmp_lineshape,
                                           atomic_mass=tmp_ma, tec406=tmp_tec, length=tmp_l, isotope=isotope,
                                           ion=ion, plasma=self.plasma)

        :param cwl:
        :param wavelengths:
        :param lineshape:
        :param atomic_mass:
        :param tec406:
        :param length:
        :param species:
        :param ion:
        :param plasma:
        '''

        self.wavelengths = wavelengths

        super(ADAS406Line, self).__init__(cwl, self.wavelengths, lineshape, atomic_mass,
                                          vectorise=len(plasma['los']))

        # length = diff (self.plasma['los'] )[0]
        # self.length_per_sr = length / (4 * pi)

        self.plasma = plasma

        self.species = species
        self.ion = ion

        self.jj_frac = jj_frac

        if type(tec406)==str:
            self.tec406 = build_tec406(tec406)
        else:
            self.tec406 = tec406

        self.tec_input = []


    def __call__(self):

        self.times = [clock.time()]
        self.time_labels = ['start of call']

        n0 = self.plasma[self.species]['conc']
        ti = self.plasma[self.species]['ti']

        ne = self.plasma['electron_density']
        te = self.plasma['electron_temperature']
        tau = np.zeros( len(ne) ) + self.plasma[self.species]['tau']

        tec_in = np.array([tau, ne, te]).T
        # self.tec_input.append(tec_in)

        length = diff (self.plasma['los'] )[0]
        length_per_sr = length / (4 * pi)

        self.times.append(clock.time())
        self.time_labels.append('before tec evaluation')

        # ems = n0 * ne * np.nan_to_num( self.tec406( tec_in ) ) * length_per_sr * self.jj_frac
        ems = n0 * np.nan_to_num( self.tec406( tec_in ) ) * length_per_sr * self.jj_frac
        # TODO only takes one at a time or returns NaNs

        ems = ems.clip(min=0)

        self.emission_profile = ems

        self.ems_ne = sum(ems * ne) / sum(ems)

        self.times.append(clock.time())
        self.time_labels.append('after tec evaluation')

        return super(ADAS406Line, self).__call__(ti, ems)

        # peak = self.zeros_peak  # np.zeros(len(self.wavelengths))
        # peak[min(self.reduced_wavelength_indicies):max(self.reduced_wavelength_indicies) + 1] = \
        #     super(ADAS406Line, self).__call__(ti, ems)
        #
        # assert len(peak) == len(self.long_wavelengths), \
        #     'len(peak) ' + str( len(peak) ) + ' != len(self.wavelengths) ' \
        #     + str( len(self.long_wavelengths) )
        #
        # self.times.append(clock.time())
        # self.time_labels.append('end of call')
        #
        # return peak

    def call(self, ti, n0, ne, te, tau):

        length = diff(self.plasma['los'])[0]
        length_per_sr = length / (4 * pi)

        # ems = n0 * ne * self.tec406( (tau, ne, te) ) * length_per_sr  * self.jj_frac
        ems = n0 * self.tec406( (tau, ne, te) ) * length_per_sr  * self.jj_frac

        self.emission_profile = ems

        # return super(ADAS406Line, self).__call__(ti, ems)

        peak = self.zeros_peak  # np.zeros(len(self.wavelengths))
        peak[min(self.reduced_wavelength_indicies):max(self.reduced_wavelength_indicies) + 1] = \
            super(ADAS406Line, self).__call__(ti, ems)

        assert len(peak) == len(self.long_wavelengths), \
            'len(peak) ' + str(len(peak)) + ' != len(self.wavelengths) ' \
            + str(len(self.long_wavelengths))

        return peak


class ADAS406Lines(object):

    """
    This class is used to calculate Gaussian lineshapes as a function of impurity concentration,
    ion temperature, emission length, ion confinement time, electron temperature and density. The
    units returned are in spectra radiance (ph cm^-3 A^-1 s^-1).

    This class can be used to model multiplets not just singlets. In the case of multiplets one
    ne*TEC(tau, ne, Te) is used for the LS transition and the fraction of the emission for each
    each jj transition is scaled by a fixed scalar (ne*TEC(tau, ne, Te)*f_jj).
    """

    def __init__(self, cwls, wavelengths, lineshape, atomic_mass, tec406,
                 species, ion, plasma, jj_frac, half_range=10, fractions=[]):

        '''
        tmp_line = ADAS406Line(cwl=tmp_cwl, wavelengths=tmp_wavelengths, lineshape=tmp_lineshape,
                                           atomic_mass=tmp_ma, tec406=tmp_tec, length=tmp_l, isotope=isotope,
                                           ion=ion, plasma=self.plasma)

        :param cwl:
        :param wavelengths:
        :param lineshape:
        :param atomic_mass:
        :param tec406:
        :param length:
        :param species:
        :param ion:
        :param plasma:
        '''

        self.wavelengths = wavelengths

        if type(cwls) != list:
            self.cwls = [cwls]
        else:
            self.cwls = cwls

        self.lines = []

        for counter, tmp_cwl in enumerate(self.cwls):

            out_of_bounds = ( (tmp_cwl < min(self.wavelengths)) or
                              (tmp_cwl > max(self.wavelengths))     )

            if out_of_bounds:
                pass
            else:

                tmp_line = DopplerLine(tmp_cwl, self.wavelengths, lineshape, atomic_mass,
                                       vectorise=len(plasma['los']))

                self.lines.append(tmp_line)


        # super(ADAS406Line, self).__init__(cwl, self.wavelengths, lineshape, atomic_mass,
        #                                   vectorise=len(plasma['electron_density']))

        # length = diff (self.plasma['los'] )[0]
        # self.length_per_sr = length / (4 * pi)

        self.plasma = plasma

        self.species = species
        self.ion = ion

        if type(jj_frac) != list:
            self.jj_frac = [jj_frac]
        else:
            self.jj_frac = jj_frac

        if type(tec406)==str:
            self.tec406 = build_tec406(tec406)
        else:
            self.tec406 = tec406

        self.tec_input = []

    def __call__(self):

        self.times = [clock.time()]
        self.time_labels = ['start of call']

        n0 = self.plasma[self.species][self.ion]['conc']
        ti = self.plasma[self.species][self.ion]['ti']

        ne = self.plasma['electron_density']
        te = self.plasma['electron_temperature']
        tau = np.zeros(len(ne)) + self.plasma[self.species][self.ion]['tau']

        tec_in = np.log10( np.array([tau, ne, te]).T )
        # self.tec_input.append(tec_in)

        length = diff(self.plasma['los'])[0]
        length_per_sr = length / (4 * pi)

        self.times.append(clock.time())
        self.time_labels.append('before tec evaluation')

        # ems = n0 * ne * np.nan_to_num( self.tec406( tec_in ) ) * length_per_sr * self.jj_frac
        ems = n0 * np.power(10, self.tec406(tec_in)) * length_per_sr # * self.jj_frac
        # TODO only takes one at a time or returns NaNs

        ems = ems.clip(min=1e-20)

        self.emission_profile = ems

        self.ems_ne = sum(ems*ne) / sum(ems)
        self.ems_conc = n0 / self.ems_ne
        self.ems_te = sum(ems*te) / sum(ems)

        self.times.append(clock.time())
        self.time_labels.append('after tec evaluation')

        try:
            peaks = [ l(ti, ems) * self.jj_frac[counter] for counter, l in enumerate(self.lines) ]
        except TypeError:
            print(self.jj_frac)
            print(self.lines)
            raise
        except:
            raise

        # return super(ADAS406Line, self).__call__(ti, ems)

        sum_peaks = sum(peaks)

        self.times.append(clock.time())
        self.time_labels.append('end of call')

        return sum_peaks


class HydrogenLineShape(object):

    def __init__(self, cwl, wavelengths, n_upper, n_lower, atomic_mass, zeeman=True):

        self.cwl = cwl
        self.wavelengths = wavelengths

        self.zeeman = zeeman

        from BaySAR.lineshapes import GaussiansNorm # , Gaussian

        wavelengths_doppler_num = len(self.wavelengths)

        if type(wavelengths_doppler_num/2) != int:
            wavelengths_doppler_num += 1

        self.wavelengths_doppler = np.linspace(self.cwl-10, self.cwl+10, wavelengths_doppler_num)

        self.doppler_function = DopplerLine(self.cwl, self.wavelengths_doppler, GaussiansNorm, atomic_mass)

        self.loman_dict  =  {'32': [0.7665, 0.064, 3.710e-18],  # Balmer Series
                             '42': [0.7803, 0.050, 8.425e-18],
                             '52': [0.6796, 0.030, 1.310e-15],
                             '62': [0.7149, 0.028, 3.954e-16],
                             '72': [0.7120, 0.029, 6.258e-16],
                             '82': [0.7159, 0.032, 7.378e-16],
                             '92': [0.7177, 0.033, 8.947e-16],
                             '102': [0.7158, 0.032, 1.239e-15],
                             '112': [0.7146, 0.028, 1.632e-15],
                             '122': [0.7388, 0.026, 6.459e-16],
                             '132': [0.7356, 0.020, 9.012e-16],

                             '43': [0.7449, 0.045, 1.330e-16],  # Paschen Series
                             '53': [0.7356, 0.044, 6.640e-16],
                             '63': [0.7118, 0.016, 2.481e-15],
                             '73': [0.7137, 0.029, 3.270e-15],
                             '83': [0.7133, 0.032, 4.343e-15],
                             '93': [0.7165, 0.033, 5.588e-15], }

        self.n_upper = n_upper
        self.n_lower = n_lower

        self.loman_ij_abc = self.loman_dict[str(self.n_upper) + str(self.n_lower)]

        self.get_delta_magnetic_quantum_number()
        self.bohr_magnaton = scipy.constants.physical_constants['Bohr magneton in K/T'][0]

        self.electron_charge = scipy.constants.e
        self.electron_mass = scipy.constants.m_e
        self.speed_of_light = scipy.constants.c

        pass

    def __call__(self, theta):

        if self.zeeman:
            electron_density, electron_temperature, ion_temperature, b_field, viewangle = theta
        else:
            electron_density, electron_temperature, ion_temperature = theta

        stark_component = self.stehle_param(electron_density, electron_temperature)

        doppler_component = self.doppler_function(ion_temperature, 1)

        peak = fftconvolve(stark_component, doppler_component, 'same')
        peak /= np.trapz(peak, self.wavelengths)

        if self.zeeman:
            return self.zeeman_split(peak, b_field, viewangle)

        else:
            return peak

    def stehle_param(self, electron_density, electron_temperature):


        # Paramaterised MMM Stark profile coefficients from Bart's paper

        # loman_abc = self.loman_ij_abc[str(self.n_upper) + str(self.n_lower)]
        a_ij, b_ij, c_ij = self.loman_ij_abc

        delta_lambda_12ij = c_ij * ((1e6 * electron_density) ** a_ij) / (electron_temperature ** b_ij)  # nm

        ls_s = 1 / (abs((self.wavelengths - self.cwl)) ** (5. / 2.) +
                    (10 * delta_lambda_12ij / 2) ** (5. / 2.))

        ls_s /= np.trapz(ls_s, self.wavelengths)

        return ls_s

    def zeeman_split(self, peak, b_field, viewangle):

        """
         returns input lineshape, with Zeeman splitting accounted for by a simple model

        :param x:
        :param x_centre:
        :param ls:
        :param x_units:
        :return:

        """

        viewangle *= np.pi

        rel_intensity_pi = 0.5 * np.sin(viewangle) ** 2
        rel_intensity_sigma = 0.25 * (1 + np.cos(viewangle) ** 2)

        freq_shift_sigma = self.electron_charge / (4 * np.pi * self.electron_mass) * b_field

        # wave_shift_sigma = self.delta_magnetic_quantum_number * self.bohr_magnaton * b_field
        # wave_shift_sigma = self.delta_magnetic_quantum_number * 0.5 * b_field
        wave_shift_sigma = abs(self.cwl - 1 / (1/(self.cwl) - freq_shift_sigma / 299792458.0e10))


        # print(freq_shift_sigma, wave_shift_sigma)

        # relative intensities normalised to sum to one

        ls_sigma_minus = rel_intensity_sigma * np.interp(self.wavelengths + wave_shift_sigma, self.wavelengths, peak)
        ls_sigma_plus = rel_intensity_sigma * np.interp(self.wavelengths - wave_shift_sigma, self.wavelengths, peak)
        ls_pi = rel_intensity_pi * peak

        return ls_sigma_minus + ls_pi + ls_sigma_plus

    def get_delta_magnetic_quantum_number(self):

        upper_m = 1
        lower_m = 0

        self.delta_magnetic_quantum_number = upper_m - lower_m

    def comparison(self, theta, line_model='stehle_param'):

        electron_density, ion_temperature, b_field, viewangle = theta

        line = BalmerLineshape(self.n_upper, electron_density, ion_temperature, bfield=b_field,
                               viewangle=viewangle, line_model=line_model,
                               wl_axis=self.wavelengths / 1e10, wl_centre=self.cwl / 1e10,
                               override_input_check=True)

        return line.ls_szd / np.trapz(line.ls_szd, self.wavelengths)



from pystark import BalmerLineshape, ls_norm


class BalmerHydrogenLine(object):

    def __init__(self, cwl, wavelengths, n_upper, atomic_mass, pec,
                 species, ion, plasma, half_range=40, zeeman=True):

        self.plasma = plasma

        self.n_upper = n_upper
        self.atomic_mass = atomic_mass

        self.cwl = cwl
        self.wavelengths = wavelengths

        self.reduced_wavelength, self.reduced_wavelength_indicies = \
            reduce_wavelength(wavelengths, cwl, half_range, return_indicies=True)

        self.zeros_peak = np.zeros(len(self.wavelengths))


        self.species = species
        self.ion = ion

        # if type(pec) == str:
        #     self.pec = build_pec(pec)
        # else:
        #     self.pec = pec

        # self.exc_tec = build_tec406(pec[0])
        self.exc_pec = build_pec(pec[1], 0)
        self.rec_pec = build_pec(pec[1], 1)

        self.lineshape = HydrogenLineShape(self.cwl, self.reduced_wavelength, self.n_upper, n_lower=2,
                                           atomic_mass=self.atomic_mass, zeeman=zeeman)

        self.tec_input = []

        self.two_pi = 2 * np.pi

    def __call__(self):

        n0 = self.plasma[self.species][self.ion]['conc']
        n1 = self.plasma['main_ion_density']
        ti = self.plasma[self.species][self.ion]['ti']

        bfield = self.plasma['B-field']
        viewangle = self.plasma['view_angle'] / self.two_pi

        ne = self.plasma['electron_density']
        te = self.plasma['electron_temperature']
        # tau = np.zeros(len(ne)) + self.plasma[self.species]['tau']
        #
        # tec_in = np.array([tau, ne, te]).T

        length = diff(self.plasma['los'])[0]
        length_per_sr = length / (4 * pi)

        rec_pec = [ self.rec_pec(tmp_ne, te[counter])[0][0] for counter, tmp_ne in enumerate(ne) ]
        rec_profile = n1 * ne * length_per_sr * rec_pec

        exc_pec = [self.exc_pec(tmp_ne, te[counter])[0][0] for counter, tmp_ne in enumerate(ne)]
        exc_profile = n0 * ne * length_per_sr * np.nan_to_num( exc_pec )
        # exc_profile = n0 * length_per_sr * np.nan_to_num( self.exc_tec(tec_in) )

        low_te = 0.2
        low_ne = 1e11

        low_te_indicies = np.where(te < low_te)
        low_ne_indicies = np.where(ne < low_ne)

        rec_profile[low_te_indicies] = 0.0
        exc_profile[low_te_indicies] = 0.0

        rec_profile[low_ne_indicies] = 0.0
        exc_profile[low_ne_indicies] = 0.0

        self.f_rec = sum(rec_profile) / sum(rec_profile + exc_profile)

        self.exc_profile = exc_profile
        self.rec_profile = rec_profile
        self.ems_profile = rec_profile + exc_profile

        rec_weighted_electron_density = np.sum( rec_profile * ne ) / sum( rec_profile )
        exc_weighted_electron_density = np.sum( exc_profile * ne ) / sum( exc_profile )

        rec_weighted_electron_temperature = np.sum( rec_profile * te ) / sum( rec_profile )
        exc_weighted_electron_temperature = np.sum( exc_profile * te ) / sum( exc_profile )

        ems_weighted_electron_density = np.sum( self.ems_profile * ne ) / sum( self.ems_profile )
        ems_weighted_electron_temperature = np.sum( self.ems_profile * te ) / sum( self.ems_profile )

        self.exc_ne = exc_weighted_electron_density
        self.rec_ne = rec_weighted_electron_density

        self.exc_te = exc_weighted_electron_temperature
        self.rec_te = rec_weighted_electron_temperature

        self.ems_ne = ems_weighted_electron_density
        self.ems_te = ems_weighted_electron_temperature

        line_models = ['rosato', 'stehle', 'stehle_param', 'voigt']
        line_model_key = 2

        # tmp_wavelengths = self.wavelengths
        tmp_wavelengths = self.reduced_wavelength

        '''
        if self.zeeman:
            electron_density, electron_temperature, ion_temperature, b_field, viewangle = theta
        else:
            electron_density, electron_temperature, ion_temperature = theta
            '''

        self.exc_lineshape_input = [exc_weighted_electron_density, self.exc_te, ti, bfield, viewangle]
        self.rec_lineshape_input = [rec_weighted_electron_density, self.rec_te, ti, bfield, viewangle]

        exc_peak = np.nan_to_num( self.lineshape(self.exc_lineshape_input) )
        rec_peak = np.nan_to_num( self.lineshape(self.rec_lineshape_input) )

        self.exc_peak = exc_peak
        self.rec_peak = rec_peak

        tmp_peak = rec_peak * sum(rec_profile) + exc_peak * sum(exc_profile)

        self.tmp_peak = tmp_peak

        peak = self.zeros_peak # np.zeros(len(self.wavelengths))
        peak[min(self.reduced_wavelength_indicies):max(self.reduced_wavelength_indicies)+1] = tmp_peak

        assert len(peak) == len(self.wavelengths), 'len(peak) != len(self.wavelengths)'

        return peak

        # return tmp_peak

    def OLDCALL(self):

        n0 = self.plasma[self.species][self.ion]['conc']
        n1 = self.plasma['main_ion_density']
        ti = self.plasma[self.species][self.ion]['ti']

        bfield = self.plasma['B-field']
        viewangle = self.plasma['view_angle'] / self.two_pi

        ne = self.plasma['electron_density']
        te = self.plasma['electron_temperature']
        # tau = np.zeros(len(ne)) + self.plasma[self.species]['tau']
        #
        # tec_in = np.array([tau, ne, te]).T

        length = diff(self.plasma['los'])[0]
        length_per_sr = length / (4 * pi)

        rec_pec = [ self.rec_pec(tmp_ne, te[counter])[0][0] for counter, tmp_ne in enumerate(ne) ]
        rec_profile = n1 * ne * length_per_sr * rec_pec

        exc_pec = [self.exc_pec(tmp_ne, te[counter])[0][0] for counter, tmp_ne in enumerate(ne)]
        exc_profile = n0 * ne * length_per_sr * np.nan_to_num( exc_pec )
        # exc_profile = n0 * length_per_sr * np.nan_to_num( self.exc_tec(tec_in) )

        low_te = 0.2
        low_te_indicies = np.where(te < low_te)

        rec_profile[low_te_indicies] = 0.0
        exc_profile[low_te_indicies] = 0.0

        self.f_rec = sum(rec_profile) / sum(rec_profile + exc_profile)

        self.exc_profile = exc_profile
        self.rec_profile = rec_profile
        self.ems_profile = rec_profile + exc_profile

        rec_weighted_electron_density = np.sum( rec_profile * ne ) / sum( rec_profile )
        exc_weighted_electron_density = np.sum( exc_profile * ne ) / sum( exc_profile )

        ems_weighted_electron_density = np.sum( self.ems_profile * ne ) / sum( self.ems_profile )
        ems_weighted_electron_temperature = np.sum( self.ems_profile * te ) / sum( self.ems_profile )

        self.exc_ne = exc_weighted_electron_density
        self.rec_ne = rec_weighted_electron_density

        self.ems_ne = ems_weighted_electron_density
        self.ems_te = ems_weighted_electron_temperature

        line_models = ['rosato', 'stehle', 'stehle_param', 'voigt']
        line_model_key = 2

        # tmp_wavelengths = self.wavelengths
        tmp_wavelengths = self.reduced_wavelength

        rec_peak = BalmerLineshape(self.n_upper, rec_weighted_electron_density*1e6, ti, bfield=bfield,
                                   viewangle=viewangle, line_model=line_models[line_model_key],
                                   wl_axis=tmp_wavelengths/1e10, wl_centre=self.cwl/1e10,
                                   override_input_check=True)

        exc_peak = BalmerLineshape(self.n_upper, exc_weighted_electron_density*1e6, ti, bfield=bfield,
                                   viewangle=viewangle, line_model=line_models[line_model_key],
                                   wl_axis=tmp_wavelengths/1e10, wl_centre=self.cwl/1e10,
                                   override_input_check=True)





        tmp_peak = ls_norm(rec_peak.ls_szd, tmp_wavelengths, norm_type='area') * sum(rec_profile) + \
                    ls_norm(exc_peak.ls_szd, tmp_wavelengths, norm_type='area') * sum(exc_profile)

        peak = self.zeros_peak # np.zeros(len(self.wavelengths))
        peak[min(self.reduced_wavelength_indicies):max(self.reduced_wavelength_indicies)+1] = tmp_peak

        assert len(peak) == len(self.wavelengths), 'len(peak) != len(self.wavelengths)'

        return peak

        # return tmp_peak

    def comparison(self, fast=True):

        new = self()

        old = self.OLDCALL()

        if fast:
            return new / old
        else:
            return new, old


class NoADASLine(DopplerLine):

    """
    This class is used to model lines of known ions but with no atomic data and instead of
    an emission an effective TEC are fitted.
    """

    def __init__(self, cwl, wavelengths, lineshape, atomic_mass,
                 species, ion, plasma, jj_frac, multiplet=None):

        '''
        tmp_line = ADAS406Line(cwl=tmp_cwl, wavelengths=tmp_wavelengths, lineshape=tmp_lineshape,
                                           atomic_mass=tmp_ma, tec406=tmp_tec, length=tmp_l, isotope=isotope,
                                           ion=ion, plasma=self.plasma)

        :param cwl:
        :param wavelengths:
        :param lineshape:
        :param atomic_mass:
        :param tec406:
        :param length:
        :param species:
        :param ion:
        :param plasma:
        '''


        super(NoADASLine, self).__init__(cwl, wavelengths, lineshape, atomic_mass,
                                          vectorise=len(plasma['los']) )

        self.plasma = plasma

        self.species = species
        self.ion = ion

        self.jj_frac = jj_frac

        self.mulitplet = multiplet

    def __call__(self):

        n0 = self.plasma[self.species][self.ion]['conc']
        ti = self.plasma[self.species][self.ion]['ti']

        ne = self.plasma['electron_density']

        try:
            conc_tec_jj_frac = self.plasma[self.species][self.ion][str(self.cwl)]['conc*tec*jj_frac']
        except KeyError:
            if self.mulitplet is not None:
                conc_tec_jj_frac = self.plasma[self.species][self.ion][str(self.mulitplet)]['conc*tec*jj_frac']
            else:
                raise
        except:
            raise

        length = diff (self.plasma['los'] )[0]
        length_per_sr = length / (4 * pi)

        try:
            ems = n0 * conc_tec_jj_frac * length_per_sr * self.jj_frac
            # TODO only takes one at a time or returns NaNs
        except TypeError:
            print('n0, conc_tec_jj_frac, length_per_sr, self.jj_frac')
            print(n0, conc_tec_jj_frac, length_per_sr, self.jj_frac)
            raise
        except:
            raise

        self.emission_profile = ems

        return super(NoADASLine, self).__call__(ti, ems)

    def call(self, ti, n0, ne, conc_tec_jj_frac):

        length = diff(self.plasma['los'])[0]
        length_per_sr = length / (4 * pi)

        ems = n0 * conc_tec_jj_frac # * length_per_sr  * self.jj_frac

        self.emission_profile = ems

        return super(NoADASLine, self).__call__(ti, ems)


class NoADASLines(object):

    def __init__(self, cwl, wavelengths, lineshape, atomic_mass,
                 species, ion, plasma, jj_frac):

        '''
        tmp_line = ADAS406Line(cwl=tmp_cwl, wavelengths=tmp_wavelengths, lineshape=tmp_lineshape,
                                           atomic_mass=tmp_ma, tec406=tmp_tec, length=tmp_l, isotope=isotope,
                                           ion=ion, plasma=self.plasma)

        :param cwl:
        :param wavelengths:
        :param lineshape:
        :param atomic_mass:
        :param tec406:
        :param length:
        :param species:
        :param ion:
        :param plasma:
        '''

        if type(cwl) == list and type(jj_frac) == list:
            self.cwl = cwl
            self.fractions = jj_frac
        else:
            self.cwl = [cwl]
            self.fractions = [jj_frac]

        self.species = species
        self.ion = ion

        self.lines = []

        for counter, c in enumerate(self.cwl):
            self.lines.append( NoADASLine(c, wavelengths, lineshape, atomic_mass,
                                          species, ion, plasma, self.fractions[counter],
                                          multiplet=self.cwl) )

    def __call__(self):

        return sum( [l() for l in self.lines] )

    def call(self, ti, n0, ne, conc_tec_jj_frac):

        return sum([l(ti, n0, ne, conc_tec_jj_frac) for l in self.lines])


class XLine(BasicLine):

    def __init__(self, cwl, fwhm, wavelengths, lineshape, plasma, species, fractions=[]):

        try:
            vectorise = len(plasma['los'])
        except:
            raise

        self.vectorise = vectorise

        super(XLine, self).__init__(cwl, wavelengths, lineshape, fractions=fractions,
                                    vectorise=self.vectorise)

        self.plasma = plasma

        self.line = str(cwl)

        self.species = species

        self.fwhm = fwhm

        length = diff(self.plasma['los'])[0]
        self.length_per_sr = length / (4 * pi)

        pass

    def __call__(self, *args, **kwargs):

        try:
            conc_tec_jj_frac = self.plasma[self.species][self.line]['conc*tec*jj_frac']
        except KeyError:
            tmp = str(self.line).replace(',', '')
            # tmp = tmp[1:len(tmp)-1]
            conc_tec_jj_frac = self.plasma[self.species][tmp[1:len(tmp)-1]]['conc*tec*jj_frac']
        except:
            raise

        ne = self.plasma['electron_density']

        ems = conc_tec_jj_frac # * ne * self.length_per_sr
        # TODO only takes one at a time or returns NaNs ??

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            try:
                assert not np.array([ t == None for t in super(XLine, self).__call__(self.fwhm, ems) ]).any()
            except AssertionError:
                raise
            except TypeError:
                # print( [p(self.fwhm, ems) for p in self.lineshape.peaks] )
                print( [p for p in self.lineshape.peaks] )
                print( [l.cwl for l in self.lineshape.__dict__['peaks']] )
                raise
            except ValueError:
                print( len( super(XLine, self).__call__(self.fwhm, ems) ) )
                print( super(XLine, self).__call__(self.fwhm, ems).shape )
                raise
            except:
                print( ems )
                print( super(XLine, self).__call__(self.fwhm, ems) )
                raise

        return super(XLine, self).__call__(self.fwhm, ems)


if __name__=='__main__':

    import numpy as np

    cwl = 3967 # e-9
    waves = np.linspace(3950, 3990, 4000)  # * 1e-9

    atomic_mass = 2

    line = HydrogenLineShape(cwl, waves, 7, 2, atomic_mass) # , False)

    from tulasa import general

    new = [5e13, 10, 10, 0, 0.5]
    old = [5e19, 10,     0, 90]

    peaks = [line(new), line.comparison(old), line.comparison(old, 'stehle'),
                        line.comparison(old, 'voigt')] #, line.comparison(old, 'rosato')]

    # t0 = clock.time()
    #
    # for tmp in np.arange(1000):
    #     # line(new)
    #     line.comparison(old)
    #
    # t1 = clock.time()
    #
    # print('Iteration time', (t1-t0)/1e3)

    general.plot(peaks, [line.wavelengths for p in peaks], multi='fake')


    pass

# else:
#
#     import os, sys, io
#
#     sys.path.append(os.path.expanduser('~/'))
#
#     from numpy import arange, random
#
#     from BaySAR.plasmas import PlasmaLine
#     from BaySAR.lineshapes import GaussiansNorm, Gaussian
#     from BaySAR.line_data import line_data
#
#     from tulasa import general
#
#     line_info = line_data['D']['0']['3968.99']
#     # line_info = line_data['D']['0']['4100.58']
#
#     cwl = line_info['wavelength']
#     dtuning = 5.
#     wave_res = 0.18
#     wavelengths = np.linspace(cwl-dtuning, cwl+dtuning, int(2*dtuning/wave_res))
#     n_upper = line_info['n_upper']
#     atomic_mass = 2
#     pec = [ line_info['exc_tec'], line_info['rec_pec'] ]
#     species = 'D'
#     ion = '0'
#
#     plasma = {}
#
#     plasma['los'] = np.linspace(-5., 5., 20)
#
#     profiles = Gaussian(x=plasma['los'])
#
#     plasma['electron_density'] = profiles( [-1, 5, 5e14] )
#     plasma['main_ion_density'] = plasma['electron_density']
#     plasma['electron_temperature'] = profiles([0, 3, 5])
#     plasma['D'] = {'tau': 1e-3, 'conc': 1e13, 'ti': 1e1}
#
#     d_epsilon_line = BalmerHydrogenLine(cwl, wavelengths, n_upper, atomic_mass, pec,
#                               species, ion, plasma)
#
#     line_info = line_data['D']['0']['4100.58']
#
#     cwl = line_info['wavelength']
#     dtuning = 5.
#     wave_res = 0.18
#     wavelengths = np.linspace(cwl-dtuning, cwl+dtuning, int(2*dtuning/wave_res))
#     n_upper = line_info['n_upper']
#     atomic_mass = 2
#     pec = [ line_info['exc_tec'], line_info['rec_pec'] ]
#     species = 'D'
#     ion = '0'
#
#     plasma = {}
#
#     plasma['los'] = np.linspace(-5., 5., 20)
#
#     profiles = Gaussian(x=plasma['los'])
#
#     plasma['electron_density'] = profiles( [-1, 5, 5e14] )
#     plasma['main_ion_density'] = plasma['electron_density']
#     plasma['electron_temperature'] = profiles([0, 3, 5])
#     plasma['D'] = {'tau': 1e-6, 'conc': 1e13, 'ti': 1e1}
#
#     d_delta_line = BalmerHydrogenLine(cwl, wavelengths, n_upper, atomic_mass, pec,
#                               species, ion, plasma)
#
#
#     general.plot([sum(d_delta_line())+1e12, sum(d_epsilon_line())+1e12], log=True, multi='fake')
#
#     print(  sum(sum(d_epsilon_line())) / sum(sum(d_delta_line())) )
#
#
#     pass