"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""


from scipy.io import readsav
from scipy.constants import pi
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

from scipy.signal import fftconvolve
import scipy.constants

import numpy as np

import warnings

import time as clock

from numpy import sqrt, linspace, diff, arange, zeros, where, nan_to_num, array, log10, trapz, sin, cos, interp

from baysar.lineshapes import Gaussian, reduce_wavelength


class XLine(object):

    def __init__(self, cwl, wavelengths, plasma, fractions, fwhm=0.3, species='X', half_range=5):
        self.plasma = plasma
        self.species = species
        self.fwhm = fwhm
        self.line_tag = self.species
        for c in cwl:
            self.line_tag += '_' + str(c)

        self.line = Gaussian(x=wavelengths, cwl=cwl, fwhm=fwhm, fractions=fractions, normalise=True)

    def __call__(self):
        ems = self.plasma.plasma_state[self.line_tag]
        return self.line(ems)

def ti_to_fwhm(cwl, atomic_mass, ti):
    tmp = 7.715e-5 * sqrt(ti / atomic_mass)
    return cwl * tmp # = fwhm

class DopplerLine(object):

    """
    This class is used to calculate Guassian lineshapes as a function of area and ion
    temperature. It is inherited by NoADASLine, Xline and ADAS406Line and used by
    NoADASLines and ADAS406Lines.
    """

    def __init__(self, cwl, wavelengths, atomic_mass, fractions=None, half_range=5):
        self.atomic_mass = atomic_mass
        self.line = Gaussian(x=wavelengths, cwl=cwl, fwhm=None, fractions=fractions, normalise=True)

    def __call__(self, ti, ems):
        fwhm = [ti_to_fwhm(cwl, self.atomic_mass, ti) for cwl in self.line.cwl]
        return self.line([fwhm, ems])


from baysar.lineshapes import put_in_iterable

elements = ['H', 'D', 'He', 'Be', 'B', 'C', 'N', 'O', 'Ne']
masses = [1, 2, 4, 9, 10.8, 12, 14, 16, 20.2]
elements_and_masses = []
for elm, mass in zip(elements, masses):
    elements_and_masses.append((elm, mass))
atomic_masses = dict(elements_and_masses)

def species_to_element(species):
    return species[0:np.where([a=='_' for a in species])[0][0]]

def get_atomic_mass(species):
    elm = species_to_element(species)
    return atomic_masses[elm]

class ADAS406Lines(object):

    """
    This class is used to calculate Gaussian lineshapes as a function of impurity concentration,
    ion temperature, emission length, ion confinement time, electron temperature and density. The
    units returned are in spectra radiance (ph cm^-3 A^-1 s^-1).

    This class can be used to model multiplets not just singlets. In the case of multiplets one
    ne*TEC(tau, ne, Te) is used for the LS transition and the fraction of the emission for each
    each jj transition is scaled by a fixed scalar (ne*TEC(tau, ne, Te)*f_jj).
    """

    def __init__(self, plasma, species, cwls, wavelengths, half_range=10):

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

        self.plasma = plasma
        self.species = species
        self.atomic_mass = get_atomic_mass(species)
        self.cwls = put_in_iterable(cwls)
        self.line = self.species+'_'+str(self.cwls)[1:-1].replace(', ', '_')
        self.jj_frac_full = plasma.input_dict[self.species][self.cwls]['jj_frac']
        self.wavelengths = wavelengths
        self.los = self.plasma.profile_function.electron_density.x

        self.lines = []
        self.jj_frac = []

        for tmp_cwl, frac in zip(self.cwls, self.jj_frac_full):
            out_of_bounds = ( (tmp_cwl < min(self.wavelengths)) or
                              (tmp_cwl > max(self.wavelengths))     )
            if not out_of_bounds:
                self.lines.append(tmp_cwl)
                self.jj_frac.append(frac)

        self.linefunction = DopplerLine(self.lines, self.wavelengths, atomic_mass=self.atomic_mass, fractions=self.jj_frac, half_range=5)
        self.tec406 = self.plasma.impurity_tecs[self.line]

    def __call__(self):

        n0 = self.plasma.plasma_state[self.species+'_dens']
        ti = self.plasma.plasma_state[self.species+'_Ti']
        ne = self.plasma.plasma_state['electron_density']
        te = self.plasma.plasma_state['electron_temperature']
        tau = np.array([self.plasma.plasma_state[self.species+'_tau'][0] for n in ne])

        tec_in = (tau, ne, te)
        tec = self.tec406(tec_in)
        tec = np.power(10, tec)
        tec = np.nan_to_num(tec)

        length = diff(self.los)[0]
        length_per_sr = length / (4 * pi)

        ems = n0 * tec * length_per_sr
        ems = ems.clip(min=1e-20)

        self.emission_profile = ems
        self.ems_ne = sum(ems*ne) / sum(ems)
        self.ems_conc = n0 / self.ems_ne
        self.ems_te = sum(ems*te) / sum(ems)

        return self.linefunction(ti, sum(ems))


class HydrogenLineShape(object):

    def __init__(self, cwl, wavelengths, n_upper, n_lower, atomic_mass, zeeman=True):

        self.cwl = cwl
        self.wavelengths = wavelengths

        self.zeeman = zeeman

        wavelengths_doppler_num = len(self.wavelengths)

        if type(wavelengths_doppler_num/2) != int:
            wavelengths_doppler_num += 1

        self.wavelengths_doppler = linspace(self.cwl-10, self.cwl+10, wavelengths_doppler_num)

        self.doppler_function = DopplerLine(cwl=self.cwl, wavelengths=self.wavelengths_doppler, atomic_mass=atomic_mass, half_range=50)

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
        peak /= trapz(peak, self.wavelengths)

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

        ls_s /= trapz(ls_s, self.wavelengths)

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

        viewangle *= pi

        rel_intensity_pi = 0.5 * sin(viewangle) ** 2
        rel_intensity_sigma = 0.25 * (1 + cos(viewangle) ** 2)

        freq_shift_sigma = self.electron_charge / (4 * pi * self.electron_mass) * b_field

        # wave_shift_sigma = self.delta_magnetic_quantum_number * self.bohr_magnaton * b_field
        # wave_shift_sigma = self.delta_magnetic_quantum_number * 0.5 * b_field
        wave_shift_sigma = abs(self.cwl - 1 / (1/(self.cwl) - freq_shift_sigma / 299792458.0e10))


        # print(freq_shift_sigma, wave_shift_sigma)

        # relative intensities normalised to sum to one

        ls_sigma_minus = rel_intensity_sigma * interp(self.wavelengths + wave_shift_sigma, self.wavelengths, peak)
        ls_sigma_plus = rel_intensity_sigma * interp(self.wavelengths - wave_shift_sigma, self.wavelengths, peak)
        ls_pi = rel_intensity_pi * peak

        return ls_sigma_minus + ls_pi + ls_sigma_plus

    def get_delta_magnetic_quantum_number(self):

        upper_m = 1
        lower_m = 0

        self.delta_magnetic_quantum_number = upper_m - lower_m

    def comparison(self, theta, line_model='stehle_param'):

        from pystark import BalmerLineshape, ls_norm

        electron_density, ion_temperature, b_field, viewangle = theta

        line = BalmerLineshape(self.n_upper, electron_density, ion_temperature, bfield=b_field,
                               viewangle=viewangle, line_model=line_model,
                               wl_axis=self.wavelengths / 1e10, wl_centre=self.cwl / 1e10,
                               override_input_check=True)

        return line.ls_szd / trapz(line.ls_szd, self.wavelengths)


class BalmerHydrogenLine(object):

    def __init__(self, plasma, species, cwl, wavelengths, half_range=40, zeeman=True):

        self.plasma = plasma
        self.species = species
        self.cwl = cwl
        self.line = self.species+'_'+str(self.cwl)
        self.wavelengths = wavelengths
        self.n_upper = self.plasma.input_dict['D_0'][self.cwl]['n_upper']
        self.get_atomic_mass()
        self.los = self.plasma.profile_function.electron_density.x

        self.reduced_wavelength, self.reduced_wavelength_indicies = \
            reduce_wavelength(wavelengths, cwl, half_range, return_indicies=True)

        self.zeros_peak = zeros(len(self.wavelengths))

        self.exc_pec = self.plasma.hydrogen_pecs[self.line+'_exc']
        self.rec_pec = self.plasma.hydrogen_pecs[self.line+'_rec']

        self.lineshape = HydrogenLineShape(self.cwl, self.reduced_wavelength, self.n_upper, n_lower=2,
                                           atomic_mass=self.atomic_mass, zeeman=zeeman)

    def __call__(self):

        n0 = self.plasma.plasma_state[self.species+'_dens']
        n1 = self.plasma.plasma_state['main_ion_density']
        ti = self.plasma.plasma_state[self.species+'_Ti']

        bfield = self.plasma.plasma_state['b-field']
        viewangle = self.plasma.plasma_state['viewangle']

        ne = self.plasma.plasma_state['electron_density']
        te = self.plasma.plasma_state['electron_temperature']

        length = diff(self.los)[0]
        length_per_sr = length / (4 * pi)

        rec_pec = np.power(10, self.rec_pec(ne, te))
        rec_profile = n1 * ne * length_per_sr * rec_pec

        exc_pec = np.power(10, self.exc_pec(ne, te))
        exc_profile = n0 * ne * length_per_sr * nan_to_num( exc_pec )
        # exc_profile = n0 * length_per_sr * nan_to_num( self.exc_tec(tec_in) )

        low_te = 0.2
        low_ne = 1e11

        low_te_indicies = where(te < low_te)
        low_ne_indicies = where(ne < low_ne)

        for indicies in [low_te_indicies, low_ne_indicies]:
            rec_profile[indicies] = 0.0
            exc_profile[indicies] = 0.0

        self.f_rec = sum(rec_profile) / sum(rec_profile + exc_profile)

        self.exc_profile = exc_profile
        self.rec_profile = rec_profile
        self.ems_profile = rec_profile + exc_profile

        rec_weighted_electron_density = sum( rec_profile * ne ) / sum( rec_profile )
        exc_weighted_electron_density = sum( exc_profile * ne ) / sum( exc_profile )

        rec_weighted_electron_temperature = sum( rec_profile * te ) / sum( rec_profile )
        exc_weighted_electron_temperature = sum( exc_profile * te ) / sum( exc_profile )

        ems_weighted_electron_density = sum( self.ems_profile * ne ) / sum( self.ems_profile )
        ems_weighted_electron_temperature = sum( self.ems_profile * te ) / sum( self.ems_profile )

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

        exc_peak = nan_to_num( self.lineshape(self.exc_lineshape_input) )
        rec_peak = nan_to_num( self.lineshape(self.rec_lineshape_input) )

        self.exc_peak = exc_peak
        self.rec_peak = rec_peak

        tmp_peak = rec_peak * sum(rec_profile) + exc_peak * sum(exc_profile)

        self.tmp_peak = tmp_peak

        peak = self.zeros_peak # zeros(len(self.wavelengths))
        peak[min(self.reduced_wavelength_indicies):max(self.reduced_wavelength_indicies)+1] = tmp_peak

        assert len(peak) == len(self.wavelengths), 'len(peak) != len(self.wavelengths)'

        return peak

    def get_atomic_mass(self):

        mass = {'H': 1, 'D': 2, 'T': 3}

        self.atomic_mass = mass[self.species[0]]


if __name__=='__main__':

    import numpy as np
    from baysar.input_functions import make_input_dict
    from baysar.plasmas import PlasmaLine
    from tulasa.general import plot

    num_chords = 1
    wavelength_axis = [np.linspace(3950, 4140, 512)]
    experimental_emission = [np.array([1e12*np.random.rand() for w in wavelength_axis[0]])]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ['D', 'N']
    ions = [ ['0'] , ['1'] ]
    noise_region = [[4040, 4050]]
    mystery_lines = [ [[4070], [4001, 4002]],
                      [[1], [0.4, 0.6]] ]

    input_dict = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                                instrument_function=instrument_function, emission_constant=emission_constant,
                                noise_region=noise_region, species=species, ions=ions,
                                mystery_lines=mystery_lines, refine=[0.01],
                                ion_resolved_temperatures=False, ion_resolved_tau=True)
    plasma = PlasmaLine(input_dict)

    rand_theta = [min(b)+np.random.rand()*(max(b)-min(b)) for b in plasma.theta_bounds]
    rand_theta[plasma.slices['D_0_Ti']][0] = 0
    plasma(rand_theta)

    ml_key = 1
    tmp_cwl = mystery_lines[0][ml_key]
    tmp_fractions = mystery_lines[1][ml_key]
    tmp_wavelengths = np.linspace(4000, 4100, 5120)
    species = 'X'

    xline = XLine(cwl=tmp_cwl, fwhm=10*diff(tmp_wavelengths)[0], wavelengths=tmp_wavelengths, plasma=plasma, fractions=tmp_fractions)
    xline()

    dopplerline = DopplerLine(tmp_cwl, tmp_wavelengths, atomic_mass=10, half_range=5)
    dopplerline(10, 10)

    species = 'N_1'
    cwls = (4026.09, 4039.35)
    line406 = ADAS406Lines(plasma, species, cwls, tmp_wavelengths)
    line406()

    species = 'D_0'
    cwl = 3968.99
    hline = BalmerHydrogenLine(plasma, species, cwl, tmp_wavelengths)
    hline()