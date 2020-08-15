"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""


from scipy.io import readsav
from scipy.constants import pi
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

from scipy.signal import fftconvolve

import warnings

import time as clock

import numpy as np
from numpy import sqrt, linspace, diff, arange, zeros, where, nan_to_num, array, log10, trapz, sin, cos, interp, dot

from baysar.lineshapes import Gaussian, reduce_wavelength


class XLine(object):
    def __init__(self, cwl, wavelengths, plasma, fractions, fwhm=0.2, species='X', half_range=5):
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

    def estimate_log_ems(self, wavelength, spectra):
        log_ems_estimate=0
        half_width=0.3
        for cwl in self.line.cwl:
            condition = ((cwl-half_width) < wavelength) and (wavelength < (cwl-half_width))
            index = np.where(condition)[0]
            log_ems_estimate += spectra[index].sum()
        self.log_ems_estimate = np.log10(log_ems_estimate)


from scipy.constants import speed_of_light
def doppler_shift(cwl, atomic_mass, velocity):
    f_velocity=velocity/speed_of_light
    return cwl*(f_velocity+1)

def update_DopplerLine_cwls(line, cwls):
    if not len(cwls)==len(line.line.cwl):
        error_string='Transition has {} line and {} where given'.format(len(line.line.cwl), len(cwls))
        raise ValueError(error_string)

    line.line.cwl=np.array(cwls)

def doppler_shift_ADAS406Lines(self,  velocity):
    # get doppler shifted wavelengths
    cwls=np.array(copy(self.cwls))
    new_cwl=doppler_shift(cwls, self.atomic_mass, velocity)
    # update wavelengths
    update_DopplerLine_cwls(self.linefunction, new_cwl)

def update_HydrogenLineShape_cwl(line, cwl):
    old_cwl=copy(line.cwl)
    # update cwl for stark and zeeman
    line.cwl=cwl
    # update for DopplerLine
    update_DopplerLine_cwls(line.doppler_function, [cwl])
    # shift the DopplerLine wavelengths so the peak remains centred for the convolution
    line.doppler_function.line.reducedx[0]+=(cwl-old_cwl)

def doppler_shift_BalmerHydrogenLine(self, velocity):
    # get doppler shifted wavelengths
    new_cwl=doppler_shift(copy(self.cwl), self.atomic_mass, velocity)
    # update wavelengths
    update_HydrogenLineShape_cwl(self.lineshape, new_cwl)

def ti_to_fwhm(cwl, atomic_mass, ti):
    return cwl * 7.715e-5 * sqrt(ti / atomic_mass)


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
        fwhm=self.line.cwl*7.715e-5*sqrt(ti/self.atomic_mass)
        # print(self.line.cwl, self.atomic_mass, fwhm, ti)
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
        # self.cwls=[cw-0.1955 for cw in self.cwls]
        self.jj_frac_full = plasma.input_dict[self.species][cwls]['jj_frac']
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

        self.linefunction = DopplerLine(self.lines, self.wavelengths, atomic_mass=self.atomic_mass,
                                        fractions=self.jj_frac, half_range=half_range)
        self.tec406 = self.plasma.impurity_tecs[self.line]

        length = diff(self.los)[0]
        self.length_per_sr = length / (4 * pi)
        len_los=len(self.los)
        self.tec_in=np.zeros((len_los, 3))

    def __call__(self):

        n0 = self.plasma.plasma_state[self.species+'_dens']
        ne = self.plasma.plasma_state['electron_density'].flatten()
        te = self.plasma.plasma_state['electron_temperature'].flatten()
        tau = self.plasma.plasma_state[self.species+'_tau'][0]
        # tau = np.zeros(len(ne))+self.plasma.plasma_state[self.species+'_tau'][0]
        if self.species+'_velocity' in self.plasma.plasma_state:
            self.velocity=self.plasma.plasma_state[self.species+'_velocity']
            doppler_shift_ADAS406Lines(self, self.velocity)

        # self.tec_in[:, 0] = self.plasma.plasma_state[self.species+'_tau'][0]# np.array([tau, ne, te])
        # self.tec_in[:, 1] = self.plasma.plasma_state['electron_density']
        # self.tec_in[:, 2] = self.plasma.plasma_state['electron_temperature']
        # TODO: comparison to weight on log(tau) vs tau
        adas_taus = self.plasma.adas_plasma_inputs['magical_tau'] # needs line below to!!!
        j=adas_taus.searchsorted(np.log10(tau))
        # adas_taus = self.plasma.adas_plasma_inputs['tau_exc']
        # j=adas_taus.searchsorted(tau)
        i=j-1

        # index checks:
        if any([not c < len(adas_taus) for c in (i, j)]):
            things=[i, j, np.round(np.log10(tau), 2), len(adas_taus), adas_taus.min(), adas_taus.max()]
            raise ValueError("Indices i and/or j ({}, {}) are out of bounds. tau = 1e{} s and len(adas_taus) = {} with range ({}, {})".format(*things))

        d_tau=1/(adas_taus[j]-adas_taus[i])
        i_weight = (tau-adas_taus[i])*d_tau
        j_weight = (adas_taus[j]-tau)*d_tau

        # tec406_lhs = self.tec406[i](self.tec_in[:, 1], self.tec_in[:, 2])
        # tec406_rhs = self.tec406[j](self.tec_in[:, 1], self.tec_in[:, 2])
        tec406_lhs = self.tec406[i].ev(ne, te)
        tec406_rhs = self.tec406[j].ev(ne, te)
        self.tec406_lhs=tec406_lhs
        self.tec406_lhs=tec406_lhs
        # tec = np.exp(tec406_lhs*i_weight+tec406_rhs*j_weight)
        tec = np.exp(tec406_lhs)*i_weight + np.exp(tec406_rhs)*j_weight
        tec = np.nan_to_num(tec) # todo - why? and fix!
        self.tec=tec

        ems = n0 * tec * self.length_per_sr
        ems = ems.clip(min=1)
        ems_sum = ems.sum()

        self.emission_profile = ems

        self.ems_ne = dot(ems, ne) / ems_sum
        self.ems_conc = n0 / self.ems_ne
        self.ems_te = dot(ems, te) / ems_sum

        if self.plasma.thermalised:
            self.ti=self.ems_te
        else:
            self.ti=self.plasma.plasma_state[self.species+'_Ti']


        peak=self.linefunction(self.ti, ems_sum)
        if any(np.isnan(peak)):
            raise TypeError('NaNs in peaks of {} (tau={}, ems_sum={})'.format(self.line, np.log10(tau), ems_sum))
        if any(np.isinf(peak)):
            raise TypeError('infs in peaks of {} (tau={}, ems_sum={}))'.format(self.line, np.log10(tau), ems_sum))

        return peak


# def comparison(self, theta, line_model='stehle_param'):
#
#     from pystark import BalmerLineshape, ls_norm
#
#     electron_density, ion_temperature, b_field, viewangle = theta
#
#     line = BalmerLineshape(self.n_upper, electron_density, ion_temperature, bfield=b_field,
#                            viewangle=viewangle, line_model=line_model,
#                            wl_axis=self.wavelengths / 1e10, wl_centre=self.cwl / 1e10,
#                            override_input_check=True)
#
#      return line.ls_szd / trapz(line.ls_szd, self.wavelengths)

loman_coeff={'32': [0.7665, 0.064, 3.710e-18],  # Balmer Series
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

def stehle_param(n_upper, n_lower, cwl, wavelengths, electron_density, electron_temperature):
    # Paramaterised MMM Stark profile coefficients from Bart's paper
    a_ij, b_ij, c_ij = loman_coeff[str(n_upper) + str(n_lower)]
    delta_lambda_12ij = c_ij*np.divide( (1e6*electron_density)**a_ij, electron_temperature**b_ij)  # nm
    ls_s = 1 / (abs((wavelengths - cwl))**(2.5) + (5 * delta_lambda_12ij)**(2.5))
    return ls_s / trapz(ls_s, wavelengths)



from scipy.constants import e as electron_charge
from scipy.constants import m_e as electron_mass
from scipy.constants import c as speed_of_light

# cf is central frequency
b_field_to_cf_shift = electron_charge / (4 * pi *electron_mass * speed_of_light*1e10)

def zeeman_split(cwl, peak, wavelengths, b_field, viewangle):
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
    freq_shift_sigma = b_field_to_cf_shift * b_field
    wave_shift_sigma = abs(cwl - cwl / (1 - cwl*freq_shift_sigma))

    # relative intensities normalised to sum to one
    ls_sigma_minus = rel_intensity_sigma * interp(wavelengths + wave_shift_sigma, wavelengths, peak)
    ls_sigma_plus = rel_intensity_sigma * interp(wavelengths - wave_shift_sigma, wavelengths, peak)
    ls_pi = rel_intensity_pi * peak
    return ls_sigma_minus + ls_pi + ls_sigma_plus

from copy import copy

class HydrogenLineShape(object):
    def __init__(self, cwl, wavelengths, n_upper, n_lower, atomic_mass, zeeman=True):
        self.cwl = cwl
        self.wavelengths = wavelengths
        self.zeeman = zeeman
        if int(n_lower)==1:
            self.n_upper=n_upper+1
            self.n_lower=2
            UserWarning("Using the Balmer Stark shape coefficients for the Lyman series. Transition n=%d -> %d | %f A"%(self.n_upper, self.n_lower, np.round(self.cwl, 2)))
        else:
            self.n_upper=n_upper
            self.n_lower=n_lower

        # print("Transition n=%d -> %d | %f A"%(self.n_upper, self.n_lower, np.round(self.cwl, 2)))

        # # TODO - why?
        # wavelengths_doppler_num = len(self.wavelengths) # todo make a power of 2
        # if wavelengths_doppler_num % 2 == 0:
        #     wavelengths_doppler_num += 1

        self.wavelengths_doppler = arange(self.cwl-10, self.cwl+10, np.diff(self.wavelengths)[0])
        self.doppler_function = DopplerLine(cwl=copy(self.cwl), wavelengths=self.wavelengths_doppler, atomic_mass=atomic_mass, half_range=5000)

    def __call__(self, theta):
        if self.zeeman:
            electron_density, electron_temperature, ion_temperature, b_field, viewangle = theta
        else:
            electron_density, electron_temperature, ion_temperature = theta

        doppler_component = self.doppler_function(ion_temperature, 1)
        if self.zeeman:
            doppler_component=zeeman_split(self.cwl, doppler_component, self.wavelengths_doppler, b_field, viewangle)

        stark_component = stehle_param(self.n_upper, self.n_lower, self.cwl, self.wavelengths, electron_density, electron_temperature)

        # peak=np.convolve(stark_component, doppler_component/doppler_component.sum(), 'same')
        peak=fftconvolve(stark_component, doppler_component, 'same')
        peak/=trapz(peak, self.wavelengths)

        return peak


from adas import read_adf11
class BalmerHydrogenLine(object):
    def __init__(self, plasma, species, cwl, wavelengths, half_range=40000, zeeman=True):
        self.plasma = plasma
        self.species = species
        self.cwl = cwl
        self.line = self.species+'_'+str(self.cwl)
        self.wavelengths = wavelengths
        self.n_upper = self.plasma.input_dict[self.species][self.cwl]['n_upper']
        self.n_lower = self.plasma.input_dict[self.species][self.cwl]['n_lower']
        self.atomic_mass = get_atomic_mass(self.species)
        self.los = self.plasma.profile_function.electron_density.x
        self.dl_per_sr = diff(self.los)[0] / (4*pi)

        # self.reduced_wavelength, self.reduced_wavelength_indicies = \
        #     reduce_wavelength(wavelengths, cwl, half_range, return_indicies=True, power2=False)

        self.len_wavelengths = len(self.wavelengths)
        self.exc_pec = self.plasma.hydrogen_pecs[self.line+'_exc']
        self.rec_pec = self.plasma.hydrogen_pecs[self.line+'_rec']

        # self.lineshape = HydrogenLineShape(self.cwl, self.reduced_wavelength, self.n_upper, n_lower=self.n_lower,
        #                                    atomic_mass=self.atomic_mass, zeeman=zeeman)
        self.lineshape = HydrogenLineShape(self.cwl, self.wavelengths, self.n_upper, n_lower=self.n_lower,
                                           atomic_mass=self.atomic_mass, zeeman=zeeman)


    def __call__(self):
        n1 = self.plasma.plasma_state['main_ion_density']
        ne = self.plasma.plasma_state['electron_density']
        te = self.plasma.plasma_state['electron_temperature']
        n0 = self.plasma.plasma_state[self.species+'_dens'][0]

        self.n0_profile=n0

        if self.plasma.zeeman:
            bfield = self.plasma.plasma_state['b-field']
            viewangle = self.plasma.plasma_state['viewangle']
        else:
            bfield=0
            viewangle=0
        if self.species+'_velocity' in self.plasma.plasma_state:
            self.velocity=self.plasma.plasma_state[self.species+'_velocity']
            doppler_shift_BalmerHydrogenLine(self, self.velocity)

        rec_pec = np.exp(self.rec_pec(ne, te))
        exc_pec = np.exp(self.exc_pec(ne, te))
        # set minimum number of photons to be 1
        self.rec_profile = (n1*ne*self.dl_per_sr*rec_pec).clip(1.)
        self.exc_profile = (n0*ne*self.dl_per_sr*nan_to_num(exc_pec)).clip(1.)

        rec_sum = self.rec_profile.sum()
        exc_sum = self.exc_profile.sum()
        ems_sum = rec_sum + exc_sum

        # used for the emission lineshape calculation
        self.ems_profile = self.rec_profile + self.exc_profile

        self.exc_ne = dot(self.exc_profile, ne) / exc_sum
        self.exc_te = dot(self.exc_profile, te) / exc_sum

        self.rec_ne = dot(self.rec_profile, ne) / rec_sum
        self.rec_te = dot(self.rec_profile, te) / rec_sum

        # just because there are nice to have
        self.f_rec = rec_sum / ems_sum
        self.ems_ne = dot(self.ems_profile, ne) / ems_sum
        self.ems_te = dot(self.ems_profile, te) / ems_sum

        if self.plasma.thermalised:
            self.exc_ti=self.exc_te
            self.rec_ti=self.rec_te
        else:
            self.exc_ti=self.plasma.plasma_state[self.species+'_Ti']
            self.rec_ti=self.plasma.plasma_state[self.species+'_Ti']


        self.exc_lineshape_input = [self.exc_ne, self.exc_te, self.exc_ti, bfield, viewangle]
        self.rec_lineshape_input = [self.rec_ne, self.rec_te, self.rec_ti, bfield, viewangle]
        self.exc_peak = nan_to_num( self.lineshape(self.exc_lineshape_input) )
        self.rec_peak = nan_to_num( self.lineshape(self.rec_lineshape_input) )

        return self.rec_peak*rec_sum + self.exc_peak*exc_sum

        # tmpline=DopplerLine(self.cwl, wavelengths=self.wavelengths, atomic_mass=2, fractions=None, half_range=50)
        # return tmpline(ti, 1)*(rec_sum + exc_sum)




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

    cwl=4100
    wavelengths=np.linspace(4090, 4110, 100)
    n_lower=2
    atomic_mass=2
    peaks=[]
    for n_upper in (3, 4, 5, 6, 7, 8, 9):
        bline=HydrogenLineShape(cwl, wavelengths, n_upper, n_lower, atomic_mass, zeeman=True)
        # electron_density, electron_temperature, ion_temperature, b_field, viewangle
        peaks.append(bline([5e14, 5, 5, 0, 0]))
    plot(peaks, log=True, multi='fake')
