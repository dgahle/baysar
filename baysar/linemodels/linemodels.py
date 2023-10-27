"""
.. moduleauther:: Daljeet Singh Gahle <daljeet.gahle@strath.ac.uk>
"""


from copy import copy
import time as clock
import warnings

import numpy as np
from numpy import dot, round
from numpy import trapz
from numpy import empty, log, nan, ndarray, power
from numpy import (
    arange,
    array,
    cos,
    diff,
    dot,
    interp,
    isinf,
    linspace,
    log10,
    nan_to_num,
    sin,
    sqrt,
    trapz,
    where,
    zeros,
)
from scipy.constants import pi
from scipy.constants import speed_of_light
from scipy.constants import c as speed_of_light
from scipy.constants import e as electron_charge
from scipy.constants import m_e as electron_mass
from scipy.constants import physical_constants
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.io import readsav
from scipy.signal import fftconvolve
from xarray import DataArray

from baysar.lineshapes import Gaussian, reduce_wavelength, put_in_iterable
from baysar.tools import clip_data
from OpenADAS import read_adf11

from .doppler import DopplerLine, doppler_shift, update_DopplerLine_cwls
from .tools import get_atomic_mass



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
        return wavelengths[lower_index : upper_index + 1], [lower_index, upper_index]
    else:
        return wavelengths[lower_index : upper_index + 1]


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
        tmp_data = readsav(filename.replace(" ", ""))
    except:
        raise

    tmp_te = np.log10(tmp_data["te"]).tolist()
    tmp_ne = np.log10(tmp_data["edens"]).tolist()
    tmp_tau = np.log10(tmp_data["tau"]).tolist()

    tmp_tec_grid0 = tmp_data["tec_grid"]
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
        tmp_data = readsav(filename.replace(" ", ""))
    except:
        raise

    tmp_te = tmp_data["te"]
    tmp_ne = tmp_data["edens"]

    tmp_tec_grid = tmp_data["pecs"][:, :, reaction_key]

    # print( tmp_ne.shape, tmp_te.shape, tmp_tec_grid.shape )

    return RectBivariateSpline(tmp_ne, tmp_te, tmp_tec_grid.T)


def reshape_tec_grid(tec_grid):
    """
    Takes the 3D ne*TEC Tensor and rearranges the dimentions into an appropriate structure for
    making the 3D interpolator using build_pec()

    :param tec_grid: 3D ne*TEC Tensor
    :return: restructured 3D ne*TEC Tensor
    """

    if type(tec_grid) == list:
        tec_grid = np.array(tec_grid)

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





def estimate_XLine(l, wave, ems, half_width=None, half_range=0.5):
    """
    Esimates the area of a mystery line (XLine) and gives bounds.

    return estimate, bounds
    """

    if half_width is None:
        half_width = 2 * diff(wave).mean()

    estimate_list = []
    for cwl in l.line.cwl:
        tmp_range = [(cwl - half_width), (cwl + half_width)]
        newx, newy = clip_data(wave, ems, tmp_range)
        estimate_list.append(trapz(newy, newx))

    estimate = log10(sum(estimate_list))
    bounds = [estimate - 2 * half_range, estimate + half_range]
    return estimate, bounds


class XLine(object):
    def __init__(
        self, cwl, wavelengths, plasma, fractions, fwhm=0.2, species="X", half_range=5
    ):
        self.plasma = plasma
        self.species = species
        self.fwhm = fwhm
        self.line_tag = self.species
        for c in cwl:
            self.line_tag += "_" + str(c)

        self.line = Gaussian(
            x=wavelengths, cwl=cwl, fwhm=fwhm, fractions=fractions, normalise=True
        )

    def __call__(self):
        self.emission_fitted = self.plasma.plasma_state[self.line_tag].mean()
        return self.line(self.emission_fitted)

    def estimate_ems_and_bounds(self, wavelength, spectra):
        self.estimate, self.bounds = estimate_XLine(self, wavelength, spectra)








def doppler_shift_ADAS406Lines(self, velocity):
    # get doppler shifted wavelengths
    cwls = np.array(copy(self.cwls))
    new_cwl = doppler_shift(cwls, self.atomic_mass, velocity)
    # update wavelengths
    update_DopplerLine_cwls(self.linefunction, new_cwl)


def ti_to_fwhm(cwl, atomic_mass, ti):
    return cwl * 7.715e-5 * sqrt(ti / atomic_mass)






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
        """
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
        """

        self.plasma = plasma
        self.species = species
        self.atomic_mass = get_atomic_mass(species)
        self.cwls = put_in_iterable(cwls)
        self.line = self.species + "_" + str(self.cwls)[1:-1].replace(", ", "_")
        # self.cwls=[cw-0.1955 for cw in self.cwls]
        self.jj_frac_full = plasma.input_dict[self.species][cwls]["jj_frac"]
        self.wavelengths = wavelengths
        self.los = self.plasma.profile_function.electron_density.x

        self.lines = []
        self.jj_frac = []

        for tmp_cwl, frac in zip(self.cwls, self.jj_frac_full):
            out_of_bounds = (tmp_cwl < min(self.wavelengths)) or (
                tmp_cwl > max(self.wavelengths)
            )
            if not out_of_bounds:
                self.lines.append(tmp_cwl)
                self.jj_frac.append(frac)

        self.linefunction = DopplerLine(
            self.lines,
            self.wavelengths,
            atomic_mass=self.atomic_mass,
            fractions=self.jj_frac,
            half_range=half_range,
        )
        self.tec406 = self.plasma.impurity_tecs[self.line]

        length = diff(self.los)[0]
        self.length_per_sr = length / (4 * pi)
        len_los = len(self.los)
        self.tec_in = np.zeros((len_los, 3))

    def __call__(self):
        n0 = self.plasma.plasma_state[self.species + "_dens"].copy()[0]
        ne = self.plasma.plasma_state["electron_density"].flatten()
        te = self.plasma.plasma_state["electron_temperature"].flatten()
        tau = self.plasma.plasma_state[self.species + "_tau"][0]

        if self.plasma.concstar:
            p = ne / ne.max()
            n0 = n0 * p

        self.n0 = n0

        if hasattr(self.plasma, "ne_tau"):
            if self.plasma.ne_tau:  # in 1e13cm-3ms
                tau = 1e3 * tau / (ne * 1e-13)
        if self.species + "_velocity" in self.plasma.plasma_state:
            self.velocity = self.plasma.plasma_state[self.species + "_velocity"]
            doppler_shift_ADAS406Lines(self, self.velocity)

        _ = self.get_tec(ne, te, tau, star=True)

        self.emission_profile = (n0 * self.tec).clip(
            1
        )  # ph/cm-3/s # why do negatives occur here?
        self.emission_fitted = np.trapz(self.emission_profile, x=self.plasma.los) / (
            4 * pi
        )  # ph/cm-2/sr/s

        self.ems_ne = dot(self.emission_profile, ne) / self.emission_profile.sum()
        self.ems_conc = (
            dot(self.emission_profile, n0 / ne) / self.emission_profile.sum()
        )
        # self.ems_conc = n0 / self.ems_ne
        self.ems_te = dot(self.emission_profile, te) / self.emission_profile.sum()

        if self.plasma.cold_ions:
            self.plasma.plasma_state[self.species + "_Ti"] = 0.01
        elif self.plasma.thermalised:
            self.plasma.plasma_state[self.species + "_Ti"] = self.ems_te

        self.ti = self.plasma.plasma_state[self.species + "_Ti"]

        peak = self.linefunction(self.ti, self.emission_fitted)  # ph/cm-2/A/sr/s

        if any(np.isnan(peak)):
            raise TypeError(
                "NaNs in peaks of {} (tau={}, ems_sum={})".format(
                    self.line, np.log10(tau), self.emission_fitted
                )
            )
        if any(np.isinf(peak)):
            raise TypeError(
                "infs in peaks of {} (tau={}, ems_sum={}))".format(
                    self.line, np.log10(tau), self.emission_fitted
                )
            )

        return peak

    def get_tec(self, ne, te, tau, grad=0, star=False):
        # TODO: comparison to weight on log(tau) vs tau
        adas_taus = self.plasma.adas_plasma_inputs[
            "magical_tau"
        ]  # needs line below to!!!
        j = adas_taus.searchsorted(np.log10(tau))
        i = j - 1

        # index checks:
        if any([not c < len(adas_taus) for c in (i, j)]):
            things = [
                i,
                j,
                np.round(np.log10(tau), 2),
                len(adas_taus),
                adas_taus.min(),
                adas_taus.max(),
            ]
            raise ValueError(
                "Indices i and/or j ({}, {}) are out of bounds. tau = 1e{} s and len(adas_taus) = {} with range ({}, {})".format(
                    *things
                )
            )

        self.da_taus = [adas_taus[i], np.log10(tau), adas_taus[j]]
        self.tec_weights = 1 + np.diff(self.da_taus) / (
            self.da_taus[0] - self.da_taus[-1]
        )
        i_weight, j_weight = self.tec_weights

        if False:
            if star:
                tec406_lhs = self.tec406[i].ev(ne, te)
                tec406_rhs = self.tec406[j].ev(ne, te)
            else:
                raise ValueError("Non concstar tec not yet implimented!")
        else:
            tec406_lhs = self.tec406[i].ev(ne, te)
            tec406_rhs = self.tec406[j].ev(ne, te)

        self.tec406_lhs = tec406_lhs
        self.tec406_rhs = tec406_rhs

        # self.tec406_wieghted=tec406_lhs*i_weight + tec406_rhs*j_weight
        # self.tec=np.nan_to_num( np.exp(self.tec406_wieghted) ) # todo - why? and fix!

        self.tec406_wieghted = (
            np.exp(tec406_lhs) * i_weight + np.exp(tec406_rhs) * j_weight
        )
        self.tec = np.nan_to_num(self.tec406_wieghted)  # todo - why? and fix!

        if grad:
            # raise ValueError(f"Gradient method yet to be implimented!")
            # calculate the d/dne
            grad_ne_tec406_lhs = self.tec406[i].ev(ne, te, dx=grad, dy=0)
            grad_ne_tec406_rhs = self.tec406[j].ev(ne, te, dx=grad, dy=0)
            # grad_ne = np.exp(grad_ne_tec406_lhs)*i_weight + np.exp(grad_ne_tec406_rhs)*j_weight
            grad_ne = grad_ne_tec406_lhs * i_weight + grad_ne_tec406_rhs * j_weight
            # calculate the d/dTe
            # grad_te_tec406_lhs = self.tec406[i].ev(ne, te, dx=0, dy=grad)
            # grad_te_tec406_rhs = self.tec406[j].ev(ne, te, dx=0, dy=grad)
            grad_te_tec406_lhs = self.tec406[i].ev(ne, te, dx=0, dy=grad)
            grad_te_tec406_rhs = self.tec406[j].ev(ne, te, dx=0, dy=grad)
            # grad_te = np.exp(grad_te_tec406_lhs)*i_weight + np.exp(grad_te_tec406_rhs)*j_weight
            grad_te = grad_te_tec406_lhs * i_weight + grad_te_tec406_rhs * j_weight
            # calculate the d/dlogtau
            grad_logtau = (np.exp(tec406_rhs) - np.exp(tec406_lhs)) / (
                adas_taus[j] - adas_taus[i]
            )

            return self.tec, np.array([grad_ne, grad_te, grad_logtau])
        else:
            return self.tec


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







def print_ADAS406Line_summary(line):
    print(
        f"{line.species}: {round(line.ems_te, 2)} eV, {round(line.ems_ne/1e14, 2)} 1e14 cm-3, {round(1e2*line.ems_conc, 2)} %"
    )


def print_BalmerLine_summary(line):
    print(
        f"{line.species} exc: {round(line.exc_te, 2)} eV, {round(line.exc_ne/1e14, 2)} 1e14 cm-3, {round(1e2*line.exc_n0_frac, 3)} %"
    )
    print(
        f"{line.species} rec: {round(line.rec_te, 2)} eV, {round(line.rec_ne/1e14, 2)} 1e14 cm-3, {round(1e2*line.f_rec, 2)} %"
    )


def weighted_line_ratio(chord, upper, lower):
    weights = (
        chord.lines[upper].emission_profile / chord.lines[upper].emission_profile.sum()
    )
    ratio = chord.lines[upper].emission_profile / chord.lines[lower].emission_profile
    return dot(weights, ratio)


if __name__ == "__main__":
    pass
