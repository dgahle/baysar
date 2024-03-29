import collections
import copy
import io
import os
import sys
import warnings
from itertools import product

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from baysar.line_data import adas_line_data
from baysar.lineshapes import MeshLine
from baysar.tools import within
from gcr import ionisation_balance_transport
from OpenADAS import read_adf11, read_adf15


def power10(var):
    return np.power(10, var)


def check_bounds_order(bounds):
    check = [b[0] < b[1] for b in bounds]
    return all(check)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class arb_obj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, *args):
        return args


class arb_obj_single_input(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, theta):
        if type(theta) not in (float, int):
            theta = theta[0]
        return theta


class arb_obj_single_log_input(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if "power" not in self.__dict__:
            self.power = 1

    def __call__(self, theta):
        if type(theta) not in (float, int):
            theta = theta[0]
        return np.power(self.power, theta)


def calibrate(num_pixels, theta):
    cwl, dispersion = theta

    pixel_array = np.arange(num_pixels, dtype=float)
    pixel_array -= np.mean(pixel_array)
    pixel_array *= dispersion
    pixel_array += cwl

    return pixel_array


# from baysar.tools import calibrate
# instrument_function_calibrator.calibrate(int_func_cal_theta)
from baysar.lineshapes import gaussian_norm


class GaussianInstrumentFunctionCalibratior:
    def __init__(self, bounds=[[-1, 1]]):
        self.number_of_variables = 1
        self.bounds = bounds

    def __call__(self, theta):
        return [np.power(10, t) for t in theta]

    def calibrate(self, x, fwhm):
        instrument_function = gaussian_norm(x, x.mean(), fwhm, 1)
        return instrument_function


# tmp_func = arb_obj_single_input(number_of_variables=1, bounds=[-5, 5])
class default_calwave_function(object):
    def __init__(self, number_of_variables=2, bounds=[[1e2, 1e5], [0.01, 1.0]]):
        self.number_of_variables = number_of_variables
        self.bounds = bounds

        self.check_init()

    def check_init(self):
        if self.number_of_variables != len(self.bounds):
            raise ValueError("self.number_of_variables!=len(self.bounds)")

    def __call__(self, *args):
        return args[0]

    def calibrate(self, x, theta):
        return calibrate(len(x), theta)


class default_cal_function(object):
    def __init__(self, number_of_variables=1, bounds=[[-5, 20]]):
        self.number_of_variables = number_of_variables
        self.bounds = bounds

        self.check_init()

    def check_init(self):
        if self.number_of_variables != len(self.bounds):
            raise ValueError("self.number_of_variables!=len(self.bounds)")

    def __call__(self, *args):
        return np.power(10, args[0])

    # def calibrate(self, x, theta):
    #     m, c=theta
    #     return m*x+c

    def inverse_calibrate(self, y, theta):
        m = theta[0]
        return y / m


class default_background_function(object):
    def __init__(self, number_of_variables=1, bounds=[[-5, 20]]):
        self.number_of_variables = number_of_variables
        self.bounds = bounds

        self.check_init()

    def check_init(self):
        if self.number_of_variables != len(self.bounds):
            raise ValueError("self.number_of_variables!=len(self.bounds)")

    def __call__(self, *args):
        return np.power(10, args[0])

    def calculate_background(self, theta):
        return theta


atomic_number = {
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
}


def get_number_of_ions(element):
    return atomic_number[element] + 1


def get_meta(element, index=0):
    num = get_number_of_ions(element)
    meta = np.zeros(num)
    meta[index] = 1
    return meta


def kms_to_ms(velocity):
    return 1e3 * velocity


adf11_dir = "/home/adas/adas/adf11/"
hydrogen_adf11_plt = adf11_dir + "plt12/plt12_h.dat"
hydrogen_adf11_prb = adf11_dir + "prb12/prb12_h.dat"

from os.path import exists as file_exists


def get_adf11(elem, yr, type, adf11_dir=adf11_dir):
    adf11 = (
        adf11_dir + type + str(yr) + "/" + type + str(yr) + "_" + elem.lower() + ".dat"
    )
    if file_exists(adf11):
        return adf11
    else:
        raise FileNotFoundError(f"{adf11} is not found")


def radiated_power(
    n0, ni, ne, te, is1, adf11_plt=None, adf11_prb=None, elem="N", yr=96, all=False
):
    # get adf11 if not passed
    if adf11_plt is None:
        adf11_plt = get_adf11(elem, yr, type="plt")
    if adf11_prb is None:
        adf11_prb = get_adf11(elem, yr, type="prb")

    plt = read_adf11(
        file=adf11_plt,
        adf11type="plt",
        is1=is1,
        index_1=-1,
        index_2=-1,
        te=te,
        dens=ne,
        all=all,
    )  # , skipzero=False, unit_te='ev')
    prb = read_adf11(
        file=adf11_prb,
        adf11type="prb",
        is1=is1,
        index_1=-1,
        index_2=-1,
        te=te,
        dens=ne,
        all=all,
    )  # , skipzero=False, unit_te='ev')

    return (n0 * ne * plt, ni * ne * prb)


def print_plasma_state(plasma, posterior_components=None):
    # print headers
    print("ions |    dens    |    tau    |     v     |    Ti    |")
    print("     | 10^12 /cm3 |     ms    |    km/s   |    eV    |")
    print("---------------------------------------------------------------")
    # print neutral and ion data
    for species in plasma.species:
        if species[0] in plasma.hydrogen_isotopes:
            states = [species]
            states.append(
                np.round(plasma.plasma_state[species + "_dens"][0] * 1e-12, 2)
            )
            states.append("N/A")
            states.append(
                np.round(plasma.plasma_state[species + "_velocity"][0] * 1e-3, 2)
            )
            states.append(np.round(plasma.plasma_state[species + "_Ti"][0], 2))
            print(" {} |    {}    |   {}   |   {}   |   {}   | ".format(*states))
        else:
            states = [species]
            states.append(
                np.round(plasma.plasma_state[species + "_dens"][0] * 1e-12, 2)
            )
            states.append(np.round(plasma.plasma_state[species + "_tau"][0] * 1e3, 2))
            states.append(
                np.round(plasma.plasma_state[species + "_velocity"][0] * 1e-3, 2)
            )
            states.append(np.round(plasma.plasma_state[species + "_Ti"][0], 2))
            print(" {} |    {}    |   {}   |   {}   |   {}   | ".format(*states))

    print()
    # get peak Te and ne
    peak_ni = np.round(plasma.plasma_state["main_ion_density"].max() * 1e-12, 2)
    peak_ne = np.round(plasma.plasma_state["electron_density"].max() * 1e-12, 2)
    peak_te = np.round(plasma.plasma_state["electron_temperature"].max(), 2)
    peaks = [peak_te, peak_ne]
    # print peak Te and ne
    print("Peak Te {} eV".format(peak_te))
    print("Peak ne {} 10^12 /cm3".format(peak_ne))
    print("Peak ni {} 10^12 /cm3".format(peak_ni))

    if posterior_components is not None:
        print()
        for p in posterior_components:
            print(p, p())


class PlasmaLine:

    """
    PlasmaLine is a object that holds the simulated 1D plasma profiles and the function for updating the
    plamsa profiles. The plasma paramaters and profiles are also organised into a dictionary which are
    accessible to the line objects in the initialised spectrometer objects. This in essence is the parameter
    handler which handles the interface between the sampler and the forward model.
    """

    def __init__(
        self,
        input_dict,
        profile_function=None,
        cal_functions=None,
        calwave_functions=None,
        calint_func_functions=None,
        background_functions=None,
    ):
        "input_dict input has been checked if created with make_input_dict"

        print("Building PlasmaLine object")

        self.input_dict = input_dict
        self.num_chords = len(self.input_dict["wavelength_axis"])
        self.species = self.input_dict["species"]
        self.impurity_species = [
            s for s in self.species if not any(["H_" in s or "D_" in s or "T_" in s])
        ]
        self.hydrogen_species = [
            s for s in self.species if s not in self.impurity_species
        ]
        self.hydrogen_isotopes = ["H", "D", "T"]
        self.contains_hydrogen = self.species != self.impurity_species

        self.scdfile = "/home/adas/adas/adf11/scd12/scd12_h.dat"
        self.acdfile = "/home/adas/adas/adf11/acd12/acd12_h.dat"

        # tau_exc=np.logspace(-7, 1, 10)
        # tau_rec=np.logspace(1, -5, 18)
        # magical_tau=np.concatenate( ((np.log10(tau_exc)-np.log10(tau_exc).max()), (-np.log10(tau_rec)+2)) )
        tau_exc = np.logspace(-7, 2, 10 + 12)
        tau_rec = np.logspace(1, -4, 12 - 4)
        magical_tau = np.concatenate(((np.log10(tau_exc)), (-np.log10(tau_rec) + 4)))
        self.adas_plasma_inputs = {
            "te": np.logspace(-1, 2, 48),  # TODO can we get the raw data instead?
            "ne": np.logspace(12, 15, 15),
            "tau_exc": tau_exc,
            "tau_rec": tau_rec,
            "magical_tau": magical_tau,
        }
        self.adas_plasma_inputs["big_ne"] = np.array(
            [self.adas_plasma_inputs["ne"] for t in self.adas_plasma_inputs["te"]]
        ).T

        if "doppler_shifts" in self.input_dict:
            self.include_doppler_shifts = self.input_dict["doppler_shifts"]
        else:
            self.include_doppler_shifts = True

        if "calibrate_wavelength" in self.input_dict:
            self.calibrate_wavelength = self.input_dict["calibrate_wavelength"]
        else:
            self.calibrate_wavelength = False

        if "calibrate_intensity" in self.input_dict:
            self.calibrate_intensity = self.input_dict["calibrate_intensity"]
        else:
            self.calibrate_intensity = False

        if "calibrate_instrument_function" in self.input_dict:
            self.calibrate_instrument_function = self.input_dict[
                "calibrate_instrument_function"
            ]
        else:
            self.calibrate_instrument_function = False

        if "zeeman" in self.input_dict:
            self.zeeman = self.input_dict["zeeman"]
        else:
            self.zeeman = False

        if "no_sample_neutrals" in self.input_dict:
            self.no_sample_neutrals = self.input_dict["no_sample_neutrals"]
        else:
            self.no_sample_neutrals = True

        if "thermalised" in self.input_dict:
            self.thermalised = self.input_dict["thermalised"]
        else:
            self.thermalised = True

        if "cold_neutrals" in self.input_dict:
            self.cold_neutrals = self.input_dict["cold_neutrals"]
        else:
            self.cold_neutrals = True

        if "cold_ions" in self.input_dict:
            self.cold_ions = self.input_dict["cold_ions"]
        else:
            self.cold_ions = True

        self.get_impurities()
        self.get_impurity_ion_bal()
        self.build_impurity_average_charge()
        self.get_impurity_tecs()
        self.build_electron_doner_interpolaters()
        if self.contains_hydrogen:
            self.get_hydrogen_pecs()

        self.get_theta_functions(
            profile_function,
            cal_functions,
            calwave_functions,
            calint_func_functions,
            background_functions,
        )
        self.build_tags_slices_and_bounds()

        self.los = self.profile_function.electron_density.x

        self(np.random.rand(self.n_params))

    def __call__(self, theta):
        self.update_plasma_state(theta)

    def get_impurities(self):
        self.impurities = []
        for species in self.impurity_species:
            split_index = np.where([p == "_" for p in species])[0][0]
            elem = species[:split_index]
            if elem not in self.impurities:
                self.impurities.append(elem)

    def append_bounds_from_functional(self, bounds, functional):
        if type(functional.bounds[0]) == list:
            if len(functional.bounds) != functional.number_of_variables:
                raise ValueError(
                    type(functional),
                    "len(functional.bounds)!=functional.number_of_variables",
                )
            [bounds.append(b) for b in functional.bounds]
        elif np.isreal(functional.bounds[0]):
            [
                bounds.append(functional.bounds)
                for num in np.arange(functional.number_of_variables)
            ]
        else:
            raise TypeError(
                "functional.bounds must be a list of len 2 or a list of shape (num_var, 2)"
            )

        if (
            functional.number_of_variables == 1
            and type(functional.bounds[0]) is not list
        ):
            check = [functional.bounds]
        else:
            check = functional.bounds
        if check != bounds[-functional.number_of_variables :]:
            raise ValueError(
                "functional.bounds!=bounds[-functional.number_of_variables:]",
                check != bounds[-functional.number_of_variables :],
                type(functional),
                type(check),
                check,
                type(bounds[-functional.number_of_variables :]),
                bounds[-functional.number_of_variables :],
            )

        return bounds

    def build_tags_slices_and_bounds(self):
        self.tags = []
        bounds = []
        slice_lengths = []
        calibration_functions = [
            self.cal_functions,
            self.calwave_functions,
            self.calintfun_functions,
            self.background_functions,
        ]
        calibration_tags = ["cal", "calwave", "calint_func", "background"]
        for calibration_tag, calibration_function in zip(
            calibration_tags, calibration_functions
        ):
            for chord, chord_calibration_function in enumerate(calibration_function):
                int_check = calibration_tag == "cal" and not self.calibrate_intensity
                wave_check = (
                    calibration_tag == "calwave" and not self.calibrate_wavelength
                )
                intfun_check = (
                    calibration_tag == "calint_func"
                    and not self.calibrate_instrument_function
                )
                checks = any([int_check, wave_check, intfun_check])
                if not checks:
                    self.tags.append(calibration_tag + "_" + str(chord))
                    slice_lengths.append(chord_calibration_function.number_of_variables)
                    bounds = self.append_bounds_from_functional(
                        bounds, chord_calibration_function
                    )

        self.tags.append("electron_density")
        self.tags.append("electron_temperature")

        slice_lengths.append(self.profile_function.electron_density.number_of_variables)
        slice_lengths.append(
            self.profile_function.electron_temperature.number_of_variables
        )

        bounds = self.append_bounds_from_functional(
            bounds, self.profile_function.electron_density
        )
        bounds = self.append_bounds_from_functional(
            bounds, self.profile_function.electron_temperature
        )

        hydrogen_shape_tags = ["b-field", "viewangle"]
        hydrogen_shape_bounds = [[0, 1], [0, 2]]

        if self.contains_hydrogen and self.zeeman:
            for tag, b in zip(hydrogen_shape_tags, hydrogen_shape_bounds):
                self.tags.append(tag)
                slice_lengths.append(1)
                bounds.append(b)

        n_bounds = [11, 14]
        ti_bounds = [-1, 2]
        tau_bounds = [-6, 4]
        if self.thermalised:
            impurity_tags = ["_dens", "_tau"]
            impurity_bounds = [n_bounds, tau_bounds]
        else:
            impurity_tags = ["_dens", "_Ti", "_tau"]
            impurity_bounds = [n_bounds, ti_bounds, tau_bounds]
            # impurity_bounds = [[10, 15], [-2, 2], [-8, 2]] # including magical tau
        if self.include_doppler_shifts:
            impurity_tags.append("_velocity")
            impurity_bounds.append([-10, 10])
        for tag in impurity_tags:
            for s in self.species:
                is_h_isotope = any([s[0:2] == h + "_" for h in self.hydrogen_isotopes])
                if tag == "_tau" and is_h_isotope:
                    sion = s + tag
                    self.tags.append(sion)
                    slice_lengths.append(1)
                    bounds.append([-10, -2])  # ([-7, -3])
                elif tag == "_dens" and is_h_isotope and self.no_sample_neutrals:
                    pass
                elif tag == "_Ti" and is_h_isotope and self.cold_neutrals:
                    pass
                elif tag == "_Ti" and not is_h_isotope and self.cold_ions:
                    pass
                else:
                    sion = s + tag
                    tmp_b = impurity_bounds[
                        np.where([t in sion for t in impurity_tags])[0][0]
                    ]

                    self.tags.append(sion)
                    slice_lengths.append(1)
                    bounds.append(tmp_b)

        if "X_lines" in self.input_dict:
            for line in self.input_dict["X_lines"]:
                line = str(line)[1:-1].replace(", ", "_")
                self.tags.append("X_" + line)
                slice_lengths.append(1)
                bounds.append([0, 20])

        # building dictionary to have flags for which parameters are resoloved
        self.is_resolved = collections.OrderedDict()
        ti_resolved = self.input_dict["ion_resolved_temperatures"]
        tau_resolved = self.input_dict["ion_resolved_tau"]
        if "ion_resolved_velocity" in self.input_dict:
            velocity_resolved = self.input_dict["ion_resolved_velocity"]
        else:
            velocity_resolved = False
        # checking if Ti and tau is resolved
        res_not_apply = [
            not any([tag.endswith(t) for t in impurity_tags]) for tag in self.tags
        ]
        velocity_resolution_check = [
            tag.endswith("_velocity") and velocity_resolved for tag in self.tags
        ]
        ti_resolution_check = [tag.endswith("_Ti") and ti_resolved for tag in self.tags]
        dens_resolved = False
        dens_resolution_check = [
            tag.endswith("_dens") and dens_resolved for tag in self.tags
        ]
        tau_resolution_check = [
            tag.endswith("_tau") and tau_resolved for tag in self.tags
        ]
        # actually making dictionary to have flags for which parameters are resoloved
        zip_list = [
            self.tags,
            velocity_resolution_check,
            ti_resolution_check,
            dens_resolution_check,
            tau_resolution_check,
            res_not_apply,
        ]
        for tag, is_v_r, is_ti_r, is_dens_r, is_tau_r, rna in zip(*zip_list):
            self.is_resolved[tag] = is_v_r or is_ti_r or is_dens_r or is_tau_r or rna

        # building the slices to map BaySAR input to model parameters
        slices = []
        keep_theta_bound = []
        for p, L, b, n in zip(self.tags, slice_lengths, bounds, np.arange(len(bounds))):
            [keep_theta_bound.append(self.is_resolved[p]) for counter in np.arange(L)]
            if len(slices) == 0:
                slices.append((p, slice(0, L)))
            elif self.is_resolved[p]:
                last = slices[-1][1].stop
                slices.append((p, slice(last, last + L)))
            # if the parameter is not resolved there is a check for previous tags that it will share a slice with
            else:
                current_imp_tag = [
                    imp_tag for imp_tag in impurity_tags if p.endswith(imp_tag)
                ]
                slc = [
                    s
                    for tag, s in slices
                    if (tag.startswith(p[:1]) and tag.endswith(current_imp_tag[0]))
                ]
                if len(slc) != 0:
                    slices.append((p, slc[0]))
                else:
                    last = slices[-1][1].stop
                    slices.append((p, slice(last, last + L)))
                    keep_theta_bound[-1] = True

        self.n_params = slices[-1][1].stop
        self.slices = collections.OrderedDict(slices)
        self.theta_bounds = np.array(
            [np.array(b) for n, b in enumerate(bounds) if keep_theta_bound[n]],
            dtype=float,
        )
        # self.theta_bounds = np.array([theta_bounds.min(1), theta_bounds.max(1)]).T
        # self.bounds = bounds

        if np.isnan(self.theta_bounds).any():
            err_msg: str = "Bounds contain NaNs!"
            raise ValueError(err_msg)

        if self.n_params != len(self.theta_bounds):
            raise ValueError("self.n_params!=len(self.theta_bounds)")

        self.assign_theta_functions()

    def update_plasma_theta(self, theta):
        plasma_theta = []
        for p in self.slices.keys():
            plasma_theta.append((p, theta[self.slices[p]]))
        self.plasma_theta = collections.OrderedDict(plasma_theta)

    def update_plasma_state(self, theta):
        if not len(theta) == self.n_params:
            raise ValueError("len(theta)!=self.n_params")

        self.update_plasma_theta(theta)

        plasma_state = []
        for p in self.slices:
            values = np.array(theta[self.slices[p]]).flatten()
            # print(p, values)
            if p in self.theta_functions:
                values = self.theta_functions[p](values)
                # print(p, values, self.theta_functions[p])
            plasma_state.append((p, values))

        if hasattr(self, "plasma_state"):
            for p, values in plasma_state:
                self.plasma_state[p] = values
        else:
            self.plasma_state = collections.OrderedDict(plasma_state)

        if not hasattr(self, "reverse_electroneutrality"):
            self.reverse_electroneutrality = False

        self.calc_total_impurity_electrons(True)
        if self.reverse_electroneutrality:
            self.plasma_state["main_ion_density"] = self.plasma_state[
                "electron_density"
            ].copy()
            self.plasma_state["electron_density"] += self.total_impurity_electrons
        else:
            self.plasma_state["main_ion_density"] = (
                self.plasma_state["electron_density"] - self.total_impurity_electrons
            )

        if self.contains_hydrogen:
            species = self.hydrogen_species[0]
            # if self.reverse_electroneutrality:
            #     self.plasma_state[species+'_dens']=self.plasma_state['main_ion_density'].copy()
            # else:
            #     self.plasma_state[species+'_dens']=self.plasma_state['electron_density'].copy()
            # self.plasma_state[species+'_dens']=self.plasma_state['electron_density'].copy()
            if not species + "_dens" in self.slices:
                self.plasma_state[species + "_dens"] = self.plasma_state[
                    "main_ion_density"
                ].copy()
            self.update_neutral_profile()

        # get impurity electrons
        if hasattr(self, "impurity_electron"):
            self.get_impurity_electron_profiles()

        self.total_power()

    def update_neutral_profile(self):
        species = self.hydrogen_species[0]
        ne = self.plasma_state["electron_density"]
        te = self.plasma_state["electron_temperature"]
        n0 = self.plasma_state[species + "_dens"]  # [0]
        n1 = self.plasma_state["main_ion_density"]

        if not self.no_sample_neutrals:
            n0 = n0[0]

        adf11types: list[str] = ["scd", "acd", "plt", "prb"]
        if not hasattr(self, "neutral_adf11s"):
            from OpenADAS import get_adf11, load_adf11

            self.neutral_adf11s = dict(
                [
                    (
                        _adf11type,
                        load_adf11(
                            get_adf11(element="h", adf11type=_adf11type, year=12),
                            passed=True,
                        ),
                    )
                    for _adf11type in adf11types
                ]
            )

        interp_args: dict = dict(
            ne=("pecs", ne),
            Te=("pecs", te),
            kwargs=dict(bounds_error=False, fill_value=None),
        )
        for _adf11type in adf11types:
            self.__dict__[_adf11type] = (
                self.neutral_adf11s[_adf11type].sel(block=1).interp(**interp_args).data
            )

        # n0_time=self.plasma_state['n0_time']
        n0_time = self.plasma_state[species + "_tau"][0]
        if hasattr(self, "recombining"):
            if self.recombining is True:
                n0 += n1 * ne * self.acd * n0_time

        n0 -= n0 * ne * self.scd * n0_time
        n0 = n0.clip(1)
        self.plasma_state[species + "_dens"] = n0

        self.h_pot = 2.18e-18
        self.ion_source = n0 * ne * self.scd
        self.ion_sink = n1 * ne * self.acd
        self.E_ion = self.plt / self.scd + self.h_pot
        self.E_rec = self.prb / self.acd - self.h_pot

    def print_plasma_state(self):
        print_plasma_state(self)

    def get_theta_functions(
        self,
        profile_function=None,
        cal_functions=None,
        calwave_functions=None,
        calintfun_functions=None,
        background_functions=None,
    ):
        if profile_function is None:
            # x = np.linspace(1, 9, 5)
            # profile_function = MeshLine(x=x, zero_bounds=-2, bounds=[0, 10], log=True)
            # self.profile_function = arb_obj(
            #     electron_density=profile_function,
            #     electron_temperature=profile_function,
            #     number_of_variables_ne=len(x),
            #     number_of_variables_te=len(x),
            #     bounds_ne=[11, 16],
            #     bounds_te=[-1, 2],
            # )
            from .lineshapes import EsymmtricCauchyPlasma

            self.profile_function = EsymmtricCauchyPlasma()
        else:
            self.profile_function = profile_function

        # tmp_func = arb_obj_single_log_input(number_of_variables=1, bounds=[-5, 20], power=10)
        if cal_functions is None:
            tmp_func = default_cal_function()
            self.cal_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.cal_functions = cal_functions

        if background_functions is None:
            tmp_func = default_background_function()
            self.background_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.background_functions = background_functions

        # tmp_func = arb_obj_single_input(number_of_variables=1, bounds=[-5, 5])
        if calwave_functions is None:
            tmp_func = default_calwave_function()
            self.calwave_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.calwave_functions = calwave_functions

        if calintfun_functions is None:
            tmp_func = GaussianInstrumentFunctionCalibratior()
            self.calintfun_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.calintfun_functions = calwave_functions

    def assign_theta_functions(self):
        theta_functions = []
        if self.calibrate_intensity:
            theta_functions.append(self.cal_functions)
        if self.calibrate_wavelength:
            theta_functions.append(self.calwave_functions)
        if self.calibrate_instrument_function:
            theta_functions.append(self.calintfun_functions)

        theta_functions.extend(
            [
                self.background_functions,
                [
                    self.profile_function.electron_density,
                    self.profile_function.electron_temperature,
                ],
                [
                    power10
                    for tag in self.tags
                    if any(
                        [
                            (tag.startswith(s + "_") and ("_velocity" not in tag))
                            for s in self.species
                        ]
                    )
                ],
            ]
        )

        if self.include_doppler_shifts:
            theta_functions.append(
                [kms_to_ms for tag in self.tags if "_velocity" in tag]
            )

        theta_functions.append([power10 for tag in self.tags if "X_" in tag])

        theta_functions = np.concatenate(theta_functions).tolist()
        # theta_functions = np.concatenate( [f if type(f)!=list else f for f in theta_functions] ).tolist()

        theta_functions_tuples = []
        function_tags = ["cal", "back", "electron_density", "electron_temperature"]
        function_tags_full = [
            tag
            for tag in self.tags
            if (
                any([check in tag for check in function_tags])
                or any([tag.startswith(s + "_") for s in self.species])
                or "X_" in tag
            )
        ]

        self.function_check = [
            any([check in tag for check in function_tags])
            or any([tag.startswith(s + "_") for s in self.species])
            or ("X_" in tag)
            for tag in self.tags
        ]
        # self.function_check = [any([check in tag for check in function_tags]) for tag in plasma.tags]

        if len(function_tags_full) != len(theta_functions):
            print(theta_functions)
            print(function_tags_full)
            raise ValueError(
                "len(function_tags_full)!=len(theta_functions) ({}!={})".format(
                    len(function_tags_full), len(theta_functions)
                )
            )

        for tag, func in zip(function_tags_full, theta_functions):
            theta_functions_tuples.append((tag, func))

        self.theta_functions = collections.OrderedDict(theta_functions_tuples)

    def calc_total_impurity_electrons(self, set_attribute=True):
        total_impurity_electrons = 0

        ne = self.plasma_state["electron_density"]
        if self.concstar:
            for elem in self.impurities:
                species = [
                    species
                    for species in self.impurity_species
                    if species.startswith(elem + "_")
                ][0]
                dens = self.plasma_state[species + "_dens"]
                conc = (dens / ne.max()) * ne
                total_impurity_electrons += (conc / ne.max()) * ne
        else:
            te = self.plasma_state["electron_temperature"]
            for species in self.impurity_species:
                tau_tag = species + "_tau"
                dens_tag = species + "_dens"

                tau = self.plasma_state[tau_tag]
                conc = self.plasma_state[dens_tag]
                tau = np.zeros(len(ne)) + tau

                tmp_in = (tau, ne, te)
                # average_charge = np.exp(self.impurity_average_charge[species](tmp_in))
                average_charge = (
                    self.impurity_average_charge[species](tmp_in).clip(min=0) / 2
                )  # TODO: ...
                total_impurity_electrons += average_charge * conc

        if set_attribute:
            self.total_impurity_electrons = np.nan_to_num(total_impurity_electrons)

        return np.nan_to_num(total_impurity_electrons)

    def build_impurity_average_charge(self):
        interps = []
        for species in self.impurity_species:
            interps.append(
                (species, self.build_impurity_electron_interpolator(species))
            )

        self.impurity_average_charge = dict(interps)

    def build_impurity_electron_interpolator(self, species):
        te = self.adas_plasma_inputs["te"]
        ne = self.adas_plasma_inputs["ne"]
        tau = self.adas_plasma_inputs["magical_tau"]
        # tau = self.adas_plasma_inputs['tau_exc']

        elem, ion = self.species_to_elem_and_ion(species)
        ionp1 = ion + 1

        z_bal = self.impurity_ion_bal[elem][:, :, :, ion]
        zp1_bal = self.impurity_ion_bal[elem][:, :, :, ionp1]
        z_eff = ion * z_bal + ionp1 * zp1_bal  # /(z_bal+zp1_bal)

        # return RegularGridInterpolator((tau, ne, te), np.log(z_eff), bounds_error=False, fill_value=None)
        return RegularGridInterpolator(
            (tau, ne, te), z_eff, bounds_error=False, fill_value=None
        )

    def get_impurity_ion_bal(self):
        ion_bals = []
        rad_power = []

        te = self.adas_plasma_inputs["te"]
        ne = self.adas_plasma_inputs["ne"]
        tau_exc = self.adas_plasma_inputs["tau_exc"]
        tau_rec = self.adas_plasma_inputs["tau_rec"]

        if not hasattr(self, "concstar"):
            self.concstar = True
            print(f"Impurity plasma model self.constar set to {self.concstar}!")

        self.adf11_types = ["scd", "acd", "plt", "prb"]
        self.adf11 = {}
        for elem in self.impurities:
            elem_adas_yr = adas_line_data[elem]["ionisation_balance_year"]

            self.adf11[elem] = {}
            for adf11_type in self.adf11_types:
                if adf11_type in adas_line_data[elem]:
                    self.adf11[elem][adf11_type] = adas_line_data[elem][adf11_type]
                elif elem_adas_yr is not None:
                    self.adf11[elem][adf11_type] = get_adf11(
                        elem, yr=elem_adas_yr, type=adf11_type
                    )
                else:
                    raise ValueError(
                        f"{elem} data is missing references for {adf11_type.capitalize()} data!"
                    )

            num_ions = get_number_of_ions(elem)
            shape = (len(tau_exc) + len(tau_rec), len(ne), len(te), num_ions)
            shape_pow = (len(tau_exc) + len(tau_rec), len(ne), len(te), num_ions - 1)
            # shape=(len(tau_exc), len(ne), len(te), num_ions)
            bal = np.zeros(shape)
            power = np.zeros(shape_pow)
            meta_index = 0
            meta = get_meta(elem, index=meta_index)
            min_frac0 = 1e-8
            min_frac1 = 1e-15
            for t_counter, t in enumerate(tau_exc):
                print(t_counter, t)
                with HiddenPrints():
                    out, pow = run_adas406(
                        files=self.adf11[elem],
                        elem=elem,
                        te=te,
                        dens=ne,
                        tint=t,
                        meta=meta,
                        all=True,
                    )
                out["ion"][out["ion"] < min_frac0] = min_frac1
                if self.concstar:
                    concstar_correction = np.arange(out["ion"].shape[-1]).clip(1)
                    out["ion"] = out["ion"] / concstar_correction[None, None, :]
                bal[t_counter, :, :, :] = out["ion"]
                for ion in range(power.shape[-1]):
                    is1 = ion + 1
                    plt = read_adf11(
                        file=self.adf11[elem]["plt"],
                        adf11type="plt",
                        is1=is1,
                        index_1=-1,
                        index_2=-1,
                        te=te,
                        dens=ne,
                        all=True,
                    )
                    prb = read_adf11(
                        file=self.adf11[elem]["prb"],
                        adf11type="prb",
                        is1=is1,
                        index_1=-1,
                        index_2=-1,
                        te=te,
                        dens=ne,
                        all=True,
                    )

                    ni = bal[t_counter, :, :, ion]
                    nr = bal[t_counter, :, :, is1]
                    tmppower = ni * plt.T + nr * prb.T
                    tmppower *= self.adas_plasma_inputs["big_ne"]
                    power[t_counter, :, :, ion] = tmppower

            # downstream transport
            min_frac0 = 1e-4
            meta_index = -1
            meta = get_meta(elem, index=meta_index)
            for t_counter, t in enumerate(tau_rec):
                with HiddenPrints():
                    # out, pow = run_adas406(year=96, elem=elem, te=te, dens=ne, tint=t, meta=meta, all=True)
                    out, pow = run_adas406(
                        files=self.adf11[elem],
                        elem=elem,
                        te=te,
                        dens=ne,
                        tint=t,
                        meta=meta,
                        all=True,
                    )
                out["ion"][out["ion"] < min_frac0] = min_frac1
                if self.concstar:
                    concstar_correction = np.arange(out["ion"].shape[-1]).clip(1)
                    out["ion"] = out["ion"] / concstar_correction[None, None, :]
                bal[len(tau_exc) + t_counter, :, :, :] = out["ion"]
                for ion in range(power.shape[-1]):
                    is1 = ion + 1
                    plt = read_adf11(
                        file=self.adf11[elem]["plt"],
                        adf11type="plt",
                        is1=is1,
                        index_1=-1,
                        index_2=-1,
                        te=te,
                        dens=ne,
                        all=True,
                    )
                    prb = read_adf11(
                        file=self.adf11[elem]["prb"],
                        adf11type="prb",
                        is1=is1,
                        index_1=-1,
                        index_2=-1,
                        te=te,
                        dens=ne,
                        all=True,
                    )

                    ni = bal[t_counter, :, :, ion]
                    nr = bal[t_counter, :, :, is1]
                    tmppower = ni * plt.T + nr * prb.T
                    tmppower *= self.adas_plasma_inputs["big_ne"]
                    power[t_counter, :, :, ion] = tmppower

            ion_bals.append((elem, bal))
            rad_power.append((elem, power))

        self.impurity_ion_bal = dict(ion_bals)
        self.impurity_raditive_power_dict = dict(rad_power)
        self.build_impurity_raditive_power_splines()

    def build_impurity_raditive_power_splines(self):
        ne = self.adas_plasma_inputs["ne"]
        te = self.adas_plasma_inputs["te"]
        self.impurity_raditive_power = {}
        rad_power = []
        for elem in self.impurities:
            for ion in range(self.impurity_raditive_power_dict[elem].shape[-1]):
                elem_ion = elem + "_" + str(int(ion))
                for tnum, tau in enumerate(self.adas_plasma_inputs["magical_tau"]):
                    rates = self.impurity_raditive_power_dict[elem][tnum, :, :, ion]
                    tmpspline = RectBivariateSpline(ne, te, np.log(rates.clip(1e-30)))
                    rad_power.append((tau, tmpspline))

                self.impurity_raditive_power[elem_ion] = dict(rad_power)

    def impurity_power(self, species, return_rates=False):
        nz = self.plasma_state[species + "_dens"]
        tau = np.log10(self.plasma_state[species + "_tau"])
        ne = self.plasma_state["electron_density"]
        te = self.plasma_state["electron_temperature"]

        upper_index = self.adas_plasma_inputs["magical_tau"].searchsorted(tau)
        lower_index = upper_index - 1
        upper_tau = self.adas_plasma_inputs["magical_tau"][upper_index][0]
        lower_tau = self.adas_plasma_inputs["magical_tau"][lower_index][0]
        # da_taus=[lower_tau, tau, upper_tau]
        # lower_weight, upper_weight=1+np.diff(da_taus)/(da_taus[0]-da_taus[-1])
        # lower_weight, upper_weight=1+np.array([da_taus[1]-da_taus[0], da_taus[-1]-da_taus[1] ])/(da_taus[0]-da_taus[-1])
        upper_weight = (tau - lower_tau) / (upper_tau - lower_tau)
        lower_weight = 1 - upper_weight
        # print(lower_tau, tau, upper_tau)
        # print(lower_weight, upper_weight)

        upper_power_rates = upper_weight * np.exp(
            self.impurity_raditive_power[species][upper_tau].ev(ne, te)
        )
        lower_power_rates = lower_weight * np.exp(
            self.impurity_raditive_power[species][lower_tau].ev(ne, te)
        )
        power_rates = upper_power_rates + lower_power_rates

        if return_rates:
            return power_rates

        power = nz * power_rates
        if "power" not in self.plasma_state:
            self.plasma_state["power"] = {}

        self.plasma_state["power"][species] = power.copy()

        power = np.trapz(power, self.los)

        return power

    def extrapolate_impurity_power(self, species="N_2", proton_number=None):
        elem = species.split("_")[0]
        if proton_number is None:
            if elem in atomic_number:
                proton_number = atomic_number[elem]
            else:
                raise ValueError(
                    f"{elem} atomic number is not in data base. Pass atomic number using function input proton_number or update atomic_number dict in baysar.plasmas."
                )

        dens = self.plasma_state[species + "_dens"]
        tau = np.log10(self.plasma_state[species + "_tau"])
        ne = self.plasma_state["electron_density"]
        te = self.plasma_state["electron_temperature"]

        power = []
        for charge in range(proton_number):
            tmp_ion = elem + f"_{charge}"
            upper_index = self.adas_plasma_inputs["magical_tau"].searchsorted(tau)
            lower_index = upper_index - 1
            upper_tau = self.adas_plasma_inputs["magical_tau"][upper_index][0]
            lower_tau = self.adas_plasma_inputs["magical_tau"][lower_index][0]
            upper_weight = (tau - lower_tau) / (upper_tau - lower_tau)
            lower_weight = 1 - upper_weight

            upper_power_rates = upper_weight * np.exp(
                self.impurity_raditive_power[tmp_ion][upper_tau].ev(ne, te)
            )
            lower_power_rates = lower_weight * np.exp(
                self.impurity_raditive_power[tmp_ion][lower_tau].ev(ne, te)
            )
            power_rates = upper_power_rates + lower_power_rates

            power.append(dens * power_rates)

        return np.array(power)

    def total_power(self, extrapolate=False):
        power = 0
        ni = self.plasma_state["main_ion_density"]
        te = self.plasma_state["electron_temperature"]
        ne = self.plasma_state["electron_density"]
        if "power" not in self.plasma_state:
            self.plasma_state["power"] = {}

        # get neutral power
        if self.contains_hydrogen:
            for species in self.hydrogen_species:
                n0 = self.plasma_state[species + "_dens"]
                exc_power = ne * n0 * self.plt
                self.plasma_state["power"][species + "_exc"] = exc_power
                power += exc_power

            rec_power = ne * ni * self.prb
            self.plasma_state["power"][self.hydrogen_species[0] + "_rec"] = rec_power
            power += rec_power

            power = np.trapz(power, self.los)

        # get impurity power
        for species in self.impurity_species:
            power += self.impurity_power(species)
        # extrapolate impurity power
        if extrapolate:
            pass

        self.plasma_state["power"]["total"] = power

        return power

    def build_impurity_tec(self, file, exc, rec, elem, ion):
        te = self.adas_plasma_inputs["te"]
        ne = self.adas_plasma_inputs["ne"]
        tau = self.adas_plasma_inputs["magical_tau"]
        # tau = self.adas_plasma_inputs['tau_exc']
        big_ne = self.adas_plasma_inputs["big_ne"]

        with HiddenPrints():
            pecs_exc, info_exc = read_adf15(file, exc, te, ne, all=True)  # (te, ne), _
            pecs_rec, info_rec = read_adf15(file, rec, te, ne, all=True)  # (te, ne), _

        print(file, exc, elem, ion, info_exc["wavelength"], info_rec["wavelength"])

        tec406 = np.zeros((len(tau), len(ne), len(te)))

        tec_splines = []
        for t_counter, t in enumerate(tau):
            # ionbal = self.impurity_ion_bal[elem]
            z_bal = self.impurity_ion_bal[elem][t_counter, :, :, ion]
            zp1_bal = self.impurity_ion_bal[elem][t_counter, :, :, ion + 1]
            rates = big_ne * (
                pecs_exc.T * z_bal + pecs_rec.T * zp1_bal
            )  # /(z_bal+zp1_bal)
            # rates_exc=pecs_exc.T*z_bal
            # rates_rec=pecs_rec.T*zp1_bal
            # rates=np.nan_to_num( np.log10(rates_exc+rates_rec) )
            # rates+=np.log10(big_ne)
            # tec406[t_counter, :, :] = big_ne*(pecs_exc.T*ionbal[t_counter, :, :, ion] +
            #                                   pecs_rec.T*ionbal[t_counter, :, :, ion+1])
            # tec_splines.append(RectBivariateSpline(ne, te, np.log(rates.clip(1e-50))).ev)
            tec_splines.append(RectBivariateSpline(ne, te, np.log(rates.clip(1e-30))))
            # tec_splines.append(RectBivariateSpline(ne, te, rates).ev)

        # # log10 is spitting out errors ::( but it still runs ::)
        # # What about scipy.interpolate.Rbf ? # TODO - 1e40 REEEEEEEEEEEE
        # return RegularGridInterpolator((tau, ne, te), np.log(tec406.clip(1e-40)), bounds_error=False)
        return tec_splines

    def get_impurity_tecs(self):
        tecs = []
        for species in self.impurity_species:
            for line in self.input_dict[species].keys():
                print(
                    "Building impurity TEC splines: {} {}".format(
                        species, line, end="\r"
                    )
                )
                line_str = str(line).replace(", ", "_")
                for bad_character in ["[", "]", "(", ")"]:
                    line_str = line_str.replace(bad_character, "")
                line_tag = species + "_" + line_str
                file = self.input_dict[species][line]["pec"]
                exc = self.input_dict[species][line]["exc_block"]
                rec = self.input_dict[species][line]["rec_block"]
                elem, ion = self.species_to_elem_and_ion(species)
                print(line, self.input_dict[species][line])
                tec = self.build_impurity_tec(file, exc, rec, elem, ion)
                tecs.append((line_tag, tec))

        self.impurity_tecs = dict(tecs)

    @staticmethod
    def species_to_elem_and_ion(species):
        split_index_array = np.where([t == "_" for t in species])[0]
        split_index = split_index_array[0]
        assert len(split_index_array) == 1, (
            species + " does not have the correct format: elem_ionstage"
        )
        return species[:split_index], int(species[split_index + 1 :])

    def get_hydrogen_pecs(self):
        te = self.adas_plasma_inputs["te"]
        ne = self.adas_plasma_inputs["ne"]

        pecs = []
        from xarray import DataArray

        from OpenADAS import get_adf15, load_adf15

        for species in self.hydrogen_species:
            _element, _charge = species.split("_")
            _element = _element.lower()
            _charge = int(_charge)
            adf15: str = get_adf15(element=_element, charge=_charge, year=12)
            adf15_model: DataArray = load_adf15(adf15, passed=True)
            for line in self.input_dict[species].keys():
                print(
                    "Building hydrogen PEC splines: {} {}".format(
                        species, line, end="\r"
                    )
                )
                line_tag = species + "_" + str(line).replace(", ", "_")
                file = self.input_dict[species][line]["pec"]
                exc = self.input_dict[species][line]["exc_block"]
                rec = self.input_dict[species][line]["rec_block"]

                for block, r_tag in zip([exc, rec], ["exc", "rec"]):
                    # rates, _ = read_adf15(
                    #     file, block, te, ne, all=True
                    # )  # (te, ne), _# TODO - 1e50 alarm
                    pecs.append(
                        (
                            line_tag + "_" + r_tag,
                            np.log(adf15_model.sel(block=block)),
                            # RectBivariateSpline(ne, te, np.log(rates.T.clip(1e-50))).ev,
                        )
                    )
                # return

        self.hydrogen_pecs = dict(pecs)

    def is_theta_within_bounds(self, theta):
        if not check_bounds_order(self.theta_bounds):
            print(
                "Not all bounds are correctly ordered. Reordering bounds. Please check bounds!"
            )
            self.theta_bounds = np.array(
                [self.theta_bounds.min(1), self.theta_bounds.max(1)]
            ).T
            if not check_bounds_order(self.theta_bounds):
                raise ValueError("Bounds could not be reordered!")
        return [
            ((bound[0] < r) and (r < bound[1])) or any([b == r for b in bound])
            for bound, r in zip(self.theta_bounds, theta)
        ]

    def build_electron_doner_interpolaters(self):
        electron_doner_interpolaters(self)

    def get_impurity_electron_profiles(self):
        get_impurity_electrons(self)


from scipy.interpolate import RectBivariateSpline


def electron_doner_interpolaters(self):
    ne = self.adas_plasma_inputs["ne"]
    te = self.adas_plasma_inputs["te"]

    impurity_electrons = {}
    impurity_electrons_interp = {}
    for z0 in self.impurities:
        impurity_electrons_interp[z0] = {}
        num_charge_states = self.impurity_ion_bal[z0].shape[-1]

        electrons_per_ion = np.arange(num_charge_states)[None, None, None, :]
        if self.concstar:
            electrons_per_ion = np.square(electrons_per_ion)

        impurity_electrons[z0] = self.impurity_ion_bal[z0] * electrons_per_ion

        for z in range(num_charge_states):
            # print(z0, z)
            tmp_e_list = []
            for tnum, tau in enumerate(self.adas_plasma_inputs["magical_tau"]):
                rates = impurity_electrons[z0][tnum, :, :, z]
                tmpspline = RectBivariateSpline(ne, te, np.log(rates.clip(1e-30)))
                tmp_e_list.append((tau, tmpspline))

            impurity_electrons_interp[z0][z] = dict(tmp_e_list)

    self.impurity_electron_dict = impurity_electrons
    self.impurity_electron = impurity_electrons_interp


def get_impurity_electrons(self):
    ne = self.plasma_state["electron_density"]
    te = self.plasma_state["electron_temperature"]

    if True:  # 'total_ion_density' not in self.plasma_state:
        self.plasma_state["total_ion_density"] = self.plasma_state[
            "main_ion_density"
        ].copy()

    # print(self.impurities)
    # print(ne.max(), te.max())
    # print(ne.shape, te.shape)
    self.electrons_per_ion = {}
    for z0 in self.impurities:
        tmp = []
        num_charge_states = self.impurity_ion_bal[z0].shape[-1]
        for z in 1 + np.arange(num_charge_states - 1):
            tmp_key = f"{z0}_{z}_tau"
            if not tmp_key in self.plasma_state:
                tmp_key = None
                tmp_z = z
                while tmp_key is None:
                    tmp_z += 1
                    tmp_key = f"{z0}_{tmp_z}_tau"
                    if tmp_key not in self.plasma_state:
                        tmp_key = None

                    if tmp_z == num_charge_states:
                        tmp_key = [
                            ts
                            for ts in self.plasma_state
                            if ts.startswith(z0 + "_") and ts.endswith("_tau")
                        ]
                        tmp_key = tmp_key[-1]

            tau = self.plasma_state[tmp_key][0]

            i = self.adas_plasma_inputs["magical_tau"].searchsorted(np.log10(tau))
            j = i - 1
            i_upper = np.power(10, self.adas_plasma_inputs["magical_tau"][i])
            j_lower = np.power(10, self.adas_plasma_inputs["magical_tau"][j])
            k_norm = abs(i_upper - j_lower)

            lower_wieght = 1 - (tau - j_lower) / k_norm
            upper_wieght = 1 - lower_wieght

            if not np.isclose(
                tau,
                lower_wieght * j_lower + upper_wieght * i_upper,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
            ):
                print(
                    z0, z, tmp_key, tau, lower_wieght * j_lower + upper_wieght * i_upper
                )
                raise ValueError(f"Linear interpolation error!")

            exc_electrons = np.exp(
                self.impurity_electron[z0][z][
                    self.adas_plasma_inputs["magical_tau"][i]
                ].ev(ne, te)
            )
            # rec_electrons = np.exp(self.impurity_electron[z0][z+1][self.adas_plasma_inputs['magical_tau'][i]].ev(ne, te))
            # mean_electrons = np.array([exc_electrons, rec_electrons]).mean(0)
            # print(z0, z, np.log10(tau), mean_electrons.max(), mean_electrons.shape)
            # tmp.append(mean_electrons)
            tmp.append(exc_electrons)

        self.electrons_per_ion[z0] = np.array(tmp)
        # print(z0, self.electrons_per_ion[z0].sum(0).max())

        if self.concstar:
            if not hasattr(self, "impurity_density_profile"):
                self.impurity_density_profile = {}

            if not hasattr(self, "impurity_electron_profile"):
                self.impurity_electron_profile = {}

            z0_dens_key = [
                ts
                for ts in self.plasma_state
                if ts.startswith(z0 + "_") and ts.endswith("_dens")
            ][0]
            z0_dens = self.plasma_state[z0_dens_key][0]
            tmp_profile = z0_dens * (ne / ne.max())
            self.impurity_density_profile[z0] = tmp_profile / self.electrons_per_ion[
                z0
            ].sum(0)
            self.impurity_electron_profile[z0] = tmp_profile
            self.plasma_state["total_ion_density"] += self.impurity_density_profile[z0]

            # print(z0, z, self.plasma_state['total_ion_density'].max(), self.plasma_state['total_ion_density'].argmax(), tmp_profile.max(), self.plasma_state['main_ion_density'].max())


class PlasmaSimple2D:
    def __init__(self, chords, plasmas, profile_function=None):
        self.chords = np.array(chords)
        self.plasmas = plasmas
        self.profile_function = profile_function

        self.get_species()
        self.build_tags_slices_and_bounds()
        self.get_theta_functions()

    def __call__(self):
        pass

    def get_species(self):
        self.species = []
        for plasma in self.plasmas:
            for species0 in plasma.species:
                k = 1 + len(species0.split("_")[-1])
                species1 = species0[:-k]
                if species1 not in self.species:
                    self.species.append(species1)

    def build_tags_slices_and_bounds(self):
        self.plasma_state = collections.OrderedDict()
        self.plasma_theta = collections.OrderedDict()

        tags = []
        slices = []
        bounds = []
        # global/2D parameters
        # emission calibration
        tags.append("cal")
        bounds.append([-1, 1])
        slices.append(slice(0, 1))
        # wave calibration
        tags.append("calwave")
        bounds.extend([[4037.5, 4042.5], [0.19, 0.20]])
        slices.append(slice(slices[-1].stop, slices[-1].stop + 2))
        # background
        tags.append("background")
        bounds.extend([[11, 13]])
        slices.append(slice(slices[-1].stop, slices[-1].stop + 1))
        # separatrix varriables
        # electron density
        tags.append("electron_density_separatrix")
        ne_sep_bounds = [[13, 14], [12, 13], [18, 20], [1, 10]]
        bounds.extend(ne_sep_bounds)
        slices.append(slice(slices[-1].stop, slices[-1].stop + len(ne_sep_bounds)))
        tags.append("electron_density_radial")
        ne_radial_bounds = [[-1, 1], [0.7, 1.0], [0.0, 2.0]]
        bounds.extend(ne_radial_bounds)
        slices.append(slice(slices[-1].stop, slices[-1].stop + len(ne_radial_bounds)))
        # electron temperature
        tags.append("electron_temperature_separatrix")
        te_sep_bounds = [[0, 1.7], [0.8, 1.0]]
        bounds.extend(te_sep_bounds)
        slices.append(slice(slices[-1].stop, slices[-1].stop + len(te_sep_bounds)))
        tags.append("electron_temperature_radial")
        te_radial_bounds = [[0.3, 0.5], [0.0, 2.0]]
        bounds.extend(te_radial_bounds)
        slices.append(slice(slices[-1].stop, slices[-1].stop + len(te_radial_bounds)))
        # species
        species_attributes = ["dens", "Ti", "velocity", "tau"]
        species_attributes_bounds = [[-2, 0], [0, 2], [-30, 30], [-6, 1]]
        for elem, (att, att_bounds) in product(
            self.species, zip(species_attributes, species_attributes_bounds)
        ):
            # print(elem, att, att_bounds)
            att_check = att == "tau"
            elem_check = elem[0] in ("H", "D")
            # check=att_check+elem_check
            # print(elem, att, elem_check, att_check, check)
            if not (elem_check and att_check):
                tags.append(elem + "_" + att)
                bounds.append(att_bounds)
                slices.append(slice(slices[-1].stop, slices[-1].stop + 1))

        self.tags = tags
        self.theta_bounds = bounds
        self.slices = collections.OrderedDict(((a, b) for a, b in zip(tags, slices)))

    def get_theta_functions(self):
        self.theta_functions = collections.OrderedDict()
        if self.profile_function is not None:
            self.theta_functions[
                "electron_density_separatrix"
            ] = self.profile_function.electron_density
            self.theta_functions[
                "electron_temperature_separatrix"
            ] = self.profile_function.electron_temperature

    def update_plasma_state(self, theta):
        self.last_theta = theta
        for tag in self.tags:
            tmp_slice = self.slices[tag]
            self.plasma_theta[tag] = theta[tmp_slice]
            if tag in self.theta_functions:
                self.plasma_state[tag] = self.theta_functions[tag](theta[tmp_slice])
            else:
                self.plasma_state[tag] = theta[tmp_slice]

    def get_1d_thetas(self):
        plasmas = self.plasmas
        fan_numbers = self.chords
        # if chord_numbers is None:
        chord_numbers = [0 for n in fan_numbers]

        thetas1d = []
        for plasma, fan_num, chord_num in zip(plasmas, fan_numbers, chord_numbers):
            # print(plasma, fan_num, chord_num)
            thetas1d.append(self.get_1d_theta(plasma, fan_num, chord_num))

        return thetas1d

    def get_1d_theta(self, plasma, fan_num, chord_num=0):
        new_theta = np.zeros(plasma.n_params)
        fan_num_index = np.where(fan_num == self.chords)[0][0]

        calibration_tags = ["cal", "calwave", "background"]
        for ctag in calibration_tags:
            tmp_tag = ctag + "_" + str(chord_num)
            new_theta[plasma.slices[tmp_tag]] = self.plasma_state[ctag]

        for ptag in ["electron_density", "electron_temperature"]:
            new_theta[plasma.slices[ptag].start] = self.plasma_state[
                ptag + "_separatrix"
            ][fan_num_index]
            new_theta[plasma.slices[ptag]][1:] = self.plasma_state[ptag + "_radial"]

        species_attributes = ["dens", "Ti", "velocity", "tau"]
        for species, att in product(plasma.species, species_attributes):
            # print(species, att)
            tmp_tag = species + "_" + att
            split_species = species.split("_")
            elem, charge = species[: -(1 + len(species.split("_")[-1]))], int(
                split_species[-1]
            )
            if tmp_tag in plasma.slices:
                # density
                if att == "dens":
                    if charge > 0:
                        dk = np.log10(charge)
                    else:
                        dk = 0
                    new_theta[plasma.slices[tmp_tag]] = self.plasma_state[
                        "electron_density_separatrix"
                    ][fan_num_index]
                    fraction = self.plasma_state[elem + "_" + att][0]
                    new_theta[plasma.slices[tmp_tag]] -= fraction + dk
                # Ti
                if att == "Ti":
                    new_theta[plasma.slices[tmp_tag]] = self.plasma_state[
                        elem + "_" + att
                    ]
                # velocity
                if att == "velocity":
                    new_theta[plasma.slices[tmp_tag]] = self.plasma_state[
                        elem + "_" + att
                    ]
                # tau
                if att == "tau":
                    new_theta[plasma.slices[tmp_tag]] = self.plasma_state[
                        elem + "_" + att
                    ]
                pass

        if plasma.zeeman:
            new_theta[plasma.slices["b-field"]] = 0.0
            new_theta[plasma.slices["viewangle"]] = 0.0

        return new_theta


if __name__ == "__main__":
    from baysar.input_functions import make_input_dict

    num_chords = 1
    wavelength_axis = [np.linspace(3900, 4300, 512)]
    experimental_emission = [
        np.array([1e12 * np.random.rand() for w in wavelength_axis[0]])
    ]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ["D_ADAS", "N"]
    ions = [["0"], ["1", "2", "3"]]
    noise_region = [[4040, 4050]]
    # mystery_lines = [ [[4070], [4001, 4002]],
    #                    [[1], [0.4, 0.6]]]
    mystery_lines = None

    input_dict = make_input_dict(
        wavelength_axis=wavelength_axis,
        experimental_emission=experimental_emission,
        instrument_function=instrument_function,
        emission_constant=emission_constant,
        noise_region=noise_region,
        species=species,
        ions=ions,
        mystery_lines=mystery_lines,
        refine=[0.01],
        ion_resolved_temperatures=False,
        ion_resolved_tau=True,
    )

    from baysar.lineshapes import ReducedBowmanTPlasma

    x = np.linspace(-15, 35, 100)
    profile_function = ReducedBowmanTPlasma(
        x=x, dr_bounds=[-5, 1], bounds_ne=[13, 15], bounds_te=[0, 1.7]
    )
    plasma = PlasmaLine(input_dict, profile_function=profile_function)

    chords = [0, 1]
    from baysar.lineshapes import SimpleSeparatrix

    profile_function = SimpleSeparatrix(chords)
    plasma2d = PlasmaSimple2D(
        chords=chords, plasmas=[plasma, plasma], profile_function=profile_function
    )
    plasma2d.update_plasma_state(
        [np.random.uniform(b[0], b[1]) for b in plasma2d.theta_bounds]
    )

    thetas1d = plasma2d.get_1d_thetas()
    print(thetas1d)
