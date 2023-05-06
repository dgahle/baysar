import numpy as np


def gaussian_high_pass_cost(tmp, threshold, error):
    # everything above the threshold is good
    return -0.5 * (max([0, threshold - tmp]) / error) ** 2


def gaussian_low_pass_cost(tmp, threshold, error):
    # everything below the threshold is good
    return -0.5 * (max([0, tmp - threshold]) / error) ** 2


class AntiprotonCost:
    def __init__(self, plasma, sigma=1e11):
        self.plasma = plasma
        self.anti_profile_varience = sigma

    def __call__(self):
        if any(self.plasma.plasma_state["main_ion_density"] < 0):
            anti_profile = self.plasma.plasma_state["main_ion_density"].clip(max=0)
            return -0.5 * np.square(anti_profile / self.anti_profile_varience).sum()
        else:
            return 0.0


class MainIonFractionCost:
    def __init__(self, plasma, threshold=0.8, sigma=0.1):
        self.plasma = plasma
        self.threshold = threshold
        self.sigma = sigma

    def __call__(self):
        ne = self.plasma.plasma_state["electron_density"].clip(1)
        n_ion = self.plasma.plasma_state["main_ion_density"]

        self.factor = ne.copy()
        self.factor /= self.factor[np.argmax(n_ion)]
        self.factor = self.factor.clip(0.3)

        n_ion_fraction = n_ion / ne

        return sum(
            [
                gaussian_high_pass_cost(f, k * self.threshold, self.sigma)
                for f, k in zip(n_ion_fraction, self.factor)
            ]
        )


class StaticElectronPressureCost:
    def __init__(self, plasma, threshold=5e15, sigma=0.1):
        self.plasma = plasma
        self.threshold = threshold
        self.sigma = sigma

    def __call__(self):
        ne = self.plasma.plasma_state["electron_density"].clip(1)
        te = self.plasma.plasma_state["electron_temperature"].clip(0.01)
        pe = te * ne

        return sum(
            [gaussian_low_pass_cost(f, 1, self.sigma) for f in pe / self.threshold]
        )


class ElectronDensityTDVCost:
    def __init__(self, plasma, threshold=1, sigma=0.2):
        self.plasma = plasma
        self.threshold = threshold
        self.sigma = sigma

    def __call__(self):
        ne_tdv = abs(np.diff(self.plasma.plasma_state["electron_density"]))  # .sum()
        ni_tdv = abs(np.diff(self.plasma.plasma_state["main_ion_density"]))  # .sum()

        mean = (ne_tdv - ni_tdv) / ni_tdv

        # return sum([gaussian_low_pass_cost(f, 1, self.sigma) for f in mean/self.threshold])
        return gaussian_low_pass_cost(mean.sum() / self.threshold, 1, self.sigma)


class NeutralFractionCost:
    def __init__(self, plasma, threshold=[5e-3, 2e-1], sigma=0.1, species="D_ADAS_0"):
        self.plasma = plasma
        self.threshold = threshold
        self.sigma = sigma
        self.species = species

    def __call__(self):
        n0 = self.plasma.plasma_state[self.species + "_dens"].clip(1)
        ne = self.plasma.plasma_state["electron_density"]
        f0 = n0 / ne  # .max()

        # low_pass=sum([gaussian_low_pass_cost(f, 1, self.sigma) for f in f0/self.threshold[1]])
        high_pass = sum(
            [gaussian_high_pass_cost(f, 1, self.sigma) for f in f0 / self.threshold[0]]
        )

        # return low_pass+high_pass
        return high_pass


def curvature(profile):
    grad1 = np.gradient(profile / max(profile))
    grad2 = np.gradient(grad1)
    return -sum(np.square(grad2) / np.power((1 + np.square(grad1)), 3))  # curvature


class CurvatureCost(object):
    def __init__(self, plasma, scale):
        self.plasma = plasma
        self.scale = scale

        self.los = plasma.profile_function.electron_density.x_points
        self.empty_array = plasma.profile_function.electron_density.empty_theta
        if plasma.profile_function.electron_density.zero_bounds is not None:
            self.slice = slice(1, -1)
        else:
            self.slice = slice(0, len(self.empty_array))

    def __call__(self):
        curves = 0
        for tag in ["electron_density"]:  # , 'electron_temperature']:
            self.empty_array[self.slice] = self.plasma.plasma_theta.get(tag)
            curves += curvature(self.empty_array)
        return curves * self.scale


from numpy import diff, gradient, ones, power


def flatness_prior(profile, scale=1, x=None):
    if x is not None:
        d = gradient(profile, x)
    else:
        d = gradient(profile)
    # return -abs(scale/profile.max())*abs(diff(profile)).sum()
    return -scale * abs(d).sum()


assert flatness_prior(ones(10), 1) == 0, "Error in flatness_prior"


class FlatnessPriorBasic(object):
    def __init__(self, scale, tag, plasma):
        self.tag = tag
        self.scale = power(10, scale)
        self.plasma = plasma

    def __call__(self):
        return flatness_prior(self.plasma.plasma_state.get(self.tag), self.scale)


class PlasmaGradientPrior:
    def __init__(self, plasma, te_scale, ne_scale):
        self.te_scale = te_scale
        self.ne_scale = ne_scale
        self.plasma = plasma

        self.functions = [
            self.plasma.profile_function.electron_density,
            self.plasma.profile_function.electron_temperature,
        ]
        self.tags = ["electron_density", "electron_temperature"]
        self.scales = [self.ne_scale, self.te_scale]

    def __call__(self):
        ne_x = self.plasma.profile_function.electron_density.x_points
        te_x = self.plasma.profile_function.electron_temperature.x_points
        self.xs = [ne_x, te_x]

        cost = 0
        for f, s, x in zip(self.functions, self.scales, self.xs):
            cost += flatness_prior(f.empty_theta, s, x=x)

        return cost


class FlatnessPrior(object):
    def __init__(self, scale, tag, plasma):
        if tag not in ("electron_density", "electron_temperature"):
            raise ValueError("tag not in ('electron_density', 'electron_temperature')")

        self.tag = tag
        self.scale = power(10, scale)
        self.plasma = plasma

        if self.tag == "electron_density":
            self.los = plasma.profile_function.electron_density.x_points
            self.empty_array = plasma.profile_function.electron_density.empty_theta
            if plasma.profile_function.electron_density.zero_bounds is not None:
                self.slice = slice(1, -1)
            else:
                self.slice = slice(0, len(self.empty_array))
        elif self.tag == "electron_temperature":
            self.los = plasma.profile_function.electron_temperature.x_points
            self.empty_array = plasma.profile_function.electron_temperature.empty_theta
            if plasma.profile_function.electron_temperature.zero_bounds is not None:
                self.slice = slice(1, -1)
            else:
                self.slice = slice(0, len(self.empty_array))
        else:
            raise ValueError("tag not in ('electron_density', 'electron_temperature')")

    def __call__(self):
        self.empty_array[self.slice] = self.plasma.plasma_theta.get(self.tag)
        return flatness_prior(self.empty_array, self.scale)


class SeparatrixTePrior(object):
    def __init__(self, scale, plasma, index, ne_scale=0):
        self.scale = scale
        self.ne_scale = ne_scale
        self.plasma = plasma
        self.separatrix_position = index

    def __call__(self):
        te = np.array(self.plasma.plasma_theta["electron_temperature"])
        ne = np.array(self.plasma.plasma_theta["electron_density"])
        te_x = self.plasma.profile_function.electron_temperature.x
        ne_x = self.plasma.profile_function.electron_density.x

        te_check = te - te.max()
        ne_check = ne - ne.max()
        # return self.scale*te_check[self.separatrix_position] + self.ne_scale*ne_check[self.separatrix_position]

        te_cost = (
            self.scale
            * abs(te_x[np.where(te == te.max())]).max()
            * te_check[self.separatrix_position]
        )
        ne_cost = (
            self.ne_scale
            * abs(ne_x[np.where(ne == ne.max())]).max()
            * ne_check[self.separatrix_position]
        )
        return te_cost + ne_cost


class PeakTePrior:
    def __init__(self, plasma, te_min=1, te_err=1):
        self.plasma = plasma
        self.te_min = te_min
        self.te_err = te_err

    def __call__(self):
        te = self.plasma.plasma_state["electron_temperature"]
        cost = gaussian_low_pass_cost(te.max(), self.te_min, self.te_err)

        return cost


def get_impurity_species(plasma):
    impurity_species = []
    tmp = {}
    for impurity in plasma.impurities:
        tmp[impurity] = []
        tmp_ions = [ion for ion in plasma.species if impurity + "_" in ion]

        for ion in tmp_ions:
            impurity_species.append(ion)
            _, charge = plasma.species_to_elem_and_ion(ion)
            tmp[impurity].append(charge)

    return tmp


def tau_difference(plasma, show_taus=False):
    imp_dict = get_impurity_species(plasma)

    taus = []
    diff_taus = []
    for imp in imp_dict:
        for ion in imp_dict[imp]:
            tmp_tau_key = imp + "_" + str(ion) + "_tau"
            taus.append(plasma.plasma_state[tmp_tau_key][0])
        for dtau in -np.diff(np.log10(taus[-len(imp_dict[imp]) :])):
            diff_taus.append(dtau)

    if show_taus:
        print(taus)

    return diff_taus


class TauPrior:
    def __init__(self, plasma, mean=0, sigma=0.2):
        self.plasma = plasma
        self.mean = mean
        self.sigma = sigma

    def __call__(self):
        self.diff_log_taus = tau_difference(self.plasma)
        logp = np.array(
            [
                gaussian_low_pass_cost(dlt, threshold=self.mean, error=self.sigma)
                for dlt in self.diff_log_taus
            ]
        )
        return logp.sum()


class NIVTauPrior:
    def __init__(self, plasma_state, mean=1, sigma=0.2, ratio=5):
        self.plasma = plasma_state
        self.mean = mean
        self.sigma = sigma
        self.ratio = ratio

    def __call__(self):
        n_iii_tau = self.plasma["N_2_tau"][0]
        n_iv_tau = self.plasma["N_3_tau"][0]

        r = n_iv_tau / n_iii_tau

        lower_cost = gaussian_high_pass_cost(r, self.mean, self.sigma)
        upper_cost = gaussian_low_pass_cost(
            r, self.ratio * self.mean, self.ratio * self.sigma
        )
        cost = lower_cost + upper_cost

        return cost


class NIIITauPrior:
    def __init__(self, plasma_state, mean=1, sigma=0.3, ratio=5):
        self.plasma = plasma_state
        self.mean = mean
        self.sigma = sigma
        self.ratio = ratio

    def __call__(self):
        n_ii_tau = self.plasma["N_1_tau"][0]
        n_iii_tau = self.plasma["N_2_tau"][0]

        r = n_ii_tau / n_iii_tau
        # r = n_iv_tau / n_iii_tau

        lower_cost = gaussian_high_pass_cost(r, self.mean, self.sigma)
        # upper_cost = gaussian_low_pass_cost(r, self.ratio*self.mean, self.ratio*self.sigma)
        cost = lower_cost  # +upper_cost

        return cost


class EmsTeOrderPrior:
    def __init__(self, posterior, mean=2.0, sigma=0.2, species=["B_1", "C_2"]):
        self.posterior = posterior
        self.mean = mean
        self.sigma = sigma
        self.species = species  # sorted(species, key=lambda x: -int(x.split('_')[1]))
        self.impurity_indicies_dict = {}
        self.impurity_indicies = []
        for counter, c in enumerate(
            self.posterior.posterior_components[: posterior.plasma.num_chords]
        ):
            for i, l in enumerate(c.lines):
                if "species" in l.__dict__:
                    check_species_is_an_impurity = (
                        l.species in self.posterior.plasma.impurity_species
                    )
                    check_if_species_collected = (
                        not l.species in self.impurity_indicies_dict
                    )
                    check_if_species_wanted = l.species in self.species
                    check = all(
                        [
                            check_species_is_an_impurity,
                            check_if_species_collected,
                            check_if_species_wanted,
                        ]
                    )
                    if check:
                        self.impurity_indicies_dict[l.species] = (counter, i)
                        self.impurity_indicies.append((l.species, counter, i))

        self.impurity_indicies = sorted(
            self.impurity_indicies, key=lambda x: -int(x[0].split("_")[1])
        )

    def __call__(self):
        ems_te = []
        self.history = []
        for species, chord, line in self.impurity_indicies:
            ems_te.append(self.posterior.posterior_components[chord].lines[line].ems_te)
            self.history.append((*species.split("_"), ems_te[-1]))

        return gaussian_high_pass_cost(ems_te[0] - ems_te[1], self.mean, self.sigma)


from numpy import diff, square


class ImpurityTauPrior:
    def __init__(self, plasma, sigma=0.3):
        self.plasma = plasma
        self.sigma = sigma

        self.species = {}
        for species in self.plasma.impurity_species:
            z0, z = species.split("_")
            if z not in self.species:
                self.species[z] = []

            self.species[z].append(z0)

    def __call__(self):
        cost = 0
        for charge in self.species:
            if len(self.species[charge]) > 1:
                tmp_ions = [z0 + f"_{charge}" for z0 in self.species[charge]]
                ion_cost = (
                    -0.5
                    * (
                        square(
                            diff(
                                [
                                    self.plasma.plasma_theta[ion + "_tau"][0]
                                    for ion in tmp_ions
                                ]
                            )
                            / self.sigma
                        )
                    ).sum()
                )
                cost += ion_cost

        return cost


class WallConditionsPrior:
    def __init__(self, plasma, te_min=0.5, ne_min=2e12, te_err=None, ne_err=None):
        self.plasma = plasma
        self.te_min = te_min
        self.ne_min = ne_min
        self.te_err = te_err
        self.ne_err = ne_err

        default_error = 0.2
        if self.te_err is None:
            self.te_err = default_error
        if self.ne_err is None:
            self.ne_err = default_error

    def __call__(self):
        te = self.plasma.plasma_state["electron_temperature"]
        ne = self.plasma.plasma_state["electron_density"]

        cost = 0
        for t in [te[0], te[-1]]:
            cost += gaussian_low_pass_cost(t / self.te_min, 1, self.te_err)
        for n in [ne[0], ne[-1]]:
            cost += gaussian_low_pass_cost(n / self.ne_min, 1, self.ne_err)

        return cost


class SimpleWallPrior:
    def __init__(self, plasma, te_min=0.5, ne_min=2e12, te_err=None, ne_err=None):
        self.plasma = plasma
        self.te_min = te_min
        self.ne_min = ne_min
        self.te_err = te_err
        self.ne_err = ne_err

        default_error = 0.5
        if self.te_err is None:
            self.te_err = default_error * self.te_min
        if self.ne_err is None:
            self.ne_err = default_error * self.ne_min

    def __call__(self):
        te = self.plasma.plasma_state["electron_temperature"][-1]
        ne = self.plasma.plasma_state["electron_density"][-1]

        cost = 0
        cost += gaussian_low_pass_cost(te, self.te_min, self.te_err)
        cost += gaussian_low_pass_cost(ne, self.ne_min, self.ne_err)

        return cost


class BowmanTeePrior:
    def __init__(self, plasma, sigma_err=0.2, nu_err=0.2):
        self.plasma = plasma
        self.sigma_err = sigma_err
        self.nu_err = nu_err
        # self.sigma_indicies=[plasma.slices['electron_density'].start+2, plasma.slices['electron_temperature'].start+1]

    def __call__(self):
        cost = 0.0
        sigma_diff = (
            self.plasma.plasma_theta["electron_temperature"][1]
            - self.plasma.plasma_theta["electron_density"][2]
        )
        nu_diff = (
            self.plasma.plasma_theta["electron_density"][4]
            - self.plasma.plasma_theta["electron_temperature"][3]
        )
        cost += gaussian_low_pass_cost(sigma_diff, 0.0, self.sigma_err)
        # cost+=gaussian_low_pass_cost(nu_diff, 0., self.nu_err)
        return cost


from numpy import diff


class ChargeStateOrderPrior:
    def __init__(self, posterior, mean=2.0, sigma=0.2):
        self.posterior = posterior
        self.mean = mean
        self.sigma = sigma
        self.impurity_indicies_dict = {}
        for counter, c in enumerate(
            self.posterior.posterior_components[: posterior.plasma.num_chords]
        ):
            for i, l in enumerate(c.lines):
                if "species" in l.__dict__:
                    if (
                        l.species in self.posterior.plasma.impurity_species
                        and not l.species in self.impurity_indicies_dict
                    ):
                        self.impurity_indicies_dict[l.species] = (counter, i)

        # print(self.impurity_indicies_dict)

    def __call__(self):
        self.history = []
        tmp = self.impurity_indicies_dict
        tmp1 = []
        cost = 0
        for elem in self.posterior.plasma.impurities:
            tmp_history = []
            for ion in sorted(
                [ion for ion in tmp if ion.startswith(elem)], key=lambda x: tmp[x]
            ):
                chords, line_index = tmp[ion]
                tmp1.append(
                    self.posterior.posterior_components[chords].lines[line_index].ems_te
                )
                tmp_history.append((elem, ion, tmp1[-1]))
                self.history.append((elem, ion, tmp1[-1]))

            dts = diff(
                [
                    ems_te[-1]
                    for ems_te in sorted(
                        tmp_history, key=lambda x: float(x[1].split("_")[1])
                    )
                ]
            )
            for dt in dts:
                cost += gaussian_high_pass_cost(dt, self.mean, self.sigma)

        return cost


class CiiiPrior:
    def __init__(self, plasma, mean=0, sigma=0.2):
        self.plasma = plasma
        self.mean = mean
        self.sigma = sigma

    def __call__(self):
        if "C_2_tau" in self.plasma.plasma_state:
            c_2_tau = self.plasma.plasma_state["C_2_tau"].mean()
            n_2_tau = self.plasma.plasma_state["N_2_tau"].mean()
            r_tau = np.log10(c_2_tau / n_2_tau)

            return -0.5 * np.square((r_tau - self.mean) / self.sigma)
        else:
            return 0


class NeutralPowerPrior:
    def __init__(self, plasma, mean=1.0, sigma=0.5):
        self.plasma = plasma
        self.mean = mean
        self.sigma = sigma

    def __call__(self):
        self.exc_power = np.trapz(
            self.plasma.plasma_state["power"]["D_ADAS_0_exc"], self.plasma.los
        )

        return gaussian_low_pass_cost(self.exc_power, self.mean, self.sigma)


from itertools import product

from numpy import array, square, trapz


class BolometryPrior:
    def __init__(self, plasma, mean=10, sigma=0.5):
        self.plasma = plasma
        self.mean = mean
        self.sigma = sigma

        self.get_references()

    def __call__(self):
        self.synthetic_measurement = self.syntetic_bolometer().sum()
        res = (self.synthetic_measurement - self.mean) / self.sigma
        return -0.5 * square(res)

    def __print__(self):
        ...

    def syntetic_bolometer(self):
        bolo_estimation = []
        # neutral power
        if self.plasma.contains_hydrogen:
            for species, reaction in product(
                self.plasma.hydrogen_species, ("exc", "rec")
            ):
                tmp_power = self.plasma.plasma_state["power"][species + f"_{reaction}"]
                tmp_power = trapz(tmp_power, self.plasma.los).sum()
                bolo_estimation.append(tmp_power)

        # impurity power
        self.power_profiles = {}
        for species in self.reference_species:
            tmp_power = self.plasma.extrapolate_impurity_power(species)
            self.power_profiles[species.split("_")[0]] = tmp_power.copy()
            tmp_power = trapz(tmp_power, self.plasma.los).sum()
            bolo_estimation.append(tmp_power)

        return array(bolo_estimation)

    def get_references(self):
        self.reference_species = []
        for elem in self.plasma.impurities:
            if elem + "_2" in self.plasma.impurity_species:
                self.reference_species.append(elem + "_2")
            else:
                ref = [
                    ref
                    for ref in self.plasma.impurity_species
                    if ref.startswith(f"{elem}_")
                ][-1]
                self.reference_species.append(ref)


class StarkToPeakPrior:
    def __init__(self, plasma, line, mean=0.65, sigma=0.075):
        self.plasma = plasma
        self.line = line
        self.mean = mean
        self.sigma = sigma

    def __call__(self):
        ne_peak = self.plasma.plasma_state["electron_density"].max()
        self.rec_stark_peak_ratop = self.line.rec_ne / ne_peak
        self.exc_stark_peak_ratop = self.line.exc_ne / ne_peak

        self.exc_cost = gaussian_high_pass_cost(
            self.exc_stark_peak_ratop, self.mean, self.sigma
        ) * (1 - self.line.f_rec)
        self.rec_cost = (
            gaussian_high_pass_cost(self.rec_stark_peak_ratop, self.mean, self.sigma)
            * self.line.f_rec
        )
        self.cost = self.exc_cost + self.rec_cost

        return self.cost


class TeMinPrior:
    def __init__(self, plasma_state, ratio=2.0, sigma=0.05):
        self.plasma_state = plasma_state
        self.ratio = ratio
        self.sigma = sigma

    def __call__(self):
        te = self.plasma_state["electron_temperature"]
        ratio = te.max() / te.min()

        return gaussian_high_pass_cost(ratio, self.ratio, self.sigma)


def gaussian_band_pass_cost(data, bounds, errors):
    funcs = [gaussian_high_pass_cost, gaussian_low_pass_cost]

    cost = 0
    for b, e, f in zip(bounds, errors, funcs):
        cost += f(data, b, e)

    return cost


from numpy import log10, power


class TeTauPrior:
    def __init__(self, plasma, sigma=0.3):
        self.plasma = plasma
        self.sigma = sigma

        self.errors = [self.sigma, self.sigma]

    def __call__(self):
        self.get_bounds()
        cost = 0
        for z in self.plasma.impurity_species:
            cost += self.get_cost(z)

        return cost

    def get_cost(self, ion):
        log_tau = log10(self.plasma.plasma_state[ion + "_tau"][0])

        return gaussian_band_pass_cost(log_tau, self.bounds, self.errors)

    def get_bounds(self):
        self.te_max = self.plasma.plasma_state["electron_temperature"].max()

        if self.te_max > 5:
            self.bounds = [-5, 0]
        elif self.te_max < 3:
            self.bounds = [3, 7]
        else:
            self.bounds = [-3, 7]


from numpy import array, trapz, where


def get_reduced_stark_spectra(self, line, half_width=10):
    """
    Crop the wavelength region of a Balmer line of the spectra, spectral error and wavelength array
    """

    wavelengths = self.x_data.copy()
    spectra = self.y_data.copy()
    spectra_error = self.error.copy()

    cwl = line.lineshape.cwl.mean()
    wavelengths_bounds = cwl + half_width * array([-1, 1])

    crop_indicies = where(
        [wavelengths_bounds.min() < w < wavelengths_bounds.max() for w in wavelengths]
    )

    return (
        spectra[crop_indicies],
        spectra_error[crop_indicies],
        wavelengths[crop_indicies],
        crop_indicies,
    )


from numpy import diff, isnan, power, round, square
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass
from scipy.signal import fftconvolve

from baysar.tools import centre_peak


class FastStarkModel:
    def __init__(self, chord, line):
        self.chord = chord
        self.line = line
        (
            self.spectra,
            self.spectra_error,
            self.wavelengths,
            self.crop_indicies,
        ) = get_reduced_stark_spectra(self.chord, self.line)

        log_ne_bounds = (12.0, 15.5)
        te_bounds = (3, 30)  # (0.5, 30)
        shift_bounds = (-2, 2)

        self.bounds = [log_ne_bounds, te_bounds]  # , shift_bounds)

        self.background_estimation = 0.5 * (
            self.spectra[:5].mean() + self.spectra[-5:].mean()
        )
        self.estimed_intensity = trapz(
            self.spectra - self.background_estimation, self.wavelengths
        )
        self.shift = 0  # self.wavelengths[self.spectra.argmax()] - self.wavelengths[lineshape.argmax()]
        self.slow = False

        test_theta = [14, 10]  # , 0]
        self(test_theta)
        # print(f"theta = {test_theta} -> cost = {self(test_theta)}")

    def theta_check(self, theta):
        test = []
        for val, b in zip(theta, self.bounds):
            test.append(
                not ((min(b) < val < max(b)) or (min(b) == val or max(b) == val))
            )

        if any(test):
            raise ValueError(f"Input is out of bounds! ({theta})")

    def min(self, theta):
        return -self(theta)

    def __call__(self, theta):
        if self.slow:
            (
                logne,
                te,
                self.shift,
                self.estimed_intensity,
                self.background_estimation,
            ) = theta
            theta = (logne, te)

        # self.theta_check(theta)
        self.forward_model(theta)

        res = (self.spectra - self.fm) / self.spectra_error
        cost = -0.5 * sum(square(res))

        return cost

    def forward_model(self, theta):
        logne, te = theta

        _lineshape = self.line.lineshape([power(10, logne), te, te, 0, 0])
        lineshape = fftconvolve(_lineshape, self.chord.instrument_function, mode="same")
        self.lineshape_interpolator = interp1d(
            self.line.lineshape.wavelengths,
            lineshape,
            bounds_error=False,
            fill_value="extrapolate",
        )
        lineshape = self.lineshape_interpolator(self.wavelengths)

        shifted_lineshape = self.lineshape_interpolator(self.wavelengths - self.shift)

        fm = self.estimed_intensity * shifted_lineshape + self.background_estimation

        if len(fm) > 1024:
            raise ValueError("need to reduce wavelengths dispersion")
        elif len(fm) != len(self.spectra):
            raise ValueError("fm array len is not equal to spectra")
        elif isnan(fm).any():
            raise TypeError("fm contains NaNs!")

        self.fm = fm


from inference.mcmc import GibbsChain, HamiltonianChain
from scipy.optimize import brute, fmin_l_bfgs_b


def estimate_stark_density(self, line):
    f = FastStarkModel(self, line)

    finish = None  # fmin_l_bfgs_b
    x0, fval, grid, Jout = brute(
        f.min, f.bounds, Ns=30, full_output=True, disp=False, finish=finish
    )

    print(f"x0 {round(x0[0], 2)}, {round(x0[1], 2)} ({fval})")

    f.slow = True
    f.bounds.extend(
        [
            (-2, 2),
            f.estimed_intensity * array([0.1, 2.0]),
            f.background_estimation * array([0.5, 2.0]),
        ]
    )
    x0_slow = [*x0, 0, f.estimed_intensity, f.background_estimation]
    x1, logP, bfgs_meta = fmin_l_bfgs_b(
        f.min,
        x0_slow,
        approx_grad=True,
        bounds=f.bounds,
        epsilon=1e-8,
        maxfun=15000,
        maxiter=15000,
        disp=None,
        callback=None,
    )

    print(f"x1 {round(x1[0], 2)}, {round(x1[1], 2)} ({logP})")

    chain = GibbsChain(posterior=f, start=x1)
    for param, bounds in enumerate(f.bounds):
        chain.set_boundaries(param, bounds)

    chain.run_for(minutes=2)
    # chain.plot_diagnostics()

    chain.autoselect_burn_and_thin()
    # chain.matrix_plot()

    print(
        f"GibbsChain {round(chain.mode()[0], 2)}, {round(chain.mode()[1], 2)} ({f.min(chain.mode())})"
    )

    # return f, chain.mode(), chain.get_interval(0.68)[0], chain.get_interval(0.95)[0]
    return f, chain.mode(), chain.get_interval(0.68), chain.get_interval(0.95), chain


def estimate_stark_densities(self):
    self.stark_density = []
    for l in self.hydrogen_lines:
        self.stark_density.append(estimate_stark_density(self.spectrometer_chord, l))


from baysar.linemodels import BalmerHydrogenLine


def find_hydrogen_lines(self):
    self.hydrogen_line_indicies = []
    self.hydrogen_lines = []
    for i, l in enumerate(self.spectrometer_chord.lines):
        if type(l) == BalmerHydrogenLine:
            self.hydrogen_line_indicies.append(i)
            self.hydrogen_lines.append(l)


from copy import copy

from numpy import logspace, square
from scipy.stats import gaussian_kde


class FastStarkPrior:
    def __init__(self, chord, line_index=0):
        self.plasma_state = chord.plasma.plasma_state
        self.line = chord.lines[line_index]
        (
            self._fast_stark_model,
            self._x0,
            self._x0_sample_68pc,
            self._x0_sample_95pc,
            self._chain,
        ) = estimate_stark_density(chord, self.line)

        self.setup_cost_function()

        if hasattr(chord.plasma, "plasma_reference"):
            ne_peak = chord.plasma.plasma_reference["electron_density"].max()
            print()
            print(f"SOLPS ne peak = {round(ne_peak / 1e13, 2)} 1e13 cm-3")
            print()

            if (self.fast_stark_ne / ne_peak) > 1.2:
                import matplotlib.pyplot as plt
                from numpy import array

                fig, (fit, pdf) = plt.subplots(1, 2, figsize=(11, 4))

                wave = self._fast_stark_model.wavelengths
                fit.plot(wave, self._fast_stark_model.spectra)
                self._fast_stark_model.slow = True
                fm = []
                for x0 in self._x0_sample_95pc[0]:
                    self._fast_stark_model(x0)
                    fm.append(self._fast_stark_model.fm)
                fm = array(fm)
                fit.plot(wave, fm.min(0), color="pink")
                fit.plot(wave, fm.max(0), color="pink")
                fit.set_xlim((wave.min(), wave.max()))
                self._fast_stark_model.slow = False

                pdf.plot(self._ne_pdf_axis, self.ne_pdf(self._ne_pdf_axis))
                pdf.set_xlim((0.0, 1e14))
                # pdf.set_xlim((1e12, 1e15))
                # pdf.set_xscale('log')

                plt.savefig("stark_plot")
                plt.close()

                log_ne = [x0[0] for x0 in self._x0_sample_95pc[0]]
                s2n = (
                    self._fast_stark_model.spectra
                    / self._fast_stark_model.spectra_error
                )
                error_message = f"Stark ne >> reference ne max! (Ratio = {round(self.fast_stark_ne / ne_peak, 2)}) "
                error_message += (
                    f"(Fast Stark = {round(self.fast_stark_ne/1e13, 2)} 1e13 cm-3, "
                )
                error_message += f"log range ({min(log_ne)}, {max(log_ne)})) (max|S2N| = {round(s2n.max(), 1)})"
                # raise ValueError(f"Stark ne >> reference ne max! (Ratio = {round(self.fast_stark_ne / ne_peak, 2)}) (Fast Stark = {round(self.fast_stark_ne/1e13, 2)} 1e13 cm-3, log range ({min(log_ne)}, {max(log_ne)})) (max|S2N| = {round(s2n.max(), 1)})")
                print(error_message)

    def __call__(self):
        ne_peak = self.line.plasma.plasma_state["electron_density"].max()
        self.ne_stark = copy(self.line.ems_ne)

        if self.fast_stark_ne > 1e13:
            stark_cost = self.ne_pdf.logpdf(self.ne_stark).sum() - self._ne_mode_logP
        else:
            stark_res = (1e13 - max([1e13, self.ne_stark])) / 1e13
            stark_cost = -0.5 * square(stark_res)

        cost = [stark_cost]

        ratio_check = self.fast_stark_ne / ne_peak
        # ratio_check = self.ne_stark / ne_peak
        ratio_error = 0.1
        # upper check
        upper_ratio_limit = 0.9
        lower_ratio_limit = 0.6

        upper_res = (
            upper_ratio_limit - max([upper_ratio_limit, ratio_check])
        ) / ratio_error
        cost.append(-0.5 * square(upper_res))
        if self.fast_stark_ne > 1e13:
            lower_res = (
                lower_ratio_limit - min([lower_ratio_limit, ratio_check])
            ) / ratio_error
        else:
            lower_res = (
                lower_ratio_limit - min([lower_ratio_limit, 1e13 / ne_peak])
            ) / ratio_error

        cost.append(-0.5 * square(lower_res))

        self.last_cost = cost

        return sum(cost)

    def setup_cost_function(self):
        self.ne_pdf = gaussian_kde(
            [10 ** theta[0] for theta in self._x0_sample_95pc[0]]
        )
        self._ne_pdf_axis = logspace(12, 16, 10000)
        self._ne_mode_index = self.ne_pdf(self._ne_pdf_axis).argmax()
        self._ne_mode = self._ne_pdf_axis[self._ne_mode_index]
        self._ne_mode_prob = self.ne_pdf(self._ne_mode)[0]
        self._ne_mode_logP = self.ne_pdf.logpdf(self._ne_mode)[0]
        self._stark_cost = self.ne_pdf(self._ne_pdf_axis) / self._ne_mode_prob

        self.fast_stark_ne = self._ne_mode

        print(f"Fast Stark dens = {round(self._ne_mode/1e13, 2)}e13 cm-3 (mode)")

        self.max_cost = 3


class WavelengthCalibrationPrior:
    def __init__(self, plasma, mean, std):
        self.plasma_state = plasma.plasma_state
        self.mean = mean
        self.std = std

    def __call__(self):
        cost = 0
        if "calwave_0" in self.plasma_state:
            cwl, disp = self.plasma_state["calwave_0"]
            cost += -0.5 * np.square(
                (self.mean - self.plasma_state["calwave_0"]) / self.std
            )

        return cost.sum()


class CalibrationPrior:
    def __init__(self, plasma):
        self.plasma_state = plasma.plasma_state
        self.mean = 1
        self.std = 0.2

    def __call__(self):
        cost = 0
        if "cal_0" in self.plasma_state:
            cost += -0.5 * np.square(
                (self.mean - self.plasma_state["cal_0"]) / self.std
            )

        return cost.sum()


class GaussianInstrumentFunctionPrior:
    def __init__(self, plasma, mean=0.8, std=0.1):
        self.plasma_state = plasma
        self.mean = mean
        self.std = std

    def __call__(self):
        cost = -0.5 * np.square(
            (self.plasma_state["calint_func_0"][0] - self.mean) / self.std
        )
        return cost


from numpy import diag, exp, eye, log, subtract, zeros
from scipy.linalg import ldl, solve_banded, solve_triangular


def tridiagonal_banded_form(A):
    B = zeros([3, A.shape[0]])
    B[0, 1:] = diag(A, k=1)
    B[1, :] = diag(A)
    B[2, :-1] = diag(A, k=-1)
    return B


class SymmetricSystemSolver(object):
    def __init__(self, A):
        L_perm, D, fwd_perms = ldl(A)

        self.L = L_perm[fwd_perms, :]
        self.DB = tridiagonal_banded_form(D)
        self.fwd_perms = fwd_perms
        self.rev_perms = fwd_perms.argsort()

    def __call__(self, b):
        # first store the permuted form of b
        b_perm = b[self.fwd_perms]

        # solve the system by substitution
        y = solve_triangular(self.L, b_perm, lower=True)
        h = solve_banded((1, 1), self.DB, y)
        x_perm = solve_triangular(self.L.T, h, lower=False)

        # now un-permute the solution vector
        x_sol = x_perm[self.rev_perms]
        return x_sol


class GaussianProcessPrior(object):
    def __init__(self, mu, plasma=None, A=2.0, L=0.02, profile=None):
        self.plasma = plasma
        self.tag = profile
        self.A = A
        self.L = L

        self.mu = mu
        self.los = plasma.profile_function.electron_density.x_points
        self.empty_array = plasma.profile_function.electron_density.empty_theta
        if plasma.profile_function.electron_density.zero_bounds is not None:
            self.slice = slice(1, -1)
        else:
            self.slice = slice(0, len(self.empty_array))

        # get the squared-euclidean distance using outer subtraction
        D = subtract.outer(self.los, self.los) ** 2

        # build covariance matrix using the 'squared-exponential' function
        K = (self.A**2) * exp(-0.5 * D / self.L**2)

        # construct the LDL solver for the covariance matrix
        solver = SymmetricSystemSolver(K)

        # use the solver to get the inverse-covariance matrix
        I = eye(K.shape[0])
        self.iK = solver(I)

        covariance_error = abs(self.iK.dot(K) - I).max()
        tolerence = 1e-6
        if covariance_error > tolerence:
            # raise ValueError("Error in construction of covariance matrix!"+
            #                 f"max(|iKK-I|) = {covariance_error:g0.3} (> {tolerence})")
            raise ValueError(
                "Error in construction of covariance matrix! \n \
                              max(|iKK-I|) = {} > {}".format(
                    covariance_error, tolerence
                )
            )

    def __call__(self):
        self.empty_array[self.slice] = self.plasma.plasma_theta.get(self.tag)
        return self.log_prior(self.empty_array - self.mu)

    def log_prior(self, field):
        return -0.5 * (field.T).dot(self.iK.dot(field))

    # def gradient(self):
    #     grad = zeros(self.plasma.N_params)
    #     grad[self.plasma.slices['Te']] = self.log_prior_gradient( log(self.plasma.get('Te')) - self.mu_Te ) / self.plasma.get('Te')
    #     grad[self.plasma.slices['ne']] = self.log_prior_gradient( log(self.plasma.get('ne')) - self.mu_ne ) / self.plasma.get('ne')
    #     # grad[self.plasma.slices['n0']] = self.log_prior_gradient( log(self.plasma.get('n0')) - self.mu_n0 ) / self.plasma.get('n0')
    #     return grad

    # def log_prior_gradient(self, field):
    #     return -self.iK.dot(field)


if __name__ == "__main__":
    pass
