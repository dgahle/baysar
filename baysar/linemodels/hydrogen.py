# Imports
from copy import copy

from numpy import (
    arange,
    cos,
    diff,
    divide,
    dot,
    empty,
    exp,
    interp,
    linspace,
    log,
    nan,
    nan_to_num,
    ndarray,
    power,
    round,
    sin,
    trapz,
)
from scipy.constants import e as electron_charge
from scipy.constants import m_e as electron_mass
from scipy.constants import physical_constants, pi, speed_of_light
from scipy.signal import fftconvolve
from xarray import DataArray

from .doppler import DopplerLine, doppler_shift_BalmerHydrogenLine
from .tools import atomic_masses

# Variables
b_field_to_cf_shift = electron_charge / (
    4 * pi * electron_mass * speed_of_light * 1e10
)  # cf is central frequency
loman_coeff = {
    "32": [0.7665, 0.064, 3.710e-18],  # Balmer Series
    "42": [0.7803, 0.050, 8.425e-18],
    "52": [0.6796, 0.030, 1.310e-15],
    "62": [0.7149, 0.028, 3.954e-16],
    "72": [0.7120, 0.029, 6.258e-16],
    "82": [0.7159, 0.032, 7.378e-16],
    "92": [0.7177, 0.033, 8.947e-16],
    "102": [0.7158, 0.032, 1.239e-15],
    "112": [0.7146, 0.028, 1.632e-15],
    "122": [0.7388, 0.026, 6.459e-16],
    "132": [0.7356, 0.020, 9.012e-16],
    "43": [0.7449, 0.045, 1.330e-16],  # Paschen Series
    "53": [0.7356, 0.044, 6.640e-16],
    "63": [0.7118, 0.016, 2.481e-15],
    "73": [0.7137, 0.029, 3.270e-15],
    "83": [0.7133, 0.032, 4.343e-15],
    "93": [0.7165, 0.033, 5.588e-15],
}


# Functions and classes
def stehle_param(
    n_upper, n_lower, cwl, wavelengths, electron_density, electron_temperature
):
    # Paramaterised MMM Stark profile coefficients from Bart's paper
    a_ij, b_ij, c_ij = loman_coeff[str(n_upper) + str(n_lower)]
    delta_lambda_12ij = (
        10.0
        * c_ij
        * divide((1e6 * electron_density) ** a_ij, electron_temperature**b_ij)
    )  # nm -> A
    gamma = delta_lambda_12ij / 2.0
    ls_s = 1 / (abs((wavelengths - cwl)) ** 2.5 + gamma**2.5)
    return ls_s / trapz(ls_s, wavelengths)


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
    wave_shift_sigma = abs(cwl - cwl / (1 - cwl * freq_shift_sigma))

    # relative intensities normalised to sum to one
    ls_sigma_minus = rel_intensity_sigma * interp(
        wavelengths + wave_shift_sigma, wavelengths, peak
    )
    ls_sigma_plus = rel_intensity_sigma * interp(
        wavelengths - wave_shift_sigma, wavelengths, peak
    )
    ls_pi = rel_intensity_pi * peak
    return ls_sigma_minus + ls_pi + ls_sigma_plus


class StarkShape:

    def __init__(self, cwl, wavelengths, n_upper, n_lower):
        # Cache args
        self.cwl = cwl
        self.wavelengths = wavelengths
        self.n_upper = n_upper
        self.n_lower = n_lower
        # Paramaterised MMM Stark profile coefficients from Bart's paper
        self.a_ij, self.b_ij, self.c_ij = loman_coeff[str(n_upper) + str(n_lower)]

    def __call__(self, electron_density, electron_temperature) -> ndarray:
        # Convert ne units from cm-3 to m-3
        electron_density *= 1e6
        # Calculate part of the denominator of the Stehle equation
        gamma = 0.5 * (
                10.0  # nm -> A
                * self.c_ij
                * divide(electron_density ** self.a_ij, electron_temperature ** self.b_ij)
        )
        # Calculate the line shape
        lineshape = 1 / (abs((self.wavelengths - self.cwl)) ** 2.5 + gamma ** 2.5)
        # Normalise
        lineshape /= trapz(lineshape, self.wavelengths)
        return lineshape


class HydrogenLineShape(object):
    def __init__(self, cwl, wavelengths, n_upper, n_lower, atomic_mass, zeeman=True):
        # Cache inputs
        self.cwl = cwl
        self.wavelengths = wavelengths
        self.zeeman = zeeman
        if int(n_lower) == 1:
            self.n_upper = n_upper + 1
            self.n_lower = 2
            UserWarning(
                "Using the Balmer Stark shape coefficients for the Lyman series. Transition n=%d -> %d | %f A"
                % (self.n_upper, self.n_lower, round(self.cwl, 2))
            )
        else:
            self.n_upper = n_upper
            self.n_lower = n_lower
        # Get lineshape components
        # Doppler
        dlambda = diff(self.wavelengths).mean()
        self.wavelengths_doppler = arange(
            self.cwl - 10, self.cwl + 10 + dlambda, dlambda
        )
        self.doppler_function = DopplerLine(
            cwl=copy(self.cwl),
            wavelengths=self.wavelengths_doppler,
            atomic_mass=atomic_mass,
            half_range=5000,
        )
        # Stark
        self.Stark = StarkShape(
            cwl, wavelengths, n_upper, n_lower
        )
        # Zeeman
        # self.get_delta_magnetic_quantum_number()
        self.bohr_magnaton = physical_constants["Bohr magneton in K/T"][0]

    def __call__(self, theta):
        # Unpack theta
        default_theta_length: int = 3
        electron_density, electron_temperature, ion_temperature = theta[
            :default_theta_length
        ]
        # Get the Doppler component
        self.doppler_component = self.doppler_function(ion_temperature, 1)
        # Get the Zeeman component
        if self.zeeman:
            b_field, viewangle = theta[default_theta_length:]
            self.doppler_component = zeeman_split(
                self.cwl,
                self.doppler_component,
                self.wavelengths_doppler,
                b_field,
                viewangle,
            )
        # Get the Stark Component
        self.stark_component = self.Stark(
            electron_density, electron_temperature
        )
        # Convolved and normalise components
        peak = fftconvolve(self.stark_component, self.doppler_component, "same")
        peak /= trapz(peak, self.wavelengths)

        return peak

    def gradient(self, theta: list[str]) -> ndarray:
        """
        Calculates the gradient of the HydrogenLineShape as a function of ne, Te, Ti, and if Zeeman splitting is
        included then also B and viewangle.

        :param (list[str]) theta:
            Parameters of the HydrogenLineShape {ne, Te, Ti} and {B, viewangle} if Zeeman splitting is included.

        :return (DataArray) gradient:
            Gradient of HydrogenLineShape as a function of its parameters {ne, Te, Ti} and {B, viewangle} if Zeeman splitting is included.
        """

        gradient: ndarray

        raise NotImplementedError

        return gradient


class BalmerHydrogenLine(object):
    def __init__(
        self, plasma, species, cwl, wavelengths, half_range=40000, zeeman=True
    ):
        self.plasma = plasma
        self.species = species
        self.element = species.split("_")[0]
        self.cwl = cwl
        self.line = self.species + "_" + str(self.cwl)
        self.wavelengths = wavelengths
        self.n_upper = self.plasma.input_dict[self.species][self.cwl]["n_upper"]
        self.n_lower = self.plasma.input_dict[self.species][self.cwl]["n_lower"]
        self.atomic_mass = atomic_masses[self.element]  # get_atomic_mass(self.species)
        self.los = self.plasma.profile_function.electron_density.x
        self.dl_per_sr = diff(self.los)[0] / (4 * pi)

        self.len_wavelengths = len(self.wavelengths)
        self.exc_pec = self.plasma.hydrogen_pecs[self.line + "_exc"]
        self.rec_pec = self.plasma.hydrogen_pecs[self.line + "_rec"]

        self.lineshape = HydrogenLineShape(
            self.cwl,
            self.wavelengths,
            self.n_upper,
            n_lower=self.n_lower,
            atomic_mass=self.atomic_mass,
            zeeman=zeeman,
        )

    def __call__(self):
        n1 = self.plasma.plasma_state["main_ion_density"]
        ne = self.plasma.plasma_state["electron_density"]
        te = self.plasma.plasma_state["electron_temperature"]
        n0 = self.plasma.plasma_state[self.species + "_dens"]

        self.n0_profile = n0

        from numpy import isnan

        if isnan(n0).any():
            raise ValueError(f"Negative numbers in neutral density profile! n0 = {n0}")

        if not self.plasma.zeeman:
            self.plasma.plasma_state["b-field"] = 0
            self.plasma.plasma_state["viewangle"] = 0

        bfield = self.plasma.plasma_state["b-field"]
        viewangle = self.plasma.plasma_state["viewangle"]

        if self.species + "_velocity" in self.plasma.plasma_state:
            self.velocity = self.plasma.plasma_state[self.species + "_velocity"]
            doppler_shift_BalmerHydrogenLine(self, self.velocity)

        interp_args: dict = dict(
            ne=("pecs", ne),
            Te=("pecs", te),
            kwargs=dict(bounds_error=False, fill_value=None),
        )
        rec_pec: DataArray = exp(self.rec_pec.interp(**interp_args))  # .data
        exc_pec: DataArray = exp(self.exc_pec.interp(**interp_args))  # .data
        # Add los as a coordinate for the PECs dimention
        rec_pec = rec_pec.assign_coords(los=("pecs", self.plasma.los))
        exc_pec = exc_pec.assign_coords(los=("pecs", self.plasma.los))

        if isnan(exc_pec).any():
            err_msg: str = "NaNs in the excitation PECs!"
            raise ValueError(err_msg)

        if isnan(rec_pec).any():
            err_msg: str = "NaNs in the recombination PECs!"
            raise ValueError(err_msg)

        # set minimum number of photons to be 1
        # need to exclude antiprotons from emission!
        self.rec_profile = n1.clip(1) * ne * rec_pec  # ph/cm-3/s
        self.exc_profile = n0 * ne * exc_pec  # ph/cm-3/s
        self.rec_ems_weights = self.rec_profile / self.rec_profile.sum()
        self.exc_ems_weights = self.exc_profile / self.exc_profile.sum()

        self.rec_sum = trapz(self.rec_profile, x=self.plasma.los) / (
            4 * pi
        )  # ph/cm-2/sr/s
        self.exc_sum = trapz(self.exc_profile, x=self.plasma.los) / (
            4 * pi
        )  # ph/cm-2/sr/s
        self.ems_profile = self.rec_profile + self.exc_profile
        self.emission_fitted = trapz(self.ems_profile, x=self.plasma.los) / (4 * pi)

        # used for the emission lineshape calculation
        self.exc_ne = dot(self.exc_ems_weights, ne)
        self.exc_te = dot(self.exc_ems_weights, te)
        self.exc_n0 = dot(self.exc_ems_weights, self.n0_profile)
        self.exc_n0_frac = dot(self.exc_ems_weights, self.n0_profile / ne)
        self.exc_n0_frac_alt = self.exc_n0 / self.exc_ne

        self.rec_ne = dot(self.rec_ems_weights, ne)
        self.rec_te = dot(self.rec_ems_weights, te)

        # just because there are nice to have
        self.f_rec = self.rec_sum / self.emission_fitted
        self.ems_ne = dot(self.ems_profile, ne) / self.ems_profile.sum()
        self.ems_te = dot(self.ems_profile, te) / self.ems_profile.sum()

        if self.plasma.cold_neutrals:
            self.plasma.plasma_state[self.species + "_Ti"] = 0.01
        elif self.plasma.thermalised:
            thermalised_ti = (1 - self.f_rec) * self.exc_te + self.f_rec * self.rec_te
            self.plasma.plasma_state[self.species + "_Ti"] = thermalised_ti

        self.exc_ti = self.plasma.plasma_state[self.species + "_Ti"]
        self.rec_ti = self.plasma.plasma_state[self.species + "_Ti"]

        self.exc_lineshape_input = [
            self.exc_ne,
            self.exc_te,
            self.exc_ti,
            bfield,
            viewangle,
        ]
        self.rec_lineshape_input = [
            self.rec_ne,
            self.rec_te,
            self.rec_ti,
            bfield,
            viewangle,
        ]
        self.exc_lineshape: ndarray = nan_to_num(
            self.lineshape(self.exc_lineshape_input)
        )
        self.rec_lineshape: ndarray = nan_to_num(
            self.lineshape(self.rec_lineshape_input)
        )

        self.exc_peak = self.exc_lineshape * self.exc_sum
        self.rec_peak = self.rec_lineshape * self.rec_sum

        self.ems_peak = self.rec_peak + self.exc_peak

        if isnan(self.ems_peak).any():
            raise ValueError(
                f"NaNs in {self.line} line peak ({self.exc_sum}, {self.rec_sum})!"
            )

        return self.ems_peak  # ph/cm-2/A/sr/s

    def gradient(self) -> ndarray:
        # Shortcut for cached values
        _ = self()
        # Building variables
        shape: tuple[int, int] = (self.plasma.n_params, *self.wavelengths.shape)
        gradient: ndarray = empty(shape)
        gradient[:, :] = nan
        # Electron density
        ne_key: str = "electron_density"
        gradient[self.plasma.slices[ne_key]] = self.electron_density_gradient()
        # Electron temperature
        te_key: str = "electron_temperature"
        gradient[self.plasma.slices[te_key]] = self.electron_temperature_gradient()
        # Neutral density
        n0_key: str = f"{self.species}_dens"
        if n0_key in self.plasma.slices:
            gradient[self.plasma.slices[n0_key]] = log(10) * self.exc_peak
        # Ionisation tau
        tau_key: str = f"{self.species}_tau"
        gradient[self.plasma.slices[tau_key]] = self.tau_gradient()
        # Doppler temperature
        doppler_temperature_key: str = f"{self.species}_velocity"
        # Doppler velocity
        velocity_key: str = f"{self.species}_velocity"
        if velocity_key in self.plasma.slices:
            raise NotImplementedError(
                "No gradients calculation for the Doppler velocity in BalmerHydrogenLine!"
            )

        raise NotImplementedError

        return gradient

    def electron_density_gradient(self) -> ndarray:
        # Get the dne/dtheta for the chain rule calculation
        ne_theta_grad: ndarray = self.plasma.profile_function.electron_density.gradient(
            self.plasma.plasma_theta["electron_density"]
        )
        # Calculate the excitation component
        ems_grad: ndarray = trapz(
            self.exc_profile.diff(dim="ne").data[None, :] * ne_theta_grad,
            x=self.plasma.los,
        )
        peak_shape_grad: ndarray = nan_to_num(
            self.lineshape.gradient(self.exc_lineshape_input)
        )
        excitation: ndarray = (
            self.exc_lineshape * ems_grad + peak_shape_grad * self.exc_sum
        )
        # Calculate the recombintion component
        ems_grad: ndarray = trapz(
            self.exc_profile.diff(dim="ne").data[None, :] * ne_theta_grad,
            x=self.plasma.los,
        )
        peak_shape_grad: ndarray = nan_to_num(
            self.lineshape.gradient(self.exc_lineshape_input)
        )
        recombination: ndarray = (
            self.rec_lineshape * ems_grad + peak_shape_grad * self.rec_sum
        )
        # Sum and return
        gradient: ndarray = excitation + recombination

        return gradient

    def electron_temperature_gradient(self) -> ndarray:
        gradient: ndarray

        raise NotImplementedError

        return gradient

    def tau_gradient(self) -> ndarray:
        # Plasma parameters
        ne = self.plasma.plasma_state["electron_density"]
        te = self.plasma.plasma_state["electron_temperature"]
        n0: ndarray = power(10, self.plasma.plasma_theta[f"{self.species}_dens"])
        # Reaction rates
        interp_args: dict = dict(
            ne=("pecs", ne),
            Te=("pecs", te),
            kwargs=dict(bounds_error=False, fill_value=None),
        )
        pec: ndarray = exp(self.exc_pec.interp(**interp_args)).data
        # Calculate gradient
        n0_dtau: ndarray = log(10) * n0 * ne * self.plasma.scd
        emission_profile_dtau: ndarray = n0_dtau * ne * pec
        gradient: ndarray = self.exc_lineshape * trapz(emission_profile_dtau)

        return gradient


def main() -> None:
    pass


if __name__ == "__main__":
    main()
