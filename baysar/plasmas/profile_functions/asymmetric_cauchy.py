# Imports
from numpy import exp, linspace, log, ndarray, power, square


# Variables


# Functions
def asymmetric_cauchy(x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float) -> ndarray:
    # Calculate the amplitude
    A: float = power(10, log_p_max) - p_min
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-(x - shift)))
    C: ndarray = sigma + (x - shift) / B
    # Calculate the profile
    D: ndarray = square((x - shift) / C)
    profile: ndarray = p_min + A / (1 + D)

    return profile


def asymmetric_cauchy_d_log_p_max(x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float) -> ndarray:
    # Calculate the amplitude
    A: float = log(10) * power(10, log_p_max)
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-(x - shift)))
    C: ndarray = sigma + (x - shift) / B
    # Calculate the profile
    D: ndarray = square((x - shift) / C)
    gradient_profile: ndarray = A / (1 + D)

    return gradient_profile


def asymmetric_cauchy_d_shift(x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float) -> ndarray:
    # Solve dB/dShift by substitution
    dx: ndarray = x - shift
    G: ndarray = 1 + exp(-dx)
    dB_dG: ndarray = - sigma / square(G)
    dG_dshift: ndarray = exp(-dx)
    dB_dshift: ndarray = dB_dG * dG_dshift
    # Solve dC/dShift by quotient rule
    B: ndarray = 0.2 * sigma + sigma / G
    dC_dshift: ndarray = - (B + dx * dB_dshift) / square(B)
    # Solve dD/dShift by substitution and quotient rule
    C: ndarray = sigma + dx / B
    dD_dshift: ndarray = 2 * (dx / C) * - (C + dC_dshift * dx) / square(C)
    # Solve dE/dShift
    D: ndarray = square(dx / C)
    dE_dshift: ndarray = dD_dshift
    # Solve dy/dShift by substitution
    E: ndarray = 1 + D
    A: ndarray = power(10, log_p_max) - p_min
    dy_dE: ndarray = - A / square(E)
    gradient_profile: ndarray = dy_dE * dE_dshift

    return gradient_profile


def asymmetric_cauchy_d_sigma(x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float) -> ndarray:
    # Solve dB/dSigma by substitution
    dx: ndarray = x - shift
    dB_dsigma: ndarray = 0.2 + 1 / (1 + exp(-dx))
    # Solve dC/dSigma by quotient rule
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-dx))
    dC_dsigma: ndarray = 1 - dx / square(B) * dB_dsigma
    # Solve dD/dSigma by substitution and quotient rule
    C: ndarray = sigma + dx / B
    D: ndarray = square(dx / C)
    dD_dsigma: ndarray = 2 * D * - dx / square(C) * dC_dsigma
    # Solve dE/dSigma
    dE_dsigma: ndarray = dD_dsigma
    # Solve dy/dSigma by substitution
    E: ndarray = 1 + D
    A: ndarray = power(10, log_p_max) - p_min
    dy_dE: ndarray = - A / square(E)
    gradient_profile: ndarray = dy_dE * dE_dsigma

    return gradient_profile


def asymmetric_cauchy_d_p_min(x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float) -> ndarray:
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-(x - shift)))
    C: ndarray = sigma + (x - shift) / B
    # Calculate the profile
    D: ndarray = square((x - shift) / C)
    gradient_profile: ndarray = 1 - 1 / (1 + D)

    return gradient_profile


def _electron_density(x: ndarray, log_ne_max: float, shift: float, sigma: float) -> ndarray:
    # Calculate the amplitude
    A: float = power(10, log_ne_max)
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-(x - shift)))
    C: ndarray = sigma + (x - shift) / B
    # Calculate the profile
    profile: ndarray = A / (1 + (x - shift) / C)

    return profile


def _electron_temperature(x: ndarray, log_te_max: float, sigma: float, te_min: float) -> ndarray:
    # Calculate the amplitude
    A: float = power(10, log_te_max) - te_min
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-x))
    C: ndarray = sigma + x / B
    # Calculate the profile
    profile: ndarray = te_min + A / (1 + x / C)

    return profile


class AsymmetricCauchyProfile:

    def __init__(self):
        self.x: ndarray = linspace(-5, 15, 101)
        pass

    def _asymmetric_cauchy(self, theta: list[float]) -> ndarray:
        profile: ndarray = asymmetric_cauchy(self.x, *theta)
        return profile


if __name__ == "__main__":
    pass
