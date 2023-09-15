# Imports
from numpy import exp, ndarray, power, square


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
        pass


if __name__ == "__main__":
    pass
