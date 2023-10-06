# Imports
from numpy import exp, linspace, log, ndarray, power, sqrt, square

# Variables


# Functions
def asymmetric_cauchy(
    x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float
) -> ndarray:
    # Calculate the amplitude
    A: float = power(10, log_p_max) - p_min
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-(x - shift)))
    C: ndarray = sigma + (x - shift) / B
    # Calculate the profile
    D: ndarray = square((x - shift) / C)
    profile: ndarray = p_min + A / (1 + D)

    return profile


def asymmetric_cauchy_d_log_p_max(
    x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float
) -> ndarray:
    # Calculate the amplitude
    A: float = log(10) * power(10, log_p_max)
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-(x - shift)))
    C: ndarray = sigma + (x - shift) / B
    # Calculate the profile
    D: ndarray = square((x - shift) / C)
    gradient_profile: ndarray = A / (1 + D)

    return gradient_profile


def asymmetric_cauchy_d_shift(
    x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float
) -> ndarray:
    # Solve dB/dShift by substitution
    dx: ndarray = x - shift
    G: ndarray = 1 + exp(-dx)
    dB_dG: ndarray = -sigma / square(G)
    dG_dshift: ndarray = exp(-dx)
    dB_dshift: ndarray = dB_dG * dG_dshift
    # Solve dC/dShift by quotient rule
    B: ndarray = 0.2 * sigma + sigma / G
    dC_dshift: ndarray = -(B + dx * dB_dshift) / square(B)
    # Solve dD/dShift by substitution and quotient rule
    C: ndarray = sigma + dx / B
    dD_dshift: ndarray = 2 * (dx / C) * -(C + dC_dshift * dx) / square(C)
    # Solve dE/dShift
    D: ndarray = square(dx / C)
    dE_dshift: ndarray = dD_dshift
    # Solve dy/dShift by substitution
    E: ndarray = 1 + D
    A: ndarray = power(10, log_p_max) - p_min
    dy_dE: ndarray = -A / square(E)
    gradient_profile: ndarray = dy_dE * dE_dshift

    return gradient_profile


def asymmetric_cauchy_d_sigma(
    x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float
) -> ndarray:
    # Calculate terms needed form the original asymmetric_cauchy function
    dx: ndarray = x - shift
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-dx))
    C: ndarray = sigma + dx / B
    D: ndarray = square(dx / C)
    A: ndarray = power(10, log_p_max) - p_min
    # Predefine dB/dSigma to make the final code/calculation easier to read
    dB_dsigma: ndarray = 0.2 + 1 / (1 + exp(-dx))
    # Calculate the gradients over the profile
    lhs0: ndarray = sqrt(A) / (1 + D)
    lhs: ndarray = 2 * power(C, -3) * square(lhs0 * dx)
    rhs: ndarray = 1 - (((x - shift) / square(B)) * dB_dsigma)
    gradient_profile: ndarray = rhs * lhs

    return gradient_profile


def asymmetric_cauchy_d_p_min(
    x: ndarray, log_p_max: float, shift: float, sigma: float, p_min: float
) -> ndarray:
    """
    (d/dpmin)f(x;theta)  = 1 - 1 / (1 + D), where f(x; theta) is the asymmetric_cauchy function.
    """
    # Calculate the broadening
    B: ndarray = 0.2 * sigma + sigma / (1 + exp(-(x - shift)))
    C: ndarray = sigma + (x - shift) / B
    # Calculate the profile
    D: ndarray = square((x - shift) / C)
    gradient_profile: ndarray = 1 - 1 / (1 + D)

    return gradient_profile


class ElectronDensity:

    def __init__(self, x: ndarray):
        self.x: ndarray = x
        self.number_of_variables: int = 3
        self.bounds: list[list[int]] = [
            [12, 15],
            [-2, 3],
            [0.5, 5]
        ]

    def __call__(self, theta: list[float]) -> ndarray:
        log_p_max, shift, sigma = theta
        return asymmetric_cauchy(self.x, log_p_max=log_p_max, shift=shift, sigma=sigma, p_min=0.)


class ElectronTemperature:

    def __init__(self, x: ndarray):
        self.x: ndarray = x
        self.number_of_variables: int = 3
        self.bounds: list[list[int]] = [
            [-1, 2],
            [0.5, 5],
            [0.5, 5]
        ]

    def __call__(self, theta: list[float]) -> ndarray:
        log_p_max, sigma, p_min = theta
        return asymmetric_cauchy(self.x, log_p_max=log_p_max, shift=0., sigma=sigma, p_min=p_min)


class AsymmetricCauchyProfile:
    def __init__(self, x=None):
        self.x: ndarray = linspace(-10, 25, 101) if x is None else x
        self.electron_density: ElectronDensity = ElectronDensity(self.x)
        self.electron_temperature: ElectronTemperature = ElectronTemperature(self.x)

    def _asymmetric_cauchy(self, theta: list[float]) -> ndarray:
        profile: ndarray = asymmetric_cauchy(self.x, *theta)
        return profile


if __name__ == "__main__":
    pass
