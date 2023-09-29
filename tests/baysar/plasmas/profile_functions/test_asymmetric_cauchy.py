# Imports
from numpy import arange, array, isnan, linspace, log10, ndarray, product
from pandas import DataFrame, Series
from scipy.optimize import approx_fprime

from baysar.plasmas.profile_functions.asymmetric_cauchy import asymmetric_cauchy, AsymmetricCauchyProfile
from baysar.plasmas.profile_functions.asymmetric_cauchy import asymmetric_cauchy_d_log_p_max, asymmetric_cauchy_d_shift
from baysar.plasmas.profile_functions.asymmetric_cauchy import asymmetric_cauchy_d_sigma, asymmetric_cauchy_d_p_min

# Variables
x: ndarray = linspace(-5, 15, 101)
log_p_max: float = 10.0
shift: float = 2.0
sigma: float = 1.5
p_min: float = 1.0
# Variables - test_against_approx_fprime
theta: list[float] = [log_p_max, shift, sigma, p_min]
gradient_functions: list[callable] = [
    asymmetric_cauchy_d_log_p_max,
    asymmetric_cauchy_d_shift,
    asymmetric_cauchy_d_sigma,
    asymmetric_cauchy_d_p_min
]
default_epsilon: float = 1.49e-08
epsilon: ndarray[float] = array([
    default_epsilon,
    default_epsilon,
    default_epsilon,
    1e2])


# Functions and Classes
def calculate_tolerance(reference: [float, ndarray], rtol: float = 1e-05, atol: float = 1e-08):
    """
    Use the tolerance calculation from numpy.isclose to get a standard deviation to calculate Gaussian errors for
    fractional equivalence tests within machine error.

    Useful reference: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html

    Args:
        reference (float, array_like): Variable of which to name
        rtol (float) = 1e-05: Tolerance relative to the reference
        atol (float) = 1e-08: Absolute tolerance

    Returns:
        sigma = atol + rtol * abs(reference)
    """

    return atol + rtol * abs(reference)


class TestCalculateTolerance:

    def test_with_zero(self):
        # Test
        test: float = calculate_tolerance(reference=0.)
        # Reference
        reference: float = 1e-08  # same as atol in the function
        # Check
        assert test == reference, f'Does not work with reference = {reference}!'

    def test_with_arange(self):
        # Variables
        data: ndarray[float] = arange(2)
        rtol: float = 1e-05
        atol: float = 1e-08
        # Test
        test: ndarray = calculate_tolerance(data)
        # Reference
        reference: ndarray = array([
            atol,
            atol + rtol
        ])
        # Check
        check: ndarray = test == reference
        assert check.all(), f'Does not work with reference = {reference}!'


def get_name(var: any) -> str:
    """
    Uses the `id` function to match the passed variable against the contents of globals().

    Note: This has only been tested against returning the names of variables in a list comprehension

    Args:
        var (any): Variable of which to name

    Returns:
        name (str): Name of the var matched against globals
    """

    name: str = [i for i, j in globals().items() if id(j) == id(var)][0]

    return name


def test_get_name() -> None:
    # Test
    theta: list[float] = [log_p_max, shift, sigma, p_min]
    test: list[str] = [get_name(var) for var in theta]
    # Reference
    reference: list[str] = ["log_p_max", "shift", 'sigma', "p_min"]
    # Check
    check: list[bool] = [t==r for t, r in zip(test, reference)]
    assert_msg: str = ""
    assert all(check), assert_msg


class TestAsymmetricCauchy:

    def test_peak_value(self) -> None:
        # Variables
        log_p_max: ndarray = array([13, 14, 15])
        # Calculation
        profile: ndarray = array([
            asymmetric_cauchy(x, _log_p_max, shift, sigma, p_min) for _log_p_max in log_p_max
        ])
        # Checks
        test: ndarray = log10(
            profile.max(axis=1)
        )
        checks: bool = (log_p_max - test).sum() == 0.0
        # Assert
        assert_msg: str = f"Profile peaks are {', '.join([f'{v:.2f}' for v in test])} " \
                          f"but should be {', '.join([f'{v}' for v in log_p_max])}!"
        assert checks, assert_msg

    def test_peak_position(self) -> None:
        # Variables
        shift: ndarray = array([-2.0, 0.0, 1.2])
        # Calculation
        profile: ndarray = array([
            asymmetric_cauchy(x, log_p_max, _shift, sigma, p_min) for _shift in shift
        ])
        # Checks
        test: ndarray = x[profile.argmax(1)]
        checks: bool = 1e-9 > (shift - test).sum()
        # Assert
        assert_msg: str = f"Profile centres are {', '.join([f'{v:.2f}' for v in test])} " \
                          f"but should be {', '.join([f'{v}' for v in shift])}!"
        assert checks, assert_msg

    def test_peak_min(self) -> None:
        # Variables
        p_min: ndarray = array([-2.0, 0.0, 1.2])
        # Calculation
        profile: ndarray = array([
            asymmetric_cauchy(x, log_p_max, shift, sigma, _p_min) for _p_min in p_min
        ])
        # Checks
        test: ndarray = x[profile.argmax(1)]
        checks: bool = 0.0 == (shift - test).sum()
        # Assert
        assert_msg: str = f"Profile minimum values are {', '.join([f'{v:.2f}' for v in test])} " \
                          f"but should be {', '.join([f'{v}' for v in p_min])}!"
        assert checks, assert_msg


class TestAsymmetricCauchyGradients:

    def test_asymmetric_cauchy_d_log_p_max(self) -> None:
        test: ndarray = asymmetric_cauchy_d_log_p_max(x, log_p_max, shift, sigma, p_min)
        check: ndarray = isnan(test)
        assert not check.any(), 'asymmetric_cauchy_d_log_p_max is returning NaNs!'

    def test_asymmetric_cauchy_d_shift(self) -> None:
        test: ndarray = asymmetric_cauchy_d_shift(x, log_p_max, shift, sigma, p_min)
        check: ndarray = isnan(test)
        assert not check.any(), 'asymmetric_cauchy_d_shift is returning NaNs!'

    def test_asymmetric_cauchy_d_sigma(self) -> None:
        test: ndarray = asymmetric_cauchy_d_sigma(x, log_p_max, shift, sigma, p_min)
        check: ndarray = isnan(test)
        assert not check.any(), 'asymmetric_cauchy_d_sigma is returning NaNs!'

    def test_asymmetric_cauchy_d_p_min(self) -> None:
        test: ndarray = asymmetric_cauchy_d_p_min(x, log_p_max, shift, sigma, p_min)
        check: ndarray = isnan(test, where=False)
        assert not check.any(), 'asymmetric_cauchy_d_p_min is returning NaNs!'

    def test_against_approx_fprime(self) -> None:
        # Calculate the gradient using approx_fprime
        profile_function: AsymmetricCauchyProfile = AsymmetricCauchyProfile()
        reference: ndarray = approx_fprime(theta, profile_function._asymmetric_cauchy, epsilon=epsilon)
        # Calculate gratient with local functions
        test: ndarray = array([
            f(profile_function.x, *theta) for f in gradient_functions
        ]).T
        # Check
        error_fraction: ndarray = (test - reference) / calculate_tolerance(reference)
        check: ndarray = abs(error_fraction) < 1.0
        if check.all():
            # Construct a summary DataFrame to present errors
            ndarray_functions: list[str] = ['mean', 'std', 'min', 'max']
            error_fraction_reduced_summary: list[ndarray] = [
                getattr(error_fraction, f)(axis=0) for f in ndarray_functions
            ]
            theta_names: list[str] = [get_name(var) for var in theta]
            df_error: DataFrame = DataFrame(
                error_fraction_reduced_summary,
                columns=theta_names,
                index=ndarray_functions,
            )
            df_error.index.name = 'theta'
            # Write Assert message
            success_pc: float = 100 * check.sum() / product(check.shape)
            assert_msg: str = f'{100 - success_pc:.2f} % results out of tolerance in the gradient calculation!`\n' \
                              'Tolerance Error Summary (Acceptable range 0 +/- 1):\n' \
                              f'{df_error}'

            assert check.all(), assert_msg


if __name__ == "__main__":
    pass
