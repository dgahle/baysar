# Imports
from numpy import array, isnan, linspace, log10, ndarray

from baysar.plasmas.profile_functions.asymmetric_cauchy import asymmetric_cauchy
from baysar.plasmas.profile_functions.asymmetric_cauchy import asymmetric_cauchy_d_log_p_max, asymmetric_cauchy_d_shift, asymmetric_cauchy_d_sigma, asymmetric_cauchy_d_p_min


# Variables
x: ndarray = linspace(-5, 15, 101)
log_p_max: float = 10.0
shift: float = 2.0
sigma: float = 1.5
p_min: float = 0.0


# Functions and Classes
class TestAsymmetricCauchy:

    def test_peak_value(self) -> None:
        # Variables
        log_p_max: ndarray = array([13, 14, 15])
        # Calculation
        profile: ndarray = array([
            asymmetric_cauchy(x, _log_p_max, shift, sigma, p_min) for _log_p_max in log_p_max
        ])
        # Checks
        test: ndarray = log10(profile.max(1))
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


if __name__ == "__main__":
    pass
