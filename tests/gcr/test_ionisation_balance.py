# Imports
from pathlib import Path

from xarray import DataArray, load_dataarray

from gcr import ionisation_balance, ionisation_balance_transport

# Variables
DATA_FOLDER: str = "/Users/daljeet-singh-gahle/github/baysar/tests/gcr/data"


# Functions and classes
def ionisation_balance_test(element: str) -> None:
    fractional_abundance: DataArray = ionisation_balance(element)
    # Unit test
    data_path: str = f"{DATA_FOLDER}/{element}_ionisation_balance_test_reference.nc"
    reference: DataArray = load_dataarray(data_path)
    assert fractional_abundance.interp(ne=1e14).equals(reference)


class TestIonisationBalance:
    def test_h_ionisation_balance(self) -> None:
        element: str = "h"
        ionisation_balance_test(element)

    def test_he_ionisation_balance(self) -> None:
        element: str = "he"
        ionisation_balance_test(element)

    def test_ne_ionisation_balance(self) -> None:
        element: str = "ne"
        ionisation_balance_test(element)


def ionisation_balance_transport_test(
    element: str, tau: float = 0.01, tolerance: float = 1e-10
) -> None:
    # Get transport and steady state ionisation balance
    fractional_abundance: DataArray = ionisation_balance(element, tau=tau)
    fractional_abundance_transport: DataArray = ionisation_balance_transport(element)
    # Calculate errors
    f_ion_diff: DataArray = abs(
        fractional_abundance - fractional_abundance_transport.interp(tau=tau)
    )
    diff_mean: float = f_ion_diff.mean().data[()]
    diff_std: float = f_ion_diff.mean().data[()]
    # Checks
    check_mean: bool = diff_mean < tolerance
    check_std: bool = diff_std < tolerance
    checks: list[bool] = [check_mean, check_std]
    # Assert
    assert_msg: str = f"{element.capitalize()} tau-balance precision: mean is {diff_mean:.2e} and std {diff_std:.2e}!"
    assert all(checks), assert_msg


class TestIonisationBalanceTransport:
    def test_h_ionisation_balance_transport(self) -> None:
        # Get transport and steady state ionisation balance
        element: str = "h"
        ionisation_balance_transport_test(element)

    def test_he_ionisation_balance_transport(self) -> None:
        # Get transport and steady state ionisation balance
        element: str = "he"
        ionisation_balance_transport_test(element)

    def test_c_ionisation_balance_transport(self) -> None:
        # Get transport and steady state ionisation balance
        element: str = "c"
        ionisation_balance_transport_test(element)


def main() -> None:
    pass


if __name__ == "__main__":
    pass
