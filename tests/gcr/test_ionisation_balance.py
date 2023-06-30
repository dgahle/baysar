# Imports
from pathlib import Path

from xarray import DataArray, load_dataarray

from gcr import ionisation_balance

# Variables
DATA_FOLDER: str = "/Users/daljeet-singh-gahle/github/baysar/tests/gcr/data"


# Functions and classes
class TestIonisationBalance:
    def test_h_ionisation_balance(self) -> None:
        element: str = "h"
        fractional_abundance: DataArray = ionisation_balance(element)
        # Unit test
        data_path: str = f"{DATA_FOLDER}/{element}_ionisation_balance_test_reference.nc"
        reference: DataArray = load_dataarray(data_path)
        assert fractional_abundance.interp(ne=1e14).equals(reference)

    def test_he_ionisation_balance(self) -> None:
        element: str = "he"
        fractional_abundance: DataArray = ionisation_balance(element)
        # Unit test
        data_path: str = f"{DATA_FOLDER}/{element}_ionisation_balance_test_reference.nc"
        reference: DataArray = load_dataarray(data_path)
        assert fractional_abundance.interp(ne=1e14).equals(reference)

    def test_ne_ionisation_balance(self) -> None:
        element: str = "ne"
        fractional_abundance: DataArray = ionisation_balance(element)
        # Unit test
        data_path: str = f"{DATA_FOLDER}/{element}_ionisation_balance_test_reference.nc"
        reference: DataArray = load_dataarray(data_path)
        assert fractional_abundance.interp(ne=1e14).equals(reference)


def main() -> None:
    pass


if __name__ == "__main__":
    pass
