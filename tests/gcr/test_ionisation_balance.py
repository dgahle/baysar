# Imports
from pathlib import Path
from xarray import DataArray, load_dataarray

from gcr import ionisation_balance

# Variables


# Functions and classes
def test_ionisation_balance() -> None:
    element: str = 'h'
    fractional_abundance: DataArray = ionisation_balance(element)
    # Test plotting lines
    # h_ionisation_balance: Path = Path(__file__).parent / "h_ionisation_balance_test_reference.nc"
    h_ionisation_balance: str = "/Users/daljeet-singh-gahle/github/baysar/tests/gcr/h_ionisation_balance_test_reference.nc"
    reference: DataArray = load_dataarray(h_ionisation_balance)
    assert fractional_abundance.interp(ne=1e14).equals(reference)


def main() -> None:
    pass


if __name__ == "__main__":
    pass
