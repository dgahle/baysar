# Imports
from OpenADAS import get_adf15, load_adf15, read_adf15
from numpy import array, ndarray
from xarray import DataArray


# Variables


# Functions and classes
class TestLoadAdf15:

    def test_C_III(self):
        adf15: str = get_adf15(element='c', charge=2)
        adf15_model: DataArray = load_adf15(adf15, passed=True)
        assert type(adf15_model) is DataArray

    def test_N_II(self):
        adf15: str = get_adf15(element='n', charge=1)
        adf15_model: DataArray = load_adf15(adf15, passed=True)
        assert type(adf15_model) is DataArray

    def test_Ne_III(self):
        adf15: str = get_adf15(element='ne', charge=2, visible=False)
        adf15_model: DataArray = load_adf15(adf15, passed=True)
        assert type(adf15_model) is DataArray

    def test_H_I(self):
        adf15: str = get_adf15(element='h', charge=0, year=12)
        adf15_model: DataArray = load_adf15(adf15, passed=True)
        assert type(adf15_model) is DataArray


class TestReadAdf15:

    def test_C_III(self):
        ne: ndarray = array([1e13, 1e14, 1e15])
        te: ndarray = array([3, 5, 10], dtype=float)
        adf15: str = get_adf15(element='c', charge=2)
        adf15_model: DataArray = read_adf15(adf15, block=1, ne=ne, te=te, passed=True)
        assert type(adf15_model) is DataArray

    def test_N_II(self):
        ne: ndarray = array([1e13, 1e14, 1e15])
        te: ndarray = array([3, 5, 10], dtype=float)
        adf15: str = get_adf15(element='n', charge=1)
        adf15_model: DataArray = read_adf15(adf15, block=1, ne=ne, te=te, passed=True)
        assert type(adf15_model) is DataArray


    def test_Ne_III(self):
        ne: ndarray = array([1e13, 1e14, 1e15])
        te: ndarray = array([3, 5, 10], dtype=float)
        adf15: str = get_adf15(element='ne', charge=2, visible=False)
        adf15_model: DataArray = read_adf15(adf15, block=1, ne=ne, te=te, passed=True)
        assert type(adf15_model) is DataArray

    def test_H_I(self):
        ne: ndarray = array([1e13, 1e14, 1e15])
        te: ndarray = array([3, 5, 10], dtype=float)
        adf15: str = get_adf15(element='h', charge=0, year=12)
        adf15_model: DataArray = read_adf15(adf15, block=1, ne=ne, te=te, passed=True)
        assert type(adf15_model) is DataArray


def main() -> None:
    pass


if __name__ == "__main__":
    main()
