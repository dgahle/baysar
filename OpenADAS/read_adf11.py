#
#
#
#
#
# Imports
from numpy import arange, array, ceil, ndarray, power
from pathlib import Path
from xarray import DataArray


# Variables
ADAS_PATH: Path = Path(__file__).parent.parent / 'adas'
Adf11_SEPARATOR: str = 'C-----------------------------------------------------------------------'


# Functions
def build_adf11_dataarray(adf11_raw: str) -> DataArray:
    # Get the line of the first block, this is needed to extract both the {ne, Te} grid and the rates
    first_block_str: str = "---------------------/ "  # IPRT= "  # 1  / IGRD= 1 "
    check: list = [line.startswith(first_block_str) for line in adf11_raw.split('\n')]
    check_indices: ndarray = arange(len(check))[check]
    if check_indices.shape == (0,):
        raise ValueError("Could not find the first block of rates in the adf11 file!")
    start_block: int = check_indices[0]
    # Get the data shape (block, ne, Te)
    data_shape: tuple = tuple(int(n) for n in adf11_raw.split('\n')[0].split()[:3])
    # Get the (ne, Te) grid
    flat_grid: list = ' '.join(adf11_raw.split('\n')[:start_block]).split('---')[-1].split()[1:]
    flat_grid: ndarray = array(
        [float(n) for n in flat_grid]
    )
    ne: ndarray = flat_grid[:data_shape[1]]
    te: ndarray = flat_grid[data_shape[1]:]
    # Get the number of lines in each block
    len_block: int = int(
        ceil(data_shape[1] * data_shape[2] / 8)
    )
    # Extract rates
    rates: list = []
    block: int
    for block in range(data_shape[0]):
        # Calculate lines of adf11 to extract
        # block += 1
        lhs: int = 1 + start_block + block * (1 + len_block)
        rhs: int = lhs + len_block
        index: slice = slice(lhs, rhs)
        # Extract and format rates
        _rates: list = adf11_raw.split('\n')[index]
        _rates: str = " ".join(_rates)
        _rates: ndarray = array(
            [float(r) for r in _rates.split()]
        )
        _rates = _rates.reshape(data_shape[1:][::-1]).T
        rates.append(_rates)
    # Format rates to build DataArray
    rates: ndarray = power(10, rates)
    dims: tuple = ('block', 'ne', 'Te')
    coords: dict = dict(
        block=1 + arange(data_shape[0]),
        ne=power(10, ne),
        Te=power(10, te)
    )
    attrs: dict = dict(
        description='TBC',  # Adf11_SEPARATOR.join(adf11_raw_list[1:]),
        units='cm3/s',
        adf=adf11_raw
    )
    data_adf11: DataArray = DataArray(
        rates,
        dims=dims,
        coords=coords,
        name='pecs',
        attrs=attrs
    )

    return data_adf11


def load_adf11(adf11: [str, Path], passed: bool = False) -> DataArray:
    # Load as text file
    if not passed:
        with open(adf11, 'r') as f:
            adf11_raw: str = f.read()
    else:
        adf11_raw: str = adf11
    # Separate by block nuber
    adf11_model: DataArray = build_adf11_dataarray(adf11_raw)
    return adf11_model


def read_adf11(adf11: [str, Path],
               block: [int, list],
               ne: ndarray,
               te: ndarray,
               passed: bool = False) -> DataArray:
    """
    Reads ADAS formatted Adf11 (PEC files) and returns

    :param adf11:
    :param block:
    :param te:
    :param ne:
    :param passed:
    :return:
    """
    # Load DataArray
    adf11_model: DataArray = load_adf11(adf11, passed=passed)
    # Interpolate to get desired values
    block: list = [block] if type(block) is int else block
    kwargs: dict = dict(fill_value="extrapolate")
    pecs_out: DataArray = adf11_model.sel(block=block).interp(ne=ne, Te=te, kwargs=kwargs)
    return pecs_out


def main() -> None:
    pass


if __name__ == "__main__":
    main()
    