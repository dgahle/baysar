#
#
#
#
#
# Imports
from numpy import array, ndarray, round
from pathlib import Path
from xarray import concat, DataArray


# Variables
ADAS_PATH: Path = Path(__file__).parent.parent / 'adas'
ADF15_SEPARATOR: str = 'C-----------------------------------------------------------------------'


# Functions
def get_number_of_blocks(adf15_raw: str) -> int:
    header: str = adf15_raw.split('\n')[0]
    total_blocks: int = int(
        header.split()[0]
    )
    return total_blocks


def block_str_to_array(adf15_block: list[str]) -> DataArray:
    header: list[str] = adf15_block[0].split()
    num_ne_index: int = 2 if header[1] == 'A' else 1
    num_te_index: int = 3 if header[1] == 'A' else 2
    num_ne: int = int(header[num_ne_index])
    num_te: int = int(header[num_te_index])
    pecs: str = ' '.join(adf15_block[1:])
    pecs: ndarray = array([float(p) for p in pecs.split()])
    ne_slice: slice = slice(0, num_ne)
    te_slice: slice = slice(num_ne, num_ne + num_te)
    ne: ndarray = pecs[ne_slice]
    te: ndarray = pecs[te_slice]
    pecs: ndarray = pecs[te_slice.stop:].reshape((num_ne, num_te))
    # dims: tuple[str, str] = ('ne / cm-3', 'Te / eV')
    dims: tuple[str, str] = ('ne', 'Te')
    coords: dict[str, ndarray] = dict(
        ne=ne,
        Te=te
    )
    attrs: dict[str, str] = dict(
        description=adf15_block[0],
        units='cm3/s'
    )
    block_array: DataArray = DataArray(pecs, dims=dims, coords=coords, name='pecs', attrs=attrs)
    return block_array


def block_start_check(line: str) -> bool:
    checks: list[str] = [
        'EXCIT', 'RECOM', 'CHEXC', 'TYPE', 'type'
    ]
    checks: list[bool] = [ch in line for ch in checks]
    check: int = sum(checks)
    check: bool = True if check == 2 else False
    return check


def get_header_slice(adf15_raw: str) -> int:
    adf15: list[str] = adf15_raw.split('\n')
    i: int
    line: str
    for i, line in enumerate(adf15):
        if block_start_check(line):
            return i
    else:
        raise ValueError('No line found!')


def build_adf15_DataArray(adf15_raw: str) -> dict[str, list]:
    # Get the number of blocks
    total_blocks: int = get_number_of_blocks(adf15_raw)
    # Get the number of lines in each block
    adf15_raw_list: list[str] = adf15_raw.split(ADF15_SEPARATOR)
    pec_start: int = get_header_slice(adf15_raw)
    pec_data_long: list[str] = adf15_raw_list[0].split('\n')
    pec_data: list[str] = pec_data_long[pec_start:-1]
    len_block: float = len(pec_data) / total_blocks
    # Check an integer number of lines
    if not len_block // 1 == len_block:
        raise ValueError
    # Reformat data as a dict
    len_block: int = int(len_block)
    block: int
    data: dict[str, list] = {}
    for block in range(total_blocks):
        lhs: int = block * len_block
        rhs: int = lhs + len_block
        block_slice: slice = slice(lhs, rhs)
        block_str: list[str] = pec_data[block_slice]
        block_array: DataArray = block_str_to_array(block_str)
        data[f'{block+1}'] = block_array

    dims: tuple[str, str, str] = ('block', 'ne', 'Te')
    coords: dict[str, ndarray] = dict(
        block=array([int(key) for key in data.keys()]),
        ne=block_array.ne.data,
        Te=block_array.Te.data
    )
    attrs: dict[str, str] = dict(
        description=ADF15_SEPARATOR.join(adf15_raw_list[1:]),
        units='cm3/s',
        adf=adf15_raw
    )
    data_adf15: DataArray = DataArray(
        array([d.data for d in data.values()]),
        dims=dims,
        coords=coords,
        name='pecs',
        attrs=attrs
    )
    return data_adf15


def load_adf15(adf15: [str, Path]) -> DataArray:
    # Load as text file
    with open(adf15, 'r') as f:
        adf15_raw: str = f.read()
    # Separate by block nuber
    adf15_model: DataArray = build_adf15_DataArray(adf15_raw)
    return adf15_model


def read_adf15(adf15: [str, Path],
               block: [int, list],
               ne: ndarray,
               te: ndarray) -> DataArray:
    """
    Reads ADAS formatted ADF15 (PEC files) and returns

    :param adf15:
    :param block:
    :param te:
    :param ne:
    :return:
    """
    # Load DataArray
    adf15_model: DataArray = load_adf15(adf15)
    # Interpolate to get desired values
    block: list[int] = [block] if type(block) is int else block
    pecs_out: DataArray = adf15_model.interp(block=block, ne=ne, Te=te)
    return pecs_out


def main() -> None:
    """
    Test run of load_adf15

    # adf15_block.plot.line(x='Te', xscale='log', yscale='log')
    # adf15_blocks['1'].interp(ne=[1e14], Te=[10])

    :return None:
    """
    from time import time
    # Files to test read_adf15 on
    folders: list[str] = [
        'boron',
        'nitrogen',
        'nitrogen'
    ]
    test_files: list[str] = [
        'pecXXb_pjub1.dat',
        'pec96#n_vsu#n1.dat',
        'n_v_vsu.pass'
    ]
    # Test read_adf15 and run
    file: str
    folder: str
    blocks: list[int] = [1, 2]
    ne: list[float] = [3e12, 3e14]
    te: list[float] = [2.3, 11.]
    print('Test runs:')
    for folder, file in zip(folders, test_files):
        t0: float = time()
        test_file: Path = ADAS_PATH / 'adf15' / folder / file
        _ = read_adf15(test_file, block=blocks, ne=ne, te=te)
        dt: float = round(1e3 * (time() - t0), 1)
        print(f"{4*''} - {file} loaded in {dt} ms")


if __name__ == "__main__":
    main()
