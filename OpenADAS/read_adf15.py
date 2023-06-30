#
#
#
#
#
# Imports
from pathlib import Path

from numpy import array, ndarray, round
from xarray import DataArray

# Variables
ADAS_PATH: Path = Path(__file__).parent.parent / "adas"
ADF15_SEPARATOR: str = (
    "C-----------------------------------------------------------------------"
)


# Functions
def get_number_of_blocks(adf15_raw: str) -> int:
    header: str = adf15_raw.split("\n")[0]
    total_blocks: int = int(header.split()[0])
    return total_blocks


def block_str_to_array(adf15_block: list) -> DataArray:
    header: list = adf15_block[0].split()
    num_ne_index: int = 2 if header[1] == "A" else 1
    num_te_index: int = 3 if header[1] == "A" else 2
    num_ne: int = int(header[num_ne_index])
    num_te: int = int(header[num_te_index])
    pecs: str = " ".join(adf15_block[1:])
    pecs: ndarray = array([float(p) for p in pecs.split()])
    ne_slice: slice = slice(0, num_ne)
    te_slice: slice = slice(num_ne, num_ne + num_te)
    ne: ndarray = pecs[ne_slice]
    te: ndarray = pecs[te_slice]
    pecs: ndarray = pecs[te_slice.stop :].reshape((num_ne, num_te))
    # dims: tuple = ('ne / cm-3', 'Te / eV')
    dims: tuple = ("ne", "Te")
    coords: dict = dict(ne=ne, Te=te)
    attrs: dict = dict(description=adf15_block[0], units="cm3/s")
    block_array: DataArray = DataArray(
        pecs, dims=dims, coords=coords, name="pecs", attrs=attrs
    )
    return block_array


def block_start_check(line: str) -> bool:
    checks: list = ["EXCIT", "RECOM", "CHEXC", "TYPE", "type"]
    checks: list = [ch in line for ch in checks]
    check: int = sum(checks)
    check: bool = True if check == 2 else False
    return check


def get_header_slice(adf15_raw: str) -> int:
    adf15: list = adf15_raw.split("\n")
    i: int
    line: str
    for i, line in enumerate(adf15):
        if block_start_check(line):
            return i
    else:
        raise ValueError("No line found!")


def build_adf15_dataarray(adf15_raw: str) -> DataArray:
    # Get the number of blocks
    total_blocks: int = get_number_of_blocks(adf15_raw)
    # Get the number of lines in each block
    adf15_raw_list: list = adf15_raw.split(ADF15_SEPARATOR)
    pec_start: int = get_header_slice(adf15_raw)
    pec_data_long: list = adf15_raw_list[0].split("\n")
    pec_data: list = pec_data_long[pec_start:-1]
    len_block: float = len(pec_data) / total_blocks
    # Check an integer number of lines
    if not len_block // 1 == len_block:
        raise ValueError
    # Reformat data as a dict
    len_block: int = int(len_block)
    block: int
    data: dict = {}
    for block in range(total_blocks):
        lhs: int = block * len_block
        rhs: int = lhs + len_block
        block_slice: slice = slice(lhs, rhs)
        block_str: list = pec_data[block_slice]
        block_array: DataArray = block_str_to_array(block_str)
        data[f"{block+1}"] = block_array

    dims: tuple = ("block", "ne", "Te")
    coords: dict = dict(
        block=array([int(key) for key in data.keys()]),
        ne=block_array.ne.data,
        Te=block_array.Te.data,
    )
    attrs: dict = dict(
        description=ADF15_SEPARATOR.join(adf15_raw_list[1:]),
        units="cm3/s",
        adf=adf15_raw,
    )
    data_adf15: DataArray = DataArray(
        array([d.data for d in data.values()]),
        dims=dims,
        coords=coords,
        name="pecs",
        attrs=attrs,
    )
    return data_adf15


def load_adf15(adf15: [str, Path], passed: bool = False) -> DataArray:
    # Load as text file
    if not passed:
        with open(adf15, "r") as f:
            adf15_raw: str = f.read()
    else:
        adf15_raw: str = adf15
    # Separate by block nuber
    adf15_model: DataArray = build_adf15_dataarray(adf15_raw)
    return adf15_model


def read_adf15(
    adf15: [str, Path],
    block: [int, list],
    ne: ndarray,
    te: ndarray,
    passed: bool = False,
) -> DataArray:
    """
    Reads ADAS formatted ADF15 (PEC files) and returns

    :param adf15:
    :param block:
    :param te:
    :param ne:
    :return:
    """
    # Load DataArray
    adf15_model: DataArray = load_adf15(adf15, passed=passed)
    # Interpolate to get desired values
    block: list = [block] if type(block) is int else block
    kwargs: dict = dict(fill_value="extrapolate")
    pecs_out: DataArray = adf15_model.interp(block=block, ne=ne, Te=te, kwargs=kwargs)
    return pecs_out


def main() -> None:
    pass


if __name__ == "__main__":
    main()
