# Imports
from itertools import product
from numpy import array, isclose
from numpy import arange, ndarray, zeros
from OpenADAS import load_adf11, get_adf11
from scipy.linalg import null_space
from xarray import DataArray

# Variables


# Functions and classes
def build_rates_matrix(element: str) -> DataArray:
    # Get SCD and ACD for the element
    adf11_scd: str = get_adf11(element, adf11type='scd')
    adf11_acd: str = get_adf11(element, adf11type='acd')
    scd: DataArray = load_adf11(adf11=adf11_scd, passed=True)
    acd: DataArray = load_adf11(adf11=adf11_acd, passed=True)
    # Build the rate matrix
    proton_number: int = 1 + scd.shape[0]
    # shape: tuple[int] = (proton_number, proton_number, *scd.shape[1:])
    shape: tuple[int] = (*scd.shape[1:], proton_number, proton_number)
    rate_matrix: ndarray = zeros(shape)
    charge: int
    for charge in range(proton_number):
        scd_block: int = charge if 1 + charge == proton_number else 1 + charge
        acd_data: ndarray = 0 if charge == 0 else acd.sel(block=charge).data
        scd_data: ndarray = scd.sel(block=scd_block).data
        source_data: ndarray = scd_data + acd_data
        # Neutral
        if charge == 0:
            rate_matrix[:, :, charge, charge] = - scd_data
        # # Final entry check
        # elif charge == (proton_number - 1):
        #     # Sources (off diagonal)
        #     rate_matrix[:, :, charge, charge - 1] = source_data
        #     rate_matrix[:, :, charge - 1, charge] = source_data
        #     # Losses (diagonal)
        #     rate_matrix[:, :, charge, charge] = - acd_data
        else:
            # Sources (off diagonal)
            rate_matrix[:, :, charge, charge - 1] = scd_data  # source_data
            rate_matrix[:, :, charge - 1, charge] = acd_data  # source_data
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = - acd_data  # source_data

    # Format to DataArray
    charge_array: ndarray = arange(1 + scd.coords['block'].max())
    rate_matrix: DataArray = DataArray(
        rate_matrix,
        name=f"{element} Ionisation Balance Rate Matrix",
        coords=dict(
            ne=scd.ne,
            Te=scd.Te,
            charge0=charge_array,
            charge1=charge_array,
        ),
        attrs=dict(
            description="Ionisation balance rate matrix",
            units="cm^3/s",
        )
    )
    return rate_matrix


from backend.time import TimeIt
@TimeIt
def ionisation_balance(element: str) -> DataArray:
    """
    (d/dt)f_m = R_mn * f_n

    Solve for f_m where R_mn * f_n = 0

    Notes:
        - f_n = [f_0, ..., f_A] where A is the proton number of the element

    For example:

        Solving the hydrogen ionisation balance at ne = 1e14 / cm3 and Te = 1 eV

        R_mn(ne, Te) = [
            [5.64441065e+11, 5.91890828e+11],  # [SCD, -(SCD + ACD)]
            [5.91890828e+11, 2.74497632e+10]   # [-(SCD + ACD), ACD]
        ]

        f_0 = SCD

        What is f_n where R_mn * f_n = 0?

    """
    # Get the rate matrix
    rate_matrix: DataArray = build_rates_matrix(element)
    # # Test
    # r_matrix: DataArray = rate_matrix.interp(ne=1e14, Te=1).data
    # print(r_matrix)
    # raise ValueError
    # Solve for fractional abundance
    i: int
    ne0: float
    te0: float
    fractional_abundance: list[ndarray] = []
    for i, theta in enumerate(product(rate_matrix.ne, rate_matrix.Te)):
        # Calculate the null space vector
        ne0, te0 = theta
        r_matrix: DataArray = rate_matrix.sel(ne=ne0, Te=te0)
        f_ion: ndarray = null_space(r_matrix)
        # Normalise into physical space
        f_ion /= f_ion.sum()
        # Numerics test
        assert isclose(
            r_matrix.data.dot(null_space(r_matrix)),
            0
        ).all()
        # Cache
        fractional_abundance.append(f_ion.flatten())
        pass
    # Format
    fractional_abundance: ndarray = array(fractional_abundance)
    fractional_abundance = fractional_abundance.reshape(rate_matrix.ne.shape[0], rate_matrix.Te.shape[0], 2)
    fractional_abundance: DataArray = DataArray(
        fractional_abundance,
        coords=dict(
            ne=rate_matrix.ne,
            Te=rate_matrix.Te,
            charge=[0, 1]
        )
    )

    return fractional_abundance


def main() -> None:
    pass


if __name__ == "__main__":
    main()
    pass
