# Imports
from numpy import diag, ndarray, zeros
from OpenADAS import load_adf11, get_adf11
from xarray import DataArray

# Variables


# Functions and classes
def ionisation_balance(element: str) -> DataArray:
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
    for charge in range(proton_number - 1):
        scd_block: int = 1 + charge
        # Neutral
        if charge == 0:
            rate_matrix[:, :, charge, charge] = - scd.sel(block=scd_block).data
        # Final entry check
        elif charge == proton_number:
            rate_matrix[:, :, charge, charge] = - acd.sel(block=charge).data
        else:
            acd_data: ndarray = acd.sel(block=charge).data
            scd_data: ndarray = scd.sel(block=scd_block).data
            # Sources (off diagonal)
            source_data: ndarray = scd_data + acd_data
            rate_matrix[:, :, charge, charge - 1] = source_data
            rate_matrix[:, :, charge - 1, charge] = source_data
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = - source_data

    # Solve for eigenvalues
    meta: ndarray = zeros(shape[::-1])
    meta[:, :, 0] = 1.
    from numpy.linalg import eigvals, solve
    fractional_abundance: ndarray = eigvals(rate_matrix)
    fractional_abundance: DataArray = DataArray(
        fractional_abundance,
        coords=dict(
            ne=scd.ne,
            Te=scd.Te,
            charge=[0, 1]
        )
    )
    # TODO: numpy.linalg.LinAlgError: Singular matrix
    # Solve matrix using the SVD method
    # Build output DataArray
    raise NotImplementedError


def main() -> None:
    element: str = 'h'
    ionisation_balance(element)
    pass


if __name__=="__main__":
    main()
    pass
