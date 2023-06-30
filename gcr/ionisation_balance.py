# Imports
from numpy import arange, diag, ndarray, zeros
from OpenADAS import load_adf11, get_adf11
from xarray import DataArray

# Variables


# Functions and classes
def build_rates_matrix(element: str) -> ndarray:
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
            rate_matrix[:, :, charge, charge] = - scd.sel(block=scd_block).data
        # Final entry check
        elif charge == (proton_number - 1):
            # Sources (off diagonal)
            rate_matrix[:, :, charge, charge - 1] = source_data
            rate_matrix[:, :, charge - 1, charge] = source_data
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = - acd_data
        else:
            # Sources (off diagonal)
            rate_matrix[:, :, charge, charge - 1] = source_data
            rate_matrix[:, :, charge - 1, charge] = source_data
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = - source_data

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


def ionisation_balance(element: str) -> DataArray:
    rate_matrix: ndarray = build_rates_matrix(element)
    # Solve for eigenvalues
    from numpy.linalg import eig
    eigenvalues, eigenvectors = eig(rate_matrix)
    # Normalise
    _eigenvalues_normalised: ndarray = eigenvalues - eigenvalues.min()
    eigenvalues_normalised: ndarray = _eigenvalues_normalised / _eigenvalues_normalised.sum(-1)[:, :, None]
    # Format
    fractional_abundance: DataArray = DataArray(
        eigenvalues_normalised,
        coords=dict(
            ne=scd.ne,
            Te=scd.Te,
            charge=[0, 1]
        )
    )
    return fractional_abundance


def main() -> None:
    element: str = 'h'
    ionisation_balance(element)
    pass


if __name__=="__main__":
    main()
    pass
