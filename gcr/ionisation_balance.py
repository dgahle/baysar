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
            rate_matrix[:, :, charge, charge - 1] = scd_data
            rate_matrix[:, :, charge - 1, charge] = acd_data
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = - acd_data
        else:
            # Sources (off diagonal)
            rate_matrix[:, :, charge, charge - 1] = scd_data
            rate_matrix[:, :, charge - 1, charge] = acd_data
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = - source_data

    #
    meta: ndarray = zeros(proton_number)
    meta[0] = 1
    # Solve for eigenvalues
    from numpy import einsum
    from numpy.linalg import eig, inv
    eigenvalues, eigenvectors = eig(rate_matrix)
    fractional_abundance: ndarray = einsum(
        'kl,klj->klj',
        inv(eigenvectors).dot(meta)[:, :, 0],
        eigenvectors[:, :, :, 0]
    )
    # # Normalise
    # _eigenvalues_normalised: ndarray = eigenvalues - eigenvalues.min()
    # eigenvalues_normalised: ndarray = _eigenvalues_normalised / _eigenvalues_normalised.sum(-1)[:, :, None]
    # Format
    fractional_abundance: DataArray = DataArray(
        fractional_abundance,
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
