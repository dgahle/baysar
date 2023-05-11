# Imports
from numpy import abs, arange, einsum, ix_, ndarray, zeros
from numpy.linalg import eig, inv
from OpenADAS import load_adf11, get_adf11
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

    # Format
    rate_matrix: DataArray = DataArray(
        rate_matrix,
        coords=dict(
            ne=scd.ne,
            Te=scd.Te,
            charge0=arange(proton_number),
            charge1=arange(proton_number)
        )
    )

    return rate_matrix


def ionisation_balance(element: str) -> DataArray:
    # Get rates
    rate_matrix: DataArray = build_rates_matrix(element)
    proton_number: int = len(rate_matrix.charge0)
    # Initial conditions (time independent so not important)
    meta: ndarray = zeros(proton_number)
    meta[0] = 1
    # Solve for eigenvalues
    eigenvalues, eigenvectors = eig(rate_matrix)
    # Sort
    eigenvalue_axis: int = 2
    index: list = list(ix_(*[arange(i) for i in eigenvalues.shape]))
    index[eigenvalue_axis] = abs(eigenvalues).argsort(eigenvalue_axis)
    # Calculate fractional abundance
    # LHS
    lhs_matrix: ndarray = inv(eigenvectors).dot(meta)
    lhs_matrix = lhs_matrix[index][:, :, 0]
    # RHS
    rhs_matrix: ndarray = eigenvectors.transpose((0, 1, 3, 2))
    rhs_matrix = rhs_matrix[index][:, :, 0, :]
    # Fractional abundance
    fractional_abundance: ndarray = einsum(
        'kl,klj->klj',
        lhs_matrix,
        rhs_matrix,
    )
    # Format
    fractional_abundance: DataArray = DataArray(
        fractional_abundance,
        coords=dict(
            ne=rate_matrix.ne,
            Te=rate_matrix.Te,
            charge=rate_matrix.charge0,
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
