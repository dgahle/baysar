# Imports
from itertools import product

from numpy import arange, array, logspace, ndarray, zeros
from scipy.linalg import null_space
from xarray import concat, DataArray

from backend.time import TimeIt
from OpenADAS import get_adf11, load_adf11

# Variables


# Functions and classes
def build_rates_matrix(element: str, tau: float = None) -> DataArray:
    # Get SCD and ACD for the element
    adf11_scd: str = get_adf11(element, adf11type="scd")
    adf11_acd: str = get_adf11(element, adf11type="acd")
    scd: DataArray = load_adf11(adf11=adf11_scd, passed=True)
    acd: DataArray = load_adf11(adf11=adf11_acd, passed=True)
    # Account for transport if tau is passed
    # scd = scd if tau is None else scd - (1 / (scd.ne * tau))
    scd = scd if tau is None else tau * scd
    # if tau is not None:
    #     from numpy import ones
    #     tau_array: ndarray = arange(*scd.block.shape, dtype=float).clip(min=1.)
    #     tau_array *= tau
    #     tau_array = tau_array.clip(max=1.)
    #     scd *= tau_array[:, None, None]
    # Build the rate matrix
    proton_number: int = 1 + scd.shape[0]
    # shape: tuple[int] = (proton_number, proton_number, *scd.shape[1:])
    shape: tuple[int] = (*scd.shape[1:], proton_number, proton_number)
    rate_matrix: ndarray = zeros(shape)
    charge: int
    for charge in range(proton_number):
        # Get SCD rates
        scd_block: int = charge if 1 + charge == proton_number else 1 + charge
        scd_data: ndarray = scd.sel(block=scd_block).data
        # Get ACD rates
        if charge == 0:
            acd_data: float = 0.0
        else:
            acd_block: int = charge
            acd_data: ndarray = acd.sel(block=acd_block).data
        # rate_matrix -> [ne, Te, row, col]
        # Neutral
        if charge == 0:
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = -scd.sel(block=1 + charge).data
            # Sources (off diagonal)
            rate_matrix[:, :, charge, charge + 1] = acd.sel(block=1 + charge).data
        # Ions
        elif (0 < charge) and (charge < (proton_number - 1)):
            # Sources (off diagonal)
            rate_matrix[:, :, charge, charge - 1] = scd.sel(block=charge).data
            rate_matrix[:, :, charge, charge + 1] = acd.sel(block=1 + charge).data
            # Losses (diagonal)
            loss_rate: ndarray = (
                scd.sel(block=1 + charge).data + acd.sel(block=charge).data
            )
            rate_matrix[:, :, charge, charge] = -(loss_rate)
        # Bare nuclei (Final entry check)
        elif charge == (proton_number - 1):
            # Sources (off diagonal)
            rate_matrix[:, :, charge, charge - 1] = scd.sel(block=charge).data
            # Losses (diagonal)
            rate_matrix[:, :, charge, charge] = -acd.sel(block=charge).data
        else:
            raise ValueError()

    # Format to DataArray
    charge_array: ndarray = arange(1 + scd.coords["block"].max())
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


@TimeIt
def ionisation_balance(element: str, tau: float = None) -> DataArray:
    """
    Solves the steady state solution to the ionisation rate balance

    :param (str) element:
        Element of choice.
    :param (float) tau: - Default = None
        Proxy transport parameter.
    :return (DataArray) fractional_abundance:
        Fractional abundance of the passed element evaluated over the electron density and temperature grid defined by
        the adf11 parameter space.

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
    rate_matrix: DataArray = build_rates_matrix(element, tau=tau)
    # Solve for fractional abundance
    i: int
    ne0: float
    te0: float
    proton_number: int = len(rate_matrix.charge0)
    thetas: list[tuple[float, float]] = [
        theta for theta in product(rate_matrix.ne, rate_matrix.Te)
    ]
    fractional_abundance: ndarray = zeros((len(thetas), proton_number))
    for i, theta in enumerate(thetas):
        # Calculate the null space vector
        ne0, te0 = theta
        r_matrix: DataArray = rate_matrix.sel(ne=ne0, Te=te0)
        f_ion: ndarray = null_space(r_matrix)
        # Normalise into physical space
        f_ion /= f_ion.sum()
        # Cache
        fractional_abundance[i] = f_ion.flatten()
        pass
    # Format
    fractional_abundance: ndarray = array(fractional_abundance)
    fractional_abundance = fractional_abundance.reshape(
        rate_matrix.ne.shape[0], rate_matrix.Te.shape[0], proton_number
    )
    fractional_abundance: DataArray = DataArray(
        fractional_abundance,
        coords=dict(
            ne=rate_matrix.ne, Te=rate_matrix.Te, charge=list(arange(proton_number))
        ),
    )

    return fractional_abundance


def ionisation_balance_transport(element: str) -> DataArray:
    """
    Solves the steady state solution to the ionisation rate balance

    :param (str) element:
        Element of choice.
    :param (float) tau: - Default = None
        Proxy transport parameter.
    :return (DataArray) fractional_abundance:
        Fractional abundance of the passed element evaluated over the electron density and temperature grid defined by
        the adf11 parameter space.
    """
    tau: float
    taus: ndarray = logspace(4, -6, 11)
    taus: DataArray = DataArray(taus, name='Tau', coords=dict(tau=taus))
    f_ion: list[DataArray] = [
        ionisation_balance(element, tau) for tau in taus
    ]
    fractional_abundance: DataArray = concat(f_ion, dim=taus)

    return fractional_abundance


def main() -> None:
    pass


if __name__ == "__main__":
    main()
    pass
