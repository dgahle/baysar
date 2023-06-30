#
#
#
#
#
# Imports
from urllib import request

from .tools import _adf_exists_check

# Variables


# Functions
def get_adf11(
    element: str, adf11type: str, year: int = 96, resolved: bool = False
) -> str:
    """
    Download OpenADAS ADF11 library.

    Examples:
        - Hydrogen recombination rates `adf11: str = get_adf11(element='h', adf11type='acd', year=12)`.
        - Lithium ionisation rates resolved by metastables `adf11: str = get_adf11(element='li', adf11type='scd', year=93, resolved=True)`
        - Carbon charge exchange rates resolved by metastables `adf11: str = get_adf11(element='c', adf11type='ccd', resolved=True)`
        - Nitrogen ionisation rates `adf11: str = get_adf11(element='n', adf11type='scd')`

    :param (str) element:
        Element abbreviation for the requested data (lower case). For example hydrogen is 'h' and lithium is 'li'.
    :param (str) adf11type:
        Three letter code for the types of data (lower case):
            'acd' - Recombination
            'scd' - Ionisation
            'xcd' - Spin change within a charge state
            'qcd' - Spin change within a charge state via the z+1 charge state
            'ccd' - Charge-exchange
            'plt' - Total excitation line radiative power
            'prb' - Total recombination radiative power (line, recombination continuum, and bremsstrahlung)
            'prc' - Line power from thermal neutral hydrogen-ion charge exchange
            'pls' - Line radiative power of selected transitions of key ions
    :param (int) year: 96
        Reference year for the data. Most ADF11 files will have year 93 and 96 versions. Hydrogen ACD and SCD files
        have year 12 data. Broadly the newer data is "better" as it is created to overcome limitations in the older
        data.
    :param (bool) resolved: False
        Metastable resolution of the GCR model used to calculate the data.
    :return (str) adf11:
        The adf11 file downloaded as a string
    """
    # Format inputs
    spin_change_reactions: list = ["xcd", "qcd"]
    resolved: bool = True if adf11type in spin_change_reactions else resolved
    resolution: str = "r" if resolved else ""
    adf11type = f"{adf11type}{year}{resolution}"
    # Construct the download URL
    download_url: str = (
        f"https://open.adas.ac.uk/download/adf11/{adf11type}/{adf11type}_{element}.dat"
    )
    # Download
    with request.urlopen(download_url) as f:
        adf11: str = f.read().decode("utf-8")
    # Check download
    _adf_exists_check(adf11, download_url)

    return adf11


def main() -> None:
    pass


if __name__ == "__main__":
    main()
