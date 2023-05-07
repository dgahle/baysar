# Imports
from OpenADAS import get_adf11


# Variables


# Functions and classes
class TestGetAdf11:

    def test_H_ACD_yr12(self) -> None:
        adf11: str = get_adf11(element='h', adf11type='acd', year=12)
        assert type(adf11) is str

    def test_Li_SCD_96_r(self) -> None:
        adf11: str = get_adf11(element='li', adf11type='scd', year=96, resolved=True)
        assert type(adf11) is str

    def test_C_CCD_r(self) -> None:
        adf11: str = get_adf11(element='c', adf11type='ccd', resolved=True)
        assert type(adf11) is str

    def test_N_QCD(self) -> None:
        adf11: str = get_adf11(element='n', adf11type='qcd')
        assert type(adf11) is str

    def test_Ne_XCD(self) -> None:
        adf11: str = get_adf11(element='ne', adf11type='xcd')
        assert type(adf11) is str


class TestGetAdf11Power:

    def test_H_PLT_yr12(self) -> None:
        adf11: str = get_adf11(element='h', adf11type='plt', year=12)
        assert type(adf11) is str

    def test_H_PRB_yr12(self) -> None:
        adf11: str = get_adf11(element='h', adf11type='prb', year=12)
        assert type(adf11) is str

    def test_C_PLT(self) -> None:
        adf11: str = get_adf11(element='c', adf11type='plt')
        assert type(adf11) is str

    def test_N_PRB(self) -> None:
        adf11: str = get_adf11(element='c', adf11type='prb')
        assert type(adf11) is str


def main() -> None:
    pass


if __name__ == "__main__":
    main()
