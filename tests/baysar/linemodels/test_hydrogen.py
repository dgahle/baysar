# Imports
from numpy import arange, ndarray, trapz

from baysar.linemodels.hydrogen import StarkShape

# Variables
cwl: float = 4101.
dx: float = 0.2
wavelengths: ndarray = cwl + arange(-5, 5, dx)
n_upper: int = 7
n_lower: int = 2
ne: float = 2e14
te: float = 5.


# Functions and classes
class TestStarkShape:

    def test_call(self) -> None:
        stark: StarkShape = StarkShape(cwl, wavelengths, n_upper, n_lower)
        lineshape: ndarray = stark(ne, te)
        area: float = trapz(lineshape, stark.wavelengths)

        assert abs(area - 1.) < 1e-6, f"Stark shape area is {area:.3f}!"


if __name__ == "__main__":
    pass
