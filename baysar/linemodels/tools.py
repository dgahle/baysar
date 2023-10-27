# Imports
from numpy import array, sqrt, where
from scipy.constants import speed_of_light

from ..lineshapes import Gaussian

# Variables
elements = ["H", "D", "He", "Be", "B", "C", "N", "O", "Ne"]
masses = [1, 2, 4, 9, 10.8, 12, 14, 16, 20.2]
elements_and_masses = []
for elm, mass in zip(elements, masses):
    elements_and_masses.append((elm, mass))
atomic_masses = dict(elements_and_masses)


# Functions and classes
def species_to_element(species):
    return species[0 : where([a == "_" for a in species])[0][0]]


def get_atomic_mass(species):
    elm = species_to_element(species)
    return atomic_masses[elm]


def doppler_shift(cwl, atomic_mass, velocity):
    f_velocity = velocity / speed_of_light
    return cwl * (f_velocity + 1)


def update_DopplerLine_cwls(line, cwls):
    if not len(cwls) == len(line.line.cwl):
        error_string = "Transition has {} line and {} where given".format(
            len(line.line.cwl), len(cwls)
        )
        raise ValueError(error_string)

    line.line.cwl = array(cwls)


class DopplerLine(object):

    """
    This class is used to calculate Guassian lineshapes as a function of area and ion
    temperature. It is inherited by NoADASLine, Xline and ADAS406Line and used by
    NoADASLines and ADAS406Lines.
    """

    def __init__(self, cwl, wavelengths, atomic_mass, fractions=None, half_range=5):
        self.atomic_mass = atomic_mass
        self.line = Gaussian(
            x=wavelengths, cwl=cwl, fwhm=None, fractions=fractions, normalise=True
        )

    def __call__(self, ti, ems):
        fwhm = self.line.cwl * 7.715e-5 * sqrt(ti / self.atomic_mass)
        # print(self.line.cwl, self.atomic_mass, fwhm, ti)
        return self.line([fwhm, ems])


def main() -> None:
    pass


if __name__ == "__main__":
    main()
