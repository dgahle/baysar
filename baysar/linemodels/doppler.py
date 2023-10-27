# Imports
from copy import copy

from numpy import array, sqrt
from scipy.constants import speed_of_light

from ..lineshapes import Gaussian

# Variables


# Functions and classes
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


def update_HydrogenLineShape_cwl(line, cwl):
    old_cwl = copy(line.cwl)
    # update cwl for stark and zeeman
    line.cwl = cwl
    # update for DopplerLine
    update_DopplerLine_cwls(line.doppler_function, [cwl])
    # shift the DopplerLine wavelengths so the peak remains centred for the convolution
    line.doppler_function.line.reducedx[0] += cwl - old_cwl


def doppler_shift_BalmerHydrogenLine(self, velocity):
    # get doppler shifted wavelengths
    new_cwl = doppler_shift(copy(self.cwl), self.atomic_mass, velocity)
    # update wavelengths
    update_HydrogenLineShape_cwl(self.lineshape, new_cwl)


if __name__ == "__main__":
    pass
