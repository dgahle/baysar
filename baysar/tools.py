import io
import os
import sys
import time as clock
import warnings

import numpy as np
from numpy import (
    arange,
    concatenate,
    diff,
    linspace,
    log,
    mean,
    nan,
    sqrt,
    square,
    where,
    zeros,
)
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


def clip_data(x_data, y_data, x_range):
    """
    returns sections of x and y data that is within the desired x range

    :param x_data:
    :param y_data:
    :param x_range:
    :return:
    """

    assert len(x_data) == len(y_data), "len(x_data)!=len(y_data)"

    x_index = where((min(x_range) < x_data) & (x_data < max(x_range)))

    if not np.isreal(x_data[x_index]).all():
        print("not isreal(x_data[x_index])")
        print(x_data)
    if not np.isreal(y_data[x_index]).all():
        print("not isreal(Y_data[x_index])")
        print(y_data)

    return x_data[x_index], y_data[x_index]


def calibrate_wavelength(spectra, waves, points, references, width=0.5):
    pixels = []
    for p in points:
        tmpx, tmpy = clip_data(waves, spectra, [p - width, p + width])
        com = center_of_mass(tmpy)[0]
        shift = where(waves == tmpx.min())[0][0]
        pixels.append(com + shift)

    newaxis = interp1d(pixels, references, bounds_error=False, fill_value="extrapolate")

    return newaxis(np.arange(len(spectra)))


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


from scipy.ndimage import center_of_mass

from baysar.input_functions import within


def centre_peak0(peak, places=7, centre=None):
    com = center_of_mass(peak)
    x = np.arange(len(peak))
    if centre is None:
        centre = np.mean(x)
    shift = com - centre
    interp = interp1d(x, peak, bounds_error=False, fill_value=0.0)
    return interp(x + shift)


def centre_peak(peak, places=12, centre=None):
    x = np.arange(len(peak))
    if centre is None:
        centre = np.mean(x)
    while not np.round(abs(centre - center_of_mass(peak))[0], places) == 0:
        peak = centre_peak0(peak, centre=centre)
    return peak


def within(stuff, boxes):
    """
    This function checks is a value or list/array of values (the sample) are 'within' an array or values.

    :param stuff: a list of strings, ints or floats
    :param box: Array of values to check the sample(s) against
    :return: True or False if all the samples are in the array
    """

    if type(stuff) in (tuple, list):
        return any(
            [
                any([(box.min() < float(s)) and (float(s) < box.max()) for s in stuff])
                for box in boxes
            ]
        )
    else:
        if type(stuff) is str:
            stuff = float(stuff)
        elif type(stuff) in (int, float, np.float):
            pass
        else:
            raise TypeError("Input stuff is not of an appropriate type.")

        return any((box.min() < stuff) and (stuff < box.max()) for box in boxes)


from copy import copy

from numpy import argmax, argsort, array, diff, inf, isclose, linspace
from numpy.linalg import norm as normalise
from scipy.optimize import approx_fprime


def check_input():
    """
    write a fucntion for generalised type/value checking
    """
    pass


def type_check():
    pass
