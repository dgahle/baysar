from numpy import square, sqrt, mean, linspace, nan, log, diff, arange, zeros, concatenate, where

from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import sparse
import numpy as np
import time as clock

import os, sys, io, warnings

def clip_data(x_data, y_data, x_range):

    '''
    returns sections of x and y data that is within the desired x range

    :param x_data:
    :param y_data:
    :param x_range:
    :return:
    '''

    assert len(x_data)==len(y_data), 'len(x_data)!=len(y_data)'

    x_index = where((min(x_range) < x_data) & (x_data < max(x_range)))

    if not np.isreal(x_data[x_index]).all():
        print('not isreal(x_data[x_index])')
        print(x_data)
    if not np.isreal(y_data[x_index]).all():
        print('not isreal(Y_data[x_index])')
        print(y_data)

    return x_data[x_index], y_data[x_index]

def calibrate_wavelength(spectra, waves, points, references, width=0.5):
    pixels=[]
    for p in points:
        tmpx, tmpy = clip_data(waves, spectra, [p-width, p+width])
        com=center_of_mass(tmpy)[0]
        shift=where(waves==tmpx.min())[0][0]
        pixels.append(com+shift)

    newaxis=interp1d(pixels, references, bounds_error=False, fill_value="extrapolate")

    return newaxis(np.arange(len(spectra)))

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

from baysar.input_functions import within
from scipy.ndimage.measurements import center_of_mass

def centre_peak0(peak, places=7):
    com=center_of_mass(peak)
    x=np.arange(len(peak))
    centre=np.mean(x)
    shift=com-centre
    interp=interp1d(x, peak, bounds_error=False, fill_value=0.)
    return interp(x+shift)

def centre_peak(peak, places=12):
    x=np.arange(len(peak))
    centre=np.mean(x)
    while not np.round(abs(centre-center_of_mass(peak))[0], places)==0:
        peak=centre_peak0(peak)
    return peak
