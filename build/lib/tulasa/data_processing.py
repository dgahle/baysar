import numpy as np
from numpy import random

import scipy.io as sio
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import os, sys, io

from tulasa import general

def time_posterior(posterior, theta, number=500):

    from time import time

    tmp_time = time()

    for tmp in np.arange(number):

        posterior(theta)

    runtime = time() - tmp_time

    print(number, runtime, runtime / number)


def add_noise(data, scale):

    return abs(data + random.normal( min(data), min(data) * scale, len(data)) )


def add_cal_error(chain, error):

    num_pixels = 0

    for p in chain.posterior.posterior_components:

        num_pixels += len(p.y_data)

    for counter, tmp_p in enumerate(chain.probs):

        tmp_cal_error = -0.5 * ( ( 1 - random.normal(1, error ) ) /
                                                error ) ** 2

        chain.probs[counter] = tmp_p + tmp_cal_error * num_pixels


def wave_cal(y_data):

    d_eps = np.where( y_data[:30] == max(y_data[:30]) )[0][0]
    n_ii_399 = np.where( y_data[150:165] == max(y_data[150:165]) )[0][0]
    n_ii_404 = np.where( y_data[395:410] == max(y_data[395:410]) )[0][0]

    wave_refs = [3968.99, 3995.0, 4041.32]
    tmp_x = [d_eps, 150 + n_ii_399, 395 + n_ii_404]

    cal_inter = interp1d(tmp_x, wave_refs, kind='linear', fill_value='extrapolate')

    return cal_inter( np.arange( len(y_data) ) )


def fake_calibration_error(posterior):

    for spectrometer in posterior.posterior_components:
        spectrometer.get_error(fake_cal=True)


def crop_if(tmp_if, snr):
    # snr = 100
    #
    # tmp_if = instrument_function[0]
    tmp_if[np.where(tmp_if < max(tmp_if) / snr)] = 0

    return tmp_if / sum(tmp_if)


def add_chain_bounds(chain, reduced=False, add_widths=False):

    if reduced:

        for counter, tmp_bound in enumerate(chain.posterior.reduced_theta_bounds):

            chain.set_boundaries(counter, tmp_bound)

            if add_widths:

                chain.params[counter].sigma = chain.posterior.reduced_theta_widths[counter]

    else:

        for counter, tmp_bound in enumerate(chain.posterior.plasma.theta_bounds):

            chain.set_boundaries(counter, tmp_bound)

            if add_widths:

                chain.params[counter].sigma = chain.posterior.theta_widths[counter]


def reduce_sample(sample, new_size):

    small_sample = []
    indicies = np.linspace(0, len(sample)-1, new_size, dtype='int')

    for index in indicies:
        small_sample.append(sample[index])

    return small_sample


def hdi_estimator(peak, pc=0.75, x=None, plot=False, return_peak=True):

    'takes 22 s for 100 runs for len(peak)=1e4 - 4.5 Hz'
    'takes 0.28 s for 100 runs for len(peak)=1e3 - 357 Hz'
    'takes 0.034 s for 100 runs for len(peak)=1e2 - 2941 Hz'

    num = len(peak)
    scan_peak = np.zeros((num, num))

    sort_peak_index = np.argsort(peak)[::-1]

    if plot:
        general.plot(peak)
        general.plot(np.sort(peak))

    tmp_peak = np.zeros(num)

    for counter, index in enumerate(sort_peak_index):
        tmp_peak[index] += peak[index]

        scan_peak[counter, :] = tmp_peak

    water_shead_int = scan_peak.sum(1)
    water_shead_int /= max(water_shead_int)

    if plot:
        general.plot(water_shead_int)

    water_shead_int_pc_intercept = abs(water_shead_int - pc)

    # print('peak.shape', peak.shape)
    # print('min(water_shead_int_pc_intercept)', min(water_shead_int_pc_intercept), water_shead_int.shape)
    # print(water_shead_int_pc_intercept)
    # print(water_shead_int)

    hdi_peak_index = np.where(water_shead_int_pc_intercept == min(water_shead_int_pc_intercept))

    # print(scan_peak.shape, hdi_peak_index[0].shape)

    hdi_peak = scan_peak[hdi_peak_index, :]

    hdi_peak = hdi_peak.flatten()

    # print( hdi_peak.shape )

    # general.close_plots()

    if plot:
        general.plot([peak, hdi_peak], multi='fake')

    hdi_peak_diff_abs = peak - hdi_peak
    hdi_peak_diff_abs_grad = np.gradient(hdi_peak_diff_abs)

    hdi_indicies = np.where( ( (hdi_peak != 0) + (np.gradient(hdi_peak) == 0) ) == False )

    if plot:
        general.plot(hdi_peak_diff_abs)

    # print(hdi_indicies)

    if return_peak:
        return hdi_indicies, hdi_peak
    else:
        return hdi_indicies

    pass

def calc_dl(peak, pc, x):

    indicies = hdi_estimator(peak, pc=pc, x=None, plot=False, return_peak=False)[0]

    dl = []

    for counter in 2 * np.arange(len(indicies) / 2, dtype=int):

        try:
            upper = indicies[counter + 1]
        except:
            return np.nan

        lower = indicies[counter]

        dl.append(x[upper] - x[lower])

    return sum(dl)