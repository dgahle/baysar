# O. Imports and Settings
# 0.0. Before Anything Else Settings
import os

batch_mode = os.getenv("LOADL_ACTIVE", None)
job_name = os.getenv("LOADL_JOB_NAME", None)
execution_mode = os.getenv("LOADL_STEP_TYPE", None)
if batch_mode == "yes":
    import matplotlib

    matplotlib.use("Agg")
    print("In batch mode")

import argparse
import datetime
import itertools as itert
import pickle
import sys
import time as clock
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

# 0.1. Standard Imports
import numpy as np
import scipy.integrate as scigrate

# import scipy.ndimage.filters as filters
# import scipy.io as sio
# from scipy import constants
# import scipy.integrate as integ
import scipy.interpolate as inter
import scipy.optimize as opt
import scipy.signal as scisig
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D as ax3d
# from scipy.integrate.quadrature import simps as simps
from scipy.interpolate import RectBivariateSpline

# from psi_2018.testing.pfs_batch.development.fort_convolve.learning import custer
# from fort_convolve.learning import custer


sys.path.append(os.path.expanduser("~/"))
sys.path.append(os.path.expanduser("~/inference-tools/"))
sys.path.append(os.path.expanduser("~/pystark/"))

start_time = clock.time()

# from pystark.rosato_profile import rosato_profile


# 0.2. Inference Tools

# from inference.mcmc import GibbsChain, TemperedChain #, ParallelTempering

# print(TemperedChain, TemperedChain.__file__)

# 0.3. Gahle Tools
# from gahle_tools.pdf_summary.pdf_summary_v1_1 import pdf_summary


# 0.4. Plot Settings
font = {
    # "family": "normal",
    "weight": "bold",
    "size": 14
}

mpl.rc("font", **font)


# Tools
def crop_min(data, min):
    index = np.where(data < min)[0]

    data[index] = min

    return data


def thin(array, thin, alt=False):
    """

    :param array: array to thin
    :param thin: return every other n'th thing in the array
    :param alt: return a list of n'th long
    :return: a list of the thinned array
    """

    thin_array = []
    if alt:
        thin = len(array) / thin
    else:
        pass

    indicies = np.arange(0, len(array), thin, dtype="int")

    for index in indicies:
        thin_array.append(array[index])

    return thin_array


def in_between(x, range):
    """

    :param x: input array/list
    :param range:
    :return:
    """

    # return np.where((min(range) < x) and (x < max(range)))
    return np.where([t > min(range) and t < max(range) for t in x])[0]


def middle_section(y, x, pc, return_bounds=False):
    pc /= 100

    cumtrapz = scigrate.cumtrapz(y, x)
    cumtrapz /= max(cumtrapz)

    bounds = [(1 - pc) / 2, pc + (1 - pc) / 2]

    # print( cumtrapz )
    # print( bounds )
    # print( ((min(bounds) < cumtrapz) and (cumtrapz < max(bounds))).all() )

    condition0 = min(bounds) < cumtrapz
    condition1 = cumtrapz < max(bounds)

    # condition = [c0 and c1 for c0 in condition0 and c1 in condition1]

    bounds_indices = [np.where(condition0)[0][0], np.where(condition1)[-1][-1]]

    if return_bounds:
        return bounds_indices


def whats_the_noise(y, x, range):
    y_index = in_between(x, range)

    return np.std(y[y_index])


def whats_the_signal(y, x, range):
    y_index = in_between(x, range)

    return np.mean(y[y_index])


def clip_data(x_data, y_data, x_range):
    """
    returns sections of x and y data that is within the desired x range

    :param x_data:
    :param y_data:
    :param x_range:
    :return:
    """

    x_index = np.where((min(x_range) < x_data) & (x_data < max(x_range)))

    # print(x_index)

    return x_data[x_index], y_data[x_index]


def spectrum_slicing(data, range):
    """
    is compariable to clip_data() above

    :param data:
    :param range:
    :return:
    """

    x_data_old, y_data_old = data
    wave_index = np.where((x_data_old < max(range)) & (x_data_old > min(range)))

    x_data = x_data_old[wave_index]
    y_data = y_data_old[wave_index]

    return x_data, y_data


def where_approx(xs, x_data):
    """
    returns a list of indices of the x points in the x data that is closest to the value being searched

    :param xs: desired points in x data
    :param x_data: x data
    :return:
    """

    pixels = []

    for x in xs:
        pixels.append(np.array([abs(x - t) for t in x_data]).argmin())

    return pixels


def save(data, filename, info=False):
    """
    A function for pickleing a variable in one line. This does close the created .p.

    :param data: varriable to pickle
    :param filename: file_path and filename of .p to be created
    :param info: if True prints: Saved! filename
    :return: nothing
    """

    file = open(filename, "wb")
    pickle.dump(data, file)
    file.close()

    if info:
        print("Saved!: ", filename)


def load(filename, info=False):
    """
    A function for pickleing a variable in one line. This does close the created .p.

    :param data: varriable to pickle
    :param filename: file_path and filename of .p to be created
    :param info: if True prints: Load! filename
    :return: variable with the data of the .p
    """

    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()

    if info:
        print("Loaded!: ", filename)

    return data


def time_stamp(short=True, date=False):
    now = datetime.datetime.now()

    if date:
        return now

    else:
        year = str(now.year)
        month = str(now.month)
        day = str(now.day)

        hour = str(now.hour)
        minute = str(now.minute)
        second = str(now.second)
        microsecond = str(now.microsecond)

        if short:
            stamp = year + month + day + hour + minute + second + microsecond
        else:
            stamp = (
                year + month + day + "_" + hour + minute + second + "_" + microsecond
            )

        return stamp
        # return '_' + stamp


def do_nothing():
    """
    does nothing

    :return: nothing
    """

    pass


# Plot tools
def ax_plot(
    ax,
    data,
    x=None,
    log=False,
    xlog=False,
    xlabel=None,
    ylabel=None,
    multi=None,
    leg=None,
    linestyle="-",
    ylim=None,
    xlim=None,
):
    if x is not None:
        if multi == "fake":
            for tmp, tmp_data in enumerate(data):
                if leg is not None:
                    ax.plot(x[tmp], tmp_data, linestyle, label=leg[tmp])
                else:
                    ax.plot(x[tmp], tmp_data, linestyle)  # , 'x--')

        else:
            if leg is not None:
                ax.plot(x, data, linestyle, label=leg)
            else:
                ax.plot(x, data, linestyle)

    else:
        if multi == "fake":
            for tmp_data in data:
                ax.plot(tmp_data, linestyle)  # , 'x--')

        else:
            ax.plot(data, linestyle)  # , 'x--')

    if log:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    if leg is not None:
        leg_list = ax.legend()
        leg_list.draggable()

    pass


def plot(
    data,
    x=None,
    log=False,
    xlog=False,
    xlabel=None,
    ylabel=None,
    ylim=None,
    xlim=None,
    fsize=14,
    multi=None,
    sharey=False,
    sharex=False,
    leg=None,
    linestyle="-",
):
    "look up kwargs"

    # font = {"family": "normal", "weight": "bold", "size": fsize}
    font = {"weight": "bold", "size": fsize}

    mpl.rc("font", **font)

    if multi is not None and multi != "fake":
        fig, ax = plt.subplots(1, multi, sharey=sharey, sharex=sharex)

        for tk, tmp_ax in enumerate(ax):
            if x is not None:
                tmp_ax.plot(x[tk], data[tk], linestyle)  # , 'x--')
            else:
                tmp_ax.plot(data[tk], linestyle)  # , 'x--')

            if log:
                tmp_ax.set_yscale("log")
            if xlog:
                tmp_ax.set_xscale("log")

            if xlabel is not None:
                tmp_ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax[0].set_ylabel(ylabel)
        # if ylim is not None: ax[0].set_ylim(ylim)
        # if xlim is not None: ax[0].set_xlim(xlim)

        fig.tight_layout()

        fig.show()

        ...

    else:
        fig, ax = plt.subplots()

        ax_plot(
            ax,
            data=data,
            x=x,
            log=log,
            xlog=xlog,
            xlabel=xlabel,
            ylabel=ylabel,
            multi=multi,
            leg=leg,
            linestyle=linestyle,
            ylim=ylim,
            xlim=xlim,
        )

        fig.tight_layout()

        fig.show()


def histagramm(sample, size=None, bins=15, save=None, xlabel=r"$\theta_{i}$"):
    fig, ax = plt.subplots()

    if size is not None:
        sample = thin(sample, size, True)

    ax.hist(sample, bins=bins)

    ax.set_xlabel(xlabel)

    if type(save) == str:
        fig.save(save)
    else:
        fig.show()


def close_plots():
    plt.close("all")

    pass


baker_list = [
    "Daljeet",
    "Andrew Jackson",
    "Fabio Federici",
    "Jack Leland",
    "Joe Allcock",
    "Laszlo",
    "Sam Gibson",
    "Simon Orchard",
    "Siobhan",
    "TianTian",
]


def cake_rota(people=baker_list):
    return people[int(np.random.rand() * len(people))]
