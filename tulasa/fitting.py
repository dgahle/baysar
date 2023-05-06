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
import copy
import itertools as itert
import pickle
import sys
import time as clock
import warnings
from collections import OrderedDict

import emcee as thor
import matplotlib as mpl
import matplotlib.pyplot as plt

# 0.1. Standard Imports
import numpy as np
import psutil

# import scipy.ndimage.filters as filters
# import scipy.io as sio
# from scipy import constants
# import scipy.integrate as integ
import scipy.interpolate as inter
import scipy.optimize as opt
import scipy.signal as scisig
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D as ax3d
from scipy.integrate.quadrature import simps as simps
from scipy.interpolate import RectBivariateSpline

# from psi_2018.testing.pfs_batch.development.fort_convolve.learning import custer
# from fort_convolve.learning import custer


sys.path.append(os.path.expanduser("~/pystark/"))

start_time = clock.time()

# from pystark.rosato_profile import rosato_profile


# 0.2. Inference Tools
from inference.mcmc import (  # , TemperedChain #, ParallelTempering
    GibbsChain,
    HamiltonianChain,
    PcaChain,
)

# 0.3. Tulasa
from tulasa import general
from tulasa.data_processing import add_chain_bounds

# from tulasa.plotting_functions import plot_guess, stark_pdf, impurity_pdf

# from gahle_tools.pdf_summary.pdf_summary_v1_1 import pdf_summary


# 0.4. Plot Settings
font = {"family": "normal", "weight": "bold", "size": 14}

mpl.rc("font", **font)


# Tools
def ESS(x):
    # get the autocorrelation
    f = np.fft.irfft(abs(np.fft.rfft(x - np.mean(x))) ** 2)
    # remove reflected 2nd half
    f = f[: len(f) // 2]
    # check that the first value is not negative
    if f[0] < 0.0:
        raise ValueError("First element of the autocorrelation is negative")
    # cut to first negative value
    f = f[: np.argmax(f < 0.0)]
    # sum and normalise
    thin_factor = f.sum() / f[0]
    return int(len(x) / thin_factor)


def param_ESS(self):
    burn = self.estimate_burn_in()

    return [ESS(np.array(self.get_parameter(i, burn=burn))) for i in range(self.L)]


def autocovariances(probs, test=False):
    """
    takes 1.5 ms per prob (1D array)
    expects millons of probs hence very slow
    :param probs:
    :return:
    """

    l = len(probs)
    meanp = np.mean(probs)

    ac = []

    for counter, p in enumerate(probs):
        # tmp_ac = (probs[::-1][:l] - meanp) * b
        tmp_ac = probs[counter:l] - meanp
        tmp_ac *= probs[: l - counter] - meanp

        tmp_ac = sum(tmp_ac)
        tmp_ac /= l - counter

        ac.append(tmp_ac)

    if test:
        alt_ac = (probs - meanp) * (probs[::-1] - meanp)

    return np.array(ac)


def autocorrelation_time(probs):
    ac = autocovariances(probs)

    # print(ac)

    ac = ac[np.where(ac >= 0.0)]

    return 1 + 2 * sum(ac[1:] / ac[0])


def is_minimised(prob, no_data_points):
    """
    - checks that the average residual is equal to or less than the error
    - is used by the_big_fitter as a convergance criteria

    :param prob: probability of a fit
    :param no_data_points: number of data points in the fit
    :return: True/False if the average residual is equal to or less than the error
    """

    kprob = -2 * prob

    if kprob > no_data_points:
        return False
    else:
        return True


def ta_temps(k, c, x):
    """

    :param k: is the upper limit y(k, c, 0) = k
    :param c: is the lower limit y(k, c, inf) = c
    :param x: is the counter in the tempered annealing
    :return: temperature (for tempered annealing)
    """

    return (k - c) / np.power(10, x / 2) + c


# def the_big_fitter(posterior, theta, chain_bounds=None, return_chain=False, upper_temp=1e3, nfm=1, fast=None):
#
#     '''
#
#     :param posterior: posterior class that contains the forward model and probability calculation
#                       for a fit
#     :param theta: the proposal for the fit
#     :param chain_bounds: a function that takes advantage of set_boundaries and set_non_negative
#     :param return_chain: boulean that will return the chain object
#     :param upper_temp: upper temperature for tempered chains
#     :return: dictionary that contains the pdf info, metadata info and exp data
#     '''
#
#     new_prob = posterior(theta)
#
#     print()
#     print(-new_prob)
#     print()
#
#
#     t2_counter = 0
#     t2_modes = []
#
#     temps = []
#
#     diff_prob_0 = 1
#
#     if fast is None:
#         condition_t2 = any([(diff_prob_0 > 0), (t2_counter < 3)])
#     else:
#         condition_t2 = any([(diff_prob_0 > fast[0]), (t2_counter < 3)])
#
#     while condition_t2:
#
#         temp = ta_temps(upper_temp, 1, t2_counter)
#         temps.append(temp)
#
#         t2_counter += 1
#
#         chain = TemperedChain(posterior=posterior, start=theta, temperature=temp)
#
#         if chain_bounds is not None: chain_bounds(chain, theta)
#
#         advance = int(1e2)
#
#         print('Chain temperature =', np.round(temp, 2))
#         print('Advance chain: ', advance)
#
#         # start_time = clock.time()
#
#         chain.advance(advance)
#
#         # chain.burn = advance / 2
#
#         theta = chain.mode()
#
#         old_prob = new_prob
#         new_prob = posterior(theta)
#
#         t2_modes.append(new_prob)
#
#         diff_prob_0 += abs(new_prob - old_prob) - diff_prob_0
#
#         condition_t2 = any([(diff_prob_0 > 0), (t2_counter < 2)])
#
#         print(-new_prob)
#         # print(condition_t2, diff_prob_0, t2_counter)
#         print()
#
#         if t2_counter > 9: break
#
#     t1_counter = 0
#     t1_modes = []
#
#     diff_prob = 10
#
#     if fast is None:
#         condition_t1 = any([(diff_prob > 0), (t1_counter < 2)])
#     else:
#         condition_t1 = any([(diff_prob > fast[1]), (t1_counter < 2)])
#
#     while condition_t1:
#
#         t1_counter += 1
#
#         chain = GibbsChain(posterior=posterior, start=theta)
#
#         if chain_bounds is not None: chain_bounds(chain, thetanfm=2)
#
#         if fast is None:
#             advance = int(1.25e3)
#         else:
#             advance = int(5e2)
#
#         print('T1 iteration: ', t1_counter)
#         print('Advance chain: ', advance)
#
#         # start_time = clock.time()
#
#         chain.advance(advance)
#         chain.burn = int(advance * 0.25)
#
#         theta = chain.mode()
#
#         old_prob = new_prob
#         new_prob = posterior(theta)
#
#         t1_modes.append(new_prob)
#
#         print(-posterior(theta))
#         print()
#
#         diff_prob += abs(old_prob - new_prob) - diff_prob
#
#         condition_t1 = any([(diff_prob > 0), (t1_counter < 1)])
#
#         if t1_counter > 5: break
#
#     t1_converge = not condition_t1
#
#     end_time = clock.time()
#     run_time = (end_time - start_time)
#
#     if nfm > 1:
#         no_data_points = 0
#
#         for tmp in np.arange(nfm):
#             no_data_points += len(posterior.x[tmp])
#
#     else:
#         no_data_points = len(posterior.x)
#
#     minimised = is_minimised(posterior(chain.mode()), no_data_points)
#
#     print()
#     print('Converged?: ', t1_converge)
#     print('Minimised?: ', minimised)
#     print()
#
#     if run_time < 60.0:
#         print('Time elapsed (s): ', run_time)
#     elif run_time < 1800:
#         print('Time elapsed (mins): ', run_time / 60)
#     else:
#         print('Time elapsed (hr): ', run_time / 3600)
#
#     # evaluate pdfs and save to dictionary
#     sample = chain.get_sample()
#     sample_95 = chain.get_interval()
#     final_theta = chain.mode()
#
#     t2_modes = t2_modes[0:t2_counter]
#
#     tx_modes = t2_modes
#     [tx_modes.append(t) for t in t1_modes]
#
#     output = {}
#     output['meta'] = {}
#     output['meta']['t1_counter'] = t1_counter
#     output['meta']['t2_counter'] = t2_counter
#     output['meta']['t1_modes'] = t1_modes
#     output['meta']['t2_modes'] = t2_modes
#     output['meta']['tx_modes'] = tx_modes
#     output['meta']['t2_temps'] = temps
#     output['meta']['t1_converge?'] = t1_converge
#     output['meta']['run_time'] = run_time
#     output['meta']['minimised?'] = minimised
#
#     output['pdf'] = {}
#     output['pdf']['space'] = np.array(sample)
#     output['pdf']['space 95'] = np.array(sample_95)
#     output['pdf']['mode'] = np.array(final_theta)
#
#     output['spectra'] = {}
#     output['spectra']['noise'] = posterior.sigma
#     output['spectra']['error'] = posterior.error
#     output['spectra']['wave'] = posterior.x
#     output['spectra']['intensity'] = posterior.y
#     output['spectra']['int_func'] = posterior.int_func
#
#     # fits[key_scan][param_key]['output'] = output
#
#
#     if return_chain:
#         return output, chain
#     else:
#         return output

from concurrent.futures import ProcessPoolExecutor


# TODO: Add a notes input to add to the input
def the_many_timed_fitters(
    posterior,
    thetas,
    chain_bounds=None,
    return_chain=False,
    stamp=None,
    nfm=1,
    time_limit=1,
    show=False,
    max_workers=2,
    ans=None,
    labels=None,
    file_dir=None,
):
    tmp_start_time = clock.time()

    if stamp is None:
        stamp = general.time_stamp(short=False)
    else:
        pass

    if file_dir is None:
        file_dir = os.path.expanduser(
            "~/baysar_work/benchmarking/1d/data/" + stamp + "/"
        )
    else:
        pass

    # file_dir = file_dir + stamp + '/'

    try:
        os.mkdir(file_dir)
    except FileExistsError:
        pass
    finally:
        pass

    big_theta = [[], [], [], [], [], [], [], [], [], [], []]

    for key0, tmp_theta in enumerate(thetas):
        tmp_stamp = stamp + "_chain" + str(key0)

        tmp = (
            posterior,
            tmp_theta,
            chain_bounds,
            return_chain,
            tmp_stamp,
            nfm,
            time_limit,
            show,
            ans,
            labels,
            file_dir,
        )

        for key1, tmp1 in enumerate(tmp):
            big_theta[key1].append(tmp1)

    print("Number of chains: ", len(thetas))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        outlist = list(executor.map(the_timed_fitter, *big_theta))
        # outlist = list(executor.map(the_timed_fitter, big_theta))

    # outlist=[]
    #
    # for key, tmp in enumerate(big_theta):
    #
    #     outlist.append( the_timed_fitter( *tmp ) )

    output = {}
    output["chains"] = {}

    for key0, tmp_output in enumerate(outlist):
        output["chains"][key0] = tmp_output

        pass

    output["meta"] = {}
    output["meta"]["runtime"] = (clock.time() - tmp_start_time) / 3600
    output["meta"]["stamp"] = stamp

    return output


def the_timed_fitter(
    posterior,
    theta,
    chain_bounds=None,
    return_chain=False,
    stamp=None,
    nfm=1,
    time_limit=1,
    show=False,
    ans=None,
    labels=None,
    file_dir=None,
    notes=None,
    save=False,
    samplier="Gibbs",
    frequent_plots=None,
    save_steps=None,
):
    """

    :param posterior: posterior class that contains the forward model and probability calculation
                      for a fit
    :param theta: the proposal for the fit
    :param chain_bounds: a function that takes advantage of set_boundaries and set_non_negative
    :param return_chain: boulean that will return the chain object
    :param upper_temp: upper temperature for tempered chains
    :return: dictionary that contains the pdf info, metadata info and exp data
    """

    new_prob = posterior(theta)

    print()
    print(-new_prob)
    print()

    if nfm > 1:
        no_data_points = 0

        for tmp in np.arange(nfm):
            no_data_points += len(posterior.posterior_components[tmp].y_data)

    else:
        no_data_points = 0

        for tmp_spectrometers in posterior.posterior_components:
            no_data_points += len(tmp_spectrometers.x_data)

    if chain_bounds is not None:
        chain_bounds()
    else:
        pass

    if stamp is None:
        stamp = general.time_stamp(short=False)
    else:
        pass

    if samplier == "Gibbs":
        chain = GibbsChain(
            posterior=posterior, start=theta, widths=posterior.plasma.theta_widths
        )

        add_chain_bounds(chain)
    elif samplier == "Pca":
        chain = PcaChain(
            posterior=posterior, start=theta, widths=posterior.plasma.theta_widths
        )
    elif samplier == "Hamiltonian":
        chain = HamiltonianChain(
            posterior=posterior, start=theta
        )  # , widths=posterior.plasma.theta_widths)
    else:
        chain = GibbsChain(
            posterior=posterior, start=theta, widths=posterior.plasma.theta_widths
        )

        # add_chain_bounds(chain)

    # for p in chain.params:
    #
    #     p.growth_factor = 5
    #     p.chk_int = 10
    #     p.max_tries = 25

    if file_dir is None:
        file_dir = os.path.expanduser(
            "~/baysar_work/benchmarking/1d/data/" + stamp + "/figs/"
        )
    else:
        file_dir = file_dir + stamp + "/"

        try:
            os.mkdir(file_dir)
        except FileExistsError:
            pass
        except:
            raise

    ps_fname = file_dir + "plot_summary_" + stamp + ".png"
    pd_fname = file_dir + "plot_diagnostics_" + stamp + ".png"
    mp_fname = file_dir + "matrix_plot_" + stamp + ".png"
    stark_fname = file_dir + "balmer_" + stamp + ".png"
    n_ii_fname = file_dir + "n_ii_" + stamp + ".png"
    n_iii_fname = file_dir + "n_iii_" + stamp + ".png"
    n_iv_fname = file_dir + "n_iv_" + stamp + ".png"

    save_file = file_dir + "chain_save_" + stamp

    count = 1
    min_ess, avg_ess = 0, 0
    if save_steps is None:
        min_ess_target, avg_ess_target = 500, 1000
    else:
        min_ess_target, avg_ess_target = int(1e7), int(1e7)

    if frequent_plots is not None:
        # min_ess_target, avg_ess_target = 5, 10
        t_interval = frequent_plots
    elif save_steps is not None:
        assert type(save_steps) in (
            int,
            float,
        ), "save_steps is not a int or float it is a " + str(type(save_steps))
        t_interval = save_steps
    else:
        t_interval = 1

    while (time_limit > 0) and (
        (min_ess < min_ess_target) and (avg_ess < avg_ess_target)
    ):
        print("Iteration: ", count)

        chain_len = 1

        while chain_len < 20:
            try:
                # with warnings.catch_warnings():
                #
                #     warnings.simplefilter("ignore")

                if time_limit < t_interval:
                    chain.run_for(hours=time_limit)
                else:
                    chain.run_for(hours=t_interval)

            except AssertionError:
                print(AssertionError)
                pass
            except:
                chain_len_old = chain_len
                chain_len = len(chain.probs)

                assert chain_len > chain_len_old, (
                    "Chain is not taking steps. Old and new: "
                    + str(chain_len_old)
                    + " and "
                    + str(chain_len)
                )

            chain_len = len(chain.probs)

        time_limit -= t_interval
        count += t_interval

        if save_steps:
            save_file = file_dir + "chain_save_hr_" + str(int(count)) + "_" + stamp

            ps_fname = (
                file_dir + "plot_summary_hr_" + str(int(count)) + "_" + stamp + ".png"
            )

            pd_fname = (
                file_dir
                + "plot_diagnostics_hr_"
                + str(int(count))
                + "_"
                + stamp
                + ".png"
            )
            # mp_fname = file_dir + 'matrix_plot_hr_' + str(int(count)) + '_' + stamp + '.png'

        chain.save(save_file)

        try:
            chain.plot_diagnostics(filename=pd_fname, show=show)
        except:
            pass

        if chain.estimate_burn_in() < chain.L:
            chain.burn = chain.estimate_burn_in()

        try:
            ess = param_ESS(chain)
        except:
            ess = np.ones(len(chain.mode()))

        min_ess = min(ess)
        avg_ess = int(np.mean(ess))

        print("Minimum ESS: ", min_ess)
        print("Mean ESS: ", avg_ess)
        print()

        # try:
        #     # chain.matrix_plot(params=[3, 5, 7, 9], filename=mp_fname, show=show, reference=ans, labels=labels)
        #     chain.matrix_plot(filename=mp_fname, show=show, labels=[ str(l) for l in np.arange(len(chain.mode())) ]) # , reference=ans, labels=labels)
        # except:
        #     pass

        try:
            plot_guess(
                chain.posterior, chain.mode(), chain=chain, small=False, save=ps_fname
            )
        except:
            pass

        # if 'D' or 'H' in chain.posterior.plasma.plasma_state.keys():
        #
        #     stark_pdf(chain, save=stark_fname)
        #
        #     pass # stark_pdf

        # TODO: This needs sorting bro.
        if "N" in chain.posterior.plasma.plasma_state.keys():
            # tmp_line_key = None
            #
            # for tmp_line_counter, tmp_line in enumerate(chain.posterior.posterior_components[0]):
            #
            #     while tmp_line_key is None:
            #
            #         if tmp_line.species == 'N':
            #             tmp_line_key = tmp_line_counter
            #
            #         pass
            #
            #     break
            #
            # impurity_pdf(chain, species='N', thin=5, bins=15, line=tmp_line_key, save=nitrgoen_fname)

            try:
                impurity_pdf(
                    chain, species="N", thin=5, bins=15, line=0, save=n_ii_fname
                )
            except:
                pass

            # try:
            #     impurity_pdf(chain, species='N', thin=5, bins=15, line=-3, save=n_iii_fname)
            # except:
            #     pass
            #
            # try:
            #     impurity_pdf(chain, species='N', thin=5, bins=15, line=-2, save=n_iv_fname)
            # except:
            #     pass

            # impurity_pdf(chain, species='N', thin=5, bins=15, line=2, save=n_ii_fname)
            # impurity_pdf(chain, species='N', thin=5, bins=15, line=-3, save=n_iii_fname)
            # impurity_pdf(chain, species='N', thin=5, bins=15, line=-2, save=n_iv_fname)

        plt.close("all")

    chain.burn = chain.estimate_burn_in()

    minimised = is_minimised(posterior(chain.mode()), no_data_points)

    # pwidths = []
    # pwidths_cp = []
    #
    # for p in chain.params:
    #     pwidths.append(p.sigma_values)
    #     pwidths_cp.append(p.sigma_checks)

    print()
    print("Minimised?: ", minimised)
    print()

    # # evaluate pdfs and save to dictionary
    # sample = chain.get_sample()
    # sample_95 = chain.get_interval()
    # final_theta = chain.mode()

    output = {}
    output["input"] = posterior.input_dict

    output["meta"] = {}
    # output['meta']['proposals'] = chain.dump()
    # output['meta']['widths'] = pwidths
    # output['meta']['widths_checkpoints'] = pwidths_cp
    # output['meta']['minimised?'] = minimised
    # output['meta']['plot_locations'] = [pd_fname, mp_fname]
    output["meta"]["chain_file"] = save_file
    # output['meta']['notes'] = notes
    # output['meta']['ESS'] = {'ESS': ess, 'min': min_ess, 'avg': avg_ess}

    output["pdf"] = {}
    # output['pdf']['space'] = np.array(sample)
    # output['pdf']['space 95'] = np.array(sample_95)
    # output['pdf']['mode'] = np.array(final_theta)

    output["spectra"] = {}
    # output['spectra']['noise'] = [p.__dict__['error'] for p in posterior.posterior_components] # posterior.sigma
    output["spectra"]["error"] = [
        p.__dict__["error"] for p in posterior.posterior_components
    ]  # posterior.error
    output["spectra"]["wave"] = [
        p.__dict__["x_data"] for p in posterior.posterior_components
    ]  # posterior.x
    output["spectra"]["intensity"] = [
        p.__dict__["y_data"] for p in posterior.posterior_components
    ]  # posterior.y
    output["spectra"]["int_func"] = [
        p.__dict__["instrument_function"] for p in posterior.posterior_components
    ]  # posterior.int_func
    output["spectra"]["a_cal"] = [
        p.__dict__["a_cal"] for p in posterior.posterior_components
    ]  # posterior.int_func
    output["spectra"]["noise_region"] = [
        p.__dict__["noise_region"] for p in posterior.posterior_components
    ]  # posterior.int_func

    # fits[key_scan][param_key]['output'] = output

    if save:
        output_file = file_dir + "output_" + stamp

        general.save(output, output_file)

    if return_chain:
        return output, chain
    else:
        return output


def the_sly_fitter(
    posterior, theta, chain_bounds=None, return_chain=False, num_chains=1, nfm=1
):
    """

    :param posterior: posterior class that contains the forward model and probability calculation
                      for a fit
    :param theta: the proposal for the fit
    :param chain_bounds: a function that takes advantage of set_boundaries and set_non_negative
    :param return_chain: boulean that will return the chain object
    :param upper_temp: upper temperature for tempered chains
    :return: dictionary that contains the pdf info, metadata info and exp data
    """

    new_prob = posterior(theta)

    print()
    print(-new_prob)
    print()

    t1_counter = 0
    t1_modes = []

    diff_prob = 1

    condition_t1 = any(
        [t1_counter < num_chains]
    )  # any([(diff_prob > 0), (t1_counter < 2)])
    while condition_t1:
        t1_counter += 1

        chain = GibbsChain(posterior=posterior, start=theta)

        if chain_bounds is not None:
            chain_bounds(chain, thetanfm=2)

        if t1_counter == num_chains:
            advance = int(1.5e3)
            # advance = int(2.5e3)
            burn = int(advance * 0.2)
        else:
            advance = int(1e3)
            burn = int(advance * 0.5)
        # advance = int(1.25e3)

        print("T1 iteration: ", t1_counter)
        print("Advance chain: ", advance)

        # start_time = clock.time()

        chain.advance(advance)
        chain.burn = burn

        theta = chain.mode()

        old_prob = new_prob
        new_prob = posterior(theta)

        t1_modes.append(new_prob)

        print(-posterior(theta))
        print()

        diff_prob += abs(old_prob - new_prob) - diff_prob

        condition_t1 = any([(diff_prob > 0), (t1_counter < 1)])

        if t1_counter > 5:
            break

    t1_converge = not condition_t1

    end_time = clock.time()
    run_time = end_time - start_time

    if nfm > 1:
        no_data_points = 0

        for tmp in np.arange(nfm):
            no_data_points += len(posterior.x[tmp])

    else:
        no_data_points = len(posterior.x)

    minimised = is_minimised(posterior(chain.mode()), no_data_points)

    print()
    print("Converged?: ", t1_converge)
    print("Minimised?: ", minimised)
    print()

    if run_time < 60.0:
        print("Time elapsed (s): ", run_time)
    elif run_time < 1800:
        print("Time elapsed (mins): ", run_time / 60)
    else:
        print("Time elapsed (hr): ", run_time / 3600)

    # evaluate pdfs and save to dictionary
    sample = chain.get_sample()
    sample_95 = chain.get_interval()
    final_theta = chain.mode()

    t2_modes = t2_modes[0:t2_counter]

    tx_modes = t2_modes
    [tx_modes.append(t) for t in t1_modes]

    output = {}
    output["meta"] = {}
    output["meta"]["t1_counter"] = t1_counter
    output["meta"]["t1_modes"] = t1_modes
    output["meta"]["tx_modes"] = tx_modes
    output["meta"]["t1_converge?"] = t1_converge
    output["meta"]["run_time"] = run_time
    output["meta"]["minimised?"] = minimised

    output["pdf"] = {}
    output["pdf"]["space"] = np.array(sample)
    output["pdf"]["space 95"] = np.array(sample_95)
    output["pdf"]["mode"] = np.array(final_theta)

    output["spectra"] = {}
    output["spectra"]["noise"] = posterior.sigma
    output["spectra"]["error"] = posterior.error
    output["spectra"]["wave"] = posterior.x
    output["spectra"]["intensity"] = posterior.y
    output["spectra"]["int_func"] = posterior.int_func

    # fits[key_scan][param_key]['output'] = output

    if return_chain:
        return output, chain
    else:
        return output


def plot_pdf(chain, k, axis, log=False, x_lab=None):
    """
    this function takes the chain.marginalse function and produces a plot

    :param chain: chain object produced by inference_tools.*chain
    :param k: parameter index
    :param axis: x-axis to plot PDF against
    :param log: boulean for setting the x_scale(True)
    :param x_lab: a string for setting x-axis label
    :return: makes a plot - currently does not return the figure object
    """

    fig_pdf, ax_pdf = plt.subplots()

    if type(k) != int:
        for ik in k:
            if log:
                ax_pdf.semilogx(axis, chain.marginalise(ik)(axis), label=str(ik))
            else:
                ax_pdf.plot(axis, chain.marginalise(ik)(axis), label=str(ik))
        leg = ax_pdf.legend()
        leg.draggable

    else:
        if log:
            ax_pdf.semilogx(axis, chain.marginalise(k)(axis))
        else:
            ax_pdf.plot(axis, chain.marginalise(k)(axis))

    if x_lab is not None:
        ax_pdf.set_xlabel(x_lab)

    fig_pdf.show()


# tools old
"""
old tools that may or may not get used...
"""


class fit_cwl_post(object):
    def __init__(self, x_data, y_data, int_func=None):
        self.x_data = x_data
        self.y_data = y_data

        self.error = min(self.y_data) * 0.1 + np.sqrt(self.y_data)

        self.instrument_function = int_func
        self.instrument_function_matrix = self.int_func_sparce_matrix()

        self.get_bounds_and_widths()

    def __call__(self, theta):
        return -0.5 * sum(((self.peak(theta) - self.y_data) / self.error) ** 2)

    def min(self, theta):
        return -self.__call__(theta)

    def peak(self, theta):
        cwl, fwhm, ems, background = theta

        peak = np.exp(-0.5 * ((self.x_data - cwl) / fwhm) ** 2)
        peak = np.power(10, ems) * peak + np.power(10, background)

        return self.instrument_function_matrix.dot(peak)

    def int_func_sparce_matrix(self, theta=None):
        if theta is not None:
            int_func, res = theta
        else:
            int_func, res = (self.instrument_function, len(self.y_data))

        # res = len(data)
        shape = (res, res)
        matrix = np.zeros(shape)
        # matrix = sparse.lil_matrix(shape)
        buffer = np.zeros(res)

        long_int_func = np.concatenate((buffer, int_func[::-1], buffer))
        # long_int_func = np.concatenate((buffer + int_func[0], int_func, buffer + int_func[-1]))

        rang = np.arange(res)[::-1] + int((len(int_func) + 1) / 2)  # 16

        # self.k_matrix =

        for i, key_i in enumerate(rang):
            # # print(i, key_i)
            # try:
            #     matrix_i = long_int_func[key_i:key_i + res]  # matrix_i # [::-1]
            # except:
            #     print(long_int_func.shape)
            #     print(i, res, key_i)
            #     raise

            matrix_i = long_int_func[key_i : key_i + res]  # matrix_i # [::-1]
            k = np.sum(matrix_i)

            # print('k', k)

            if k == 0:
                k = 1

            # print(len(matrix_i))
            matrix[i, :] = matrix_i / k

        matrix = sparse.csc_matrix(matrix)
        # matrix = matrix.tocsc()

        # return matrix.T

        # to do convoutions output = matrix.dot(input)

        return matrix

    def get_bounds_and_widths(self):
        self.widths = [1, 0.1, 1, 1]
        self.bounds = [
            [min(self.x_data), max(self.x_data)],
            [0.05, 2.0],
            [0, 20],
            [0, 15],
        ]


def plot_fit_cwl(post, chain):
    fig, ax = plt.subplots(2, 1, sharex=True)

    # plot data
    ax[0].plot(post.x_data, post.y_data)

    # plot error bars
    ax[0].plot(post.x_data, post.y_data + post.error, "r--")
    ax[0].plot(post.x_data, post.y_data - post.error, "r--")

    # plot fit
    sample = chain.get_sample(thin=10)

    alpha = 1
    for tmp_sample in sample:
        ax[0].plot(post.x_data, post.peak(tmp_sample), "pink", alpha=alpha)

    ax[0].set_yscale("log")

    ax[0].set_ylabel(r"$Spectral \ Radiance$")

    residuals = (post.y_data - post.peak(chain.mode())) / post.y_data
    ax[1].plot(post.x_data, residuals, "x")

    ax[1].plot(post.x_data, np.zeros(len(post.x_data)), "k-")
    ax[1].plot(post.x_data, np.zeros(len(post.x_data)) - 0.2, "k--")
    ax[1].plot(post.x_data, np.zeros(len(post.x_data)) + 0.2, "k--")

    ax[1].set_ylim([-1, 1])

    ax[1].set_ylabel(r"$Normalised \ Residuals$")
    ax[1].set_xlabel(r"$Wavelength \ / \ \AA$")

    fig.show()


def fit_cwl(x_data, y_data, int_func, least=False, check=False):
    post = fit_cwl_post(x_data=x_data, y_data=y_data, int_func=int_func)
    # post = fit_cwl_post(x_data=x_data, y_data=y_data, background=background)

    # print(post.y, post.x)

    ems = np.log10(sum(post.y_data))  # simps(post.y_data, post.x_data) * 2.5

    # print('ems = ', ems)

    # ems = np.log10( ems )

    imax = np.where(post.y_data == max(post.y_data))[0][0]
    cwlx = x_data[imax]

    back = np.log10(min(post.y_data))

    # the = [cwl, ems, 0.15]
    the = [cwlx, 0.15, ems, back]

    # least = least
    if least:
        cwl_opt = opt.minimize(post.min, the, bounds=post.bounds)

        # print(the)
        # print(cwl_opt)

        return cwl_opt.x[0]

    else:
        chain = GibbsChain(posterior=post, start=the, widths=post.widths)

        for i in np.arange(len(post.bounds)):
            chain.set_boundaries(i, post.bounds[i])

        chain.advance(int(1e4))
        chain.burn = int(0.8e4)

        if check:
            chain.plot_diagnostics()

            # chain.matrix_plot(thin=10)

            plot_fit_cwl(post, chain)

        return chain.mode()[0]
        # return np.mean( chain.get_parameter(0) )


def fit_cwls(cwls, res, x_data, y_data, int_func, least=False):
    cwls_fitted = []

    for i, cwl in enumerate(cwls):
        # print( 'cwl', cwl )

        if type(res) == int:
            x_range = [cwl - res, cwl + res]
        elif type(res) == list:
            res_i = res[i]
            x_range = [cwl - res_i, cwl + res_i]
        else:
            x_range = [cwl - res, cwl + res]

        clipped_data = general.clip_data(x_data, y_data, x_range)

        cwl_fitted = fit_cwl(clipped_data[0], clipped_data[1], int_func, least=least)

        cwls_fitted.append(cwl_fitted)

        # print( 'cwls_fitted', cwls_fitted )

    return cwls_fitted


def wavelength_calibration(
    x_data, y_data, cwls, res, nist, int_func, least=False, kind="linear"
):
    cwls_fitted = fit_cwls(cwls, res, x_data, y_data, int_func, least=least)

    pixel_loc = general.where_approx(cwls_fitted, x_data)

    # print( cwls_fitted, pixel_loc )

    wave_cali = inter.interp1d(
        pixel_loc, nist, kind=kind, bounds_error=False, fill_value="extrapolate"
    )

    pixels = np.arange(len(x_data))

    new_wave = wave_cali(pixels)

    assert all([(nw != np.nan) and (nw != np.nan) for nw in new_wave])

    return new_wave


def de_noise(y_data, background):
    index = np.where(y_data < background)[0]

    y_data[index] = background

    return y_data


def swing_hammer(hammer_in):
    stormbreaker, p0, steps, intervals, filename, break_up = hammer_in

    print("Swinging Hammer!")

    counter = 0

    steps = int(steps / intervals)

    # for f in flags[1:]:
    while counter < intervals:
        output = {}

        if counter == 0:
            # np.save(filename+'_started', output)

            if break_up:
                filename += "_" + str(counter)

            start = p0
            # stormbreaker.run_mcmc(p0, steps)
        else:
            if break_up:
                filename = filename[0:-1] + str(counter)

            start = new_start

        stormbreaker.run_mcmc(start, steps)

        # # width = 50
        # for i, result in enumerate(stormbreaker.sample(start, iterations=steps)):
        #     # n = int((width + 1) * float(i) / steps)
        #     print(i, steps)
        # #     sys.stdout.write( "\r " + str(np.round(i/steps, 1)*100) + "\r % " +
        # #                       "\r[{0}{1}]".format('#' * n, ' ' * (width - n))  )
        # # sys.stdout.write("\n")

        output["lnprobs"] = stormbreaker.lnprobability
        output["chain"] = stormbreaker.chain

        np.save(filename, output)

        new_start = stormbreaker.chain[:, -1, :]

        if break_up:
            stormbreaker.reset()

        counter += 1

        print(filename, counter, intervals)

    # return 'done'


import sys
from collections import Mapping, Set, deque
from numbers import Number

zero_depth_bases = (str, bytes, Number, range, bytearray)
iteritems = "items"


def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()

    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, "__dict__"):
            size += inner(vars(obj))
        if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
            size += sum(
                inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s)
            )
        return size

    return inner(obj_0)


import copy


def hammer_swing(hammer):
    posterior, p0, steps, filename = hammer

    nwalkers = len(p0)
    ndim = len(p0[0])

    stormbreaker = thor.EnsembleSampler(nwalkers, ndim, posterior)

    output = {}

    print("Swinging Hammer!")

    stormbreaker.run_mcmc(p0, steps)

    output["lnprobs"] = stormbreaker.lnprobability
    output["chain"] = stormbreaker.chain

    output["objectMB"] = getsize(copy.deepcopy(stormbreaker)) / (1024 * 1024)
    output["virtual_memory"] = dict(psutil.virtual_memory()._asdict())

    np.save(filename, output)

    new_start = copy.copy(output["chain"][:, -1, :])

    del stormbreaker, output, posterior

    print("Hammer struck " + filename)

    return new_start


def tidy_away_all_the_hammers(hammers):
    filename, intervals, notes = hammers

    print("Tidying away hammers")

    fileroot = filename  # [0:-4]
    file_exst = ".npy"

    for interval in np.arange(intervals):
        tmp_file = fileroot + "_" + str(interval) + file_exst

        tmp_data = np.load(tmp_file).item()

        if interval == 0:
            data = {}

            for key in ["chain", "lnprobs"]:
                data[key] = tmp_data[key]

            data["data_size"] = {
                str(interval): {
                    "objectMB": tmp_data["objectMB"],
                    "virtual_memory": tmp_data["virtual_memory"],
                }
            }

        else:
            tmp_data = np.load(tmp_file).item()

            for key in ["chain", "lnprobs"]:
                data[key] = np.concatenate((data[key], tmp_data[key]), axis=1)

            data["data_size"][str(interval)] = {
                "objectMB": tmp_data["objectMB"],
                "virtual_memory": tmp_data["virtual_memory"],
            }

        del tmp_data

        print(interval + 1, "/", intervals)

    fileout = fileroot + "_all"  # + file_exst

    data["notes"] = notes

    try:
        np.savez(fileout, **data)
    except OverflowError:
        data["filename"] = filename
        np.savez(str(int(clock.time() * 1e7)), **data)
    except:
        raise


def swing_hammer_after_hammer(hammer):
    "input needs posterior, number of steps, filename and notes about the run"

    posterior, p0, steps, fileroot, notes = hammer
    # stormbreaker, p0, steps, intervals, filename, break_up = hammer_in

    print("Picking up hammer...")

    counter = 0

    intervals = 100

    steps = int(steps / intervals)

    # for f in flags[1:]:
    while counter < intervals:
        filename = fileroot + "_" + str(counter)

        if counter == 0:
            start = p0

        start = hammer_swing([posterior, start, steps, filename])

        counter += 1

        print(filename, counter, intervals)

    tidy_away_all_the_hammers([fileroot, intervals, notes])


def hamiltonian_start(posterior, start):
    chain = HamiltonianChain(posterior=posterior, start=start)

    # chain.run_for(60)
    chain.advance(100)

    return chain.mode()
