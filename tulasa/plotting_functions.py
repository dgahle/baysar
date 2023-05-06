import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numpy import random
from scipy.interpolate import interp1d

from tulasa import general
from tulasa.data_processing import hdi_estimator, reduce_sample


def oplot_markers(ax, posterior):

    balmer = [float(l) for l in posterior.input_dict['physics']['D']['0']['lines']]
    n_ii = [float(l) for l in posterior.input_dict['physics']['N']['1']['lines']]
    n_iii = [float(l) for l in posterior.input_dict['physics']['N']['2']['lines']]
    n_iv = [float(l) for l in posterior.input_dict['physics']['N']['3']['lines']]
    n_v = [float(l) for l in posterior.input_dict['physics']['N']['4']['lines']]
    x = []

    for a in posterior.posterior_components:

        for b in a.lines:

            if b.species != 'X':
                pass
            elif b.cwl == list:
                x.append( *b.cwl )
            else:
                x.append( b.cwl)


    lines = [balmer, n_ii, n_iii, n_iv, n_v, x]
    colours = ['green', 'red', 'purple', 'brown', 'pink', 'black']

    alpha = 0.75
    linewidth = 2

    for tmp0, species in enumerate(lines):

        for line in species:

            ax.plot([line, line], [1e11, 2e16], color=colours[tmp0],
                    linestyle='--', linewidth=linewidth, alpha=alpha)


def plot_emission_profile(posterior):


    fig, ax = plt.subplots()

    # useful variables
    los = posterior.plasma.plasma_state['los']

    ne = posterior.plasma.plasma_state['electron_density']
    te = posterior.plasma.plasma_state['electron_temperature']

    balmer = posterior.posterior_components[0].lines[0].emission_profile

    n_ii = posterior.posterior_components[0].lines[2].emission_profile
    n_iii = posterior.posterior_components[0].lines[10].emission_profile
    n_iv = posterior.posterior_components[0].lines[14].emission_profile
    n_v = posterior.posterior_components[1].lines[-2].emission_profile

    linewidth = 3

    # okay first plot the te and ne profiles
    ax.plot(los, te, '--', label=r'$T_{e} \ / \ eV$', linewidth=linewidth)
    ax.plot(los, ne / 1e13, '--', label=r'$n_{e} \ / \ 10^{13} \ cm^{-3}$', linewidth=linewidth)

    # then plot the emission profile
    things_to_plot = [balmer, n_ii, n_iii, n_iv, n_v]
    labels = [r'$D_{\delta}$', r'$N \ II$', r'$N \ III$', r'$N \ IV$', r'$N \ V$']

    scale = [1e11, 6e12, 1e11, 1e12, 1e11]

    for counter, thing in enumerate(things_to_plot):
        ax.plot(los, 2 * thing / scale[counter], label=labels[counter], linewidth=linewidth)

    ax.set_xlabel(r'$LOS \ / \ cm$')

    leg = ax.legend()
    leg.draggable()

    fig.show()

    pass



def mini_matrix(chain, indicies, ref=None, burn=0, thin=1):
    plt.ion()
    for counter, tmp in enumerate(indicies):
        if ref is None:
            chain.matrix_plot(params=tmp, burn =int(burn), thin=int(thin))
        else:
            chain.matrix_plot(params=tmp, reference=ref[counter], burn=int(burn), thin=int(thin))


# TODO Update to use with inference.pdf_tool.Gaussian_KDE
def stark_pdf(chain=None, posterior=None, sample=None, size=None, thin=5, bins=15, balmer=0, save=None):
    if chain is not None:
        burn = chain.estimate_burn_in()
        try:
            sample = chain.get_sample(burn=burn, thin=thin)
        except ValueError:
            chain.burn = burn
            full_sample = chain.theta[1::]
            sample = []
            for counter in np.linspace(0, len(full_sample)-1, int( len(full_sample) / thin ), dtype=int ):
                sample.append(full_sample[counter])
        except:
            print('chain.get_sample() and chain.theta[1::] is not working')
            raise
        posterior = chain.posterior
    else:
        sample = reduce_sample(sample, size)

    stark_ne = []
    te_ems = []
    f_rec = []
    neutral_fraction = []
    for tmp_sample in sample:
        if chain is not None:
            chain.posterior(tmp_sample)
        else:
            posterior(tmp_sample)

        f_rec.append( posterior.posterior_components[0].lines[balmer].f_rec)
        tmp_nf = posterior.plasma.plasma_state['D']['0']['conc']
        tmp_te = posterior.plasma.plasma_state['electron_temperature']
        tmp_ne = posterior.plasma.plasma_state['electron_density']
        tmp_ems = posterior.posterior_components[0].lines[balmer].ems_profile
        stark_ne.append( sum(tmp_ems * tmp_ne) / sum(tmp_ems) )
        te_ems.append( sum(tmp_ems * tmp_te) / sum(tmp_ems) )
        # neutral_fraction.append( np.log10(tmp_nf / stark_ne[-1]) )
        neutral_fraction.append( tmp_nf / stark_ne[-1] )

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].hist(stark_ne, bins=bins)
    ax[0, 1].hist(te_ems, bins=bins)
    ax[1, 0].hist(f_rec, bins=bins)
    ax[1, 1].hist(neutral_fraction, bins=bins)


    ax[0, 0].set_xlabel(r'$n_{Stark} \ / \ cm^{-3}$')
    ax[0, 1].set_xlabel(r'$T_{e, Balmer} \ \ / eV $')
    ax[1, 0].set_xlabel(r'$f_{rec}$')
    ax[1, 1].set_xlabel(r'$n_{0}/n_{Stark}$')
    # ax[1, 1].set_xlabel(r'$log(n_{0}/n_{Stark})$')

    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        fig.show()


def impurity_pdf(chain=None, posterior=None, sample=None, size=None, species=None, thin=5, bins=15, line=0, save=None):
    if chain is not None:
        burn = chain.estimate_burn_in()
        try:
            sample = chain.get_sample(burn=burn, thin=thin)
        except ValueError:
            chain.burn = burn
            full_sample = chain.theta[1::]
            sample = []
            for counter in np.linspace(0, len(full_sample)-1, int( len(full_sample) / thin ), dtype=int ):
                sample.append(full_sample[counter])
        except:
            print('chain.get_sample() and chain.theta[1::] is not working')
            raise
        posterior = chain.posterior
    else:
        sample = reduce_sample(sample, size)

    stark_ne = []
    te_ems = []
    conc = []
    tau = []
    if type(line) != list:
        ion = posterior.posterior_components[0].lines[line].ion
        for tmp_sample in sample:
            posterior(tmp_sample)
            tmp_te = posterior.plasma.plasma_state['electron_temperature']
            tmp_ne = posterior.plasma.plasma_state['electron_density']
            tmp_n_ii_ne = posterior.posterior_components[0].lines[line].ems_ne
            # tmp_n_ii_te = posterior.posterior_components[0].lines[line].ems_te
            tmp_ems = posterior.posterior_components[0].lines[line].emission_profile
            stark_ne.append(tmp_n_ii_ne)
            te_ems.append(sum(tmp_ems * tmp_te) / sum(tmp_ems))
            tmp_conc = (posterior.plasma.plasma_state[species][ion]['conc'] / tmp_n_ii_ne) * 1e2
            conc.append(tmp_conc)
            tmp_tau = np.log10(posterior.plasma.plasma_state[species][ion]['tau'])
            tau.append(tmp_tau)
    else:
        for l in line:
            ion = posterior.posterior_components[0].lines[l].ion
            for tmp_sample in sample:
                posterior(tmp_sample)
                tmp_te = posterior.plasma.plasma_state['electron_temperature']
                tmp_ne = posterior.plasma.plasma_state['electron_density']
                tmp_n_ii_ne = posterior.posterior_components[0].lines[l].ems_ne
                # tmp_n_ii_te = posterior.posterior_components[0].lines[line].ems_te
                tmp_ems = posterior.posterior_components[0].lines[l].emission_profile
                stark_ne.append(tmp_n_ii_ne)
                te_ems.append(sum(tmp_ems * tmp_te) / sum(tmp_ems))
                tmp_conc = (posterior.plasma.plasma_state[species][ion]['conc'] / tmp_n_ii_ne) * 1e2
                conc.append(tmp_conc)
                tmp_tau = np.log10(posterior.plasma.plasma_state[species][ion]['tau'])
                tau.append(tmp_tau)

    stark_ne = [t for t in stark_ne if ~np.isnan(t)]
    te_ems = [t for t in te_ems if ~np.isnan(t)]
    conc = [t for t in conc if ~np.isnan(t)]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # ne
    ax[0, 0].hist(stark_ne, bins=bins)
    ax[0, 0].set_xlabel(r'$n_{e, X} \ / \ cm^{-3}$')
    # te
    ax[0, 1].hist(te_ems, bins=bins)
    ax[0, 1].set_xlabel(r'$T_{e, X} \ / \ eV$')
    # conc
    ax[1, 0].hist(conc, bins=bins)
    ax[1, 0].set_xlabel(r'$n_{X}/n_{e, X \ n} \ [\%]$')
    # tau
    ax[1, 1].hist(tau, bins=bins)
    ax[1, 1].set_xlabel(r'$log_{10} ( \tau_{x} )$')

    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        fig.show()

def line_pdf(posterior, sample, size=None, burn=int(1e4), bins=30, line=0, save=None, pc=90):
    fig, ax = plt.subplots(1, 2)
    dl = []
    ems = []
    sample = sample[:, burn:, :]
    sample = sample.reshape(sample.shape[0]*sample.shape[1], sample.shape[2])
    indicies = np.linspace(burn, len(sample)-1, size, dtype=int)
    for counter in indicies:
        test = posterior(sample[counter, :])
        if test > -1e50:
            # ne = posterior.plasma.plasma_state['electron_density']
            # te = posterior.plasma.plasma_state['electron_temperature']
            los = posterior.plasma.plasma_state['los']
            tmp_ems = posterior.posterior_components[0].lines[line].emission_profile
            ems.append( sum(tmp_ems) )
            hdi_indicies = hdi_estimator(tmp_ems, pc=pc/100)
            num_hdi_indicies = len(hdi_indicies)
            if (num_hdi_indicies%2) == 0:
                los_sections = []
                for tmp_index in hdi_indicies[0]:
                    los_sections.append(los[tmp_index])

                num_sections = num_hdi_indicies / 2
                los_sections = los_sections[0]
                los_sections_diff = np.diff(los_sections)
                if num_sections > 1:
                    dls = []
                    for counter in np.arange(num_sections):
                        counter = int(counter*2)
                        dls.append(los_sections_diff[counter])
                    tmp_dl = sum(dls)
                else:
                    tmp_dl = sum(los_sections_diff)
                dl.append(tmp_dl)
            else:
                print('num_hdi_indicies is odd')
                break

    'plot of emission profile'
    'hist of the delta l'
    # print('len(dl) = ', len(dl), dl[0])
    ax[0].hist(dl, bins=bins, label=str(pc) + r'$ \ \%$')
    ax[0].set_xlabel(r'$\Delta l \ / \ cm$') # r'$log_{10} ( \tau_{x} )$')
    leg = ax[0].legend()
    leg.draggable()

    'hist of the emission'
    ax[1].hist(ems, bins=bins)
    ax[1].set_xlabel(r'$\epsilon \ / \ ph cm^{-2} sr^{-1} s^{-1}$') # r'$log_{10} ( \tau_{x} )$')

    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        fig.show()


def ess_plot(data, params, times=50, burn = int(1.5e3)):
    fig, ax = plt.subplots(1, 2, sharex=True)
    ac_time = []
    landmarks = np.linspace(burn+500, len(data['lnprobs'][0, :]), times, dtype=int)
    ess_cb = []
    for pcount, p in enumerate(params):
        ess_cb.append([])
        for tcount, tmp in enumerate(landmarks):
            # ac_time.append( fitting.autocorrelation_time( data['lnprobs'][0, burn:tmp] ) )
            ess_cb[pcount].append( fitting.ESS( data['chain'][:, burn:tmp, p].flatten() ) )
            print(p, tcount, '/', times)

        ax[0].plot(landmarks*1e-3, ess_cb[pcount])
        ess_cb_pc = ess_cb[pcount] / (landmarks - burn)
        ess_cb_pc = [1e2*e for e in ess_cb_pc]
        ax[1].plot(landmarks*1e-3, ess_cb_pc, label=str(p))

    ylabel = [r'$ESS$', r'$ESS (\%)$']
    xlabel = [r'$Steps \ (10^{3})$', r'$Steps \ (10^{3})$']
    for acount, a in enumerate(ax):
        a.set_ylabel(ylabel[acount])
        a.set_xlabel(xlabel[acount])

    # ax[1].set_yscale('log')
    leg = ax[1].legend()
    leg.draggable()
    fig.show()


def diagnostics(data, burn=int(1e4), essticks=None):
    'fake chain.plot_diagnostics()'
    fig, ax = plt.subplots(2, 2)
    its = np.arange( len(data['lnprobs'][0, :]) )
    its = np.array(general.thin(its, 500, True))
    for p in data['lnprobs']:
        p = general.thin(p, 500, True)
        ax[0, 0].plot(its*1e-3, p, 'x-')
        ax[0, 1].plot(its[1:]*1e-3, np.diff(p)/p[1:], 'x-')
    ylim = [[-500, -100], [-1, 0.1]]
    ylabel = [r'$logP(\theta | D)$', r'$diff(logP(\theta | D))$']
    xlabel = [r'$Steps \ (10^{3})$', r'$Steps \ (10^{3})$']

    for acount, a in enumerate(ax[0]):
        a.set_ylim(ylim[acount])
        a.set_ylabel(ylabel[acount])
        a.set_xlabel(xlabel[acount])

    essdata = []
    for p in np.arange( data['chain'].shape[2] ):
        essdata.append( fitting.ESS(data['chain'][:, burn:, p].flatten()) )
    essdata = np.array(essdata)
    ax[1, 0].bar(np.arange(len(essdata)), essdata)
    ax[1, 1].bar(np.arange(len(essdata)), 1e2*essdata/data['chain'].shape[1])
    ylabel = [r'$ESS$', r'$ESS \ (\%)$']
    for acount, a in enumerate(ax[1]):
        a.set_ylabel(ylabel[acount])
        xticks_loc = np.arange( data['chain'].shape[2] )
        if essticks is not None:
            a.set_xticks(xticks_loc, essticks)
        else:
            a.set_xticks(xticks_loc, ([str(l) for l in xticks_loc]))
    # fig.tight_layout()
    fig.show()


def hist1d(data, params4, xlabel=[['1', '2'], ['3', '4']], burn=int(1e3), bins=50):
    fig, ax = plt.subplots(2, 2, sharex=False, sharey=True)
    for counter0, aa in enumerate(ax):
        for counter1, a in enumerate(aa):
            a.hist(data['chain'][:, int(burn):, params4[counter0][counter1]].flatten(), bins=bins)
            a.set_xlabel(xlabel[counter0][counter1])
    # fig.tight_layout()
    fig.show()


from scipy.stats import gaussian_kde


def histmd(data, params, labels=None, burn=0, bins=20, save=None):
    l = len(params)
    fig, ax = plt.subplots(l, l, sharex=False, sharey=False)
    ax[0, 0].set_yticks([])
    for counter0 in np.arange(l):
        tmpdata = data[int(burn):, params[counter0]]
        ax[counter0, counter0].hist(tmpdata, bins)

        for counter1 in np.arange(l):
            if not (counter0==counter1):
                tmpdata2 = data[int(burn):, params[counter1]].flatten()
                ax[counter0, counter1].hist2d(tmpdata2, tmpdata, int(bins))
            if counter0 != (l-1):
                ax[counter0, counter1].set_xticks([])
            if counter1 != 0:
                if labels is not None:
                    ax[(l-1), counter0].set_xlabel(labels[counter0])
                ax[counter0, counter1].set_yticks([])

    if save is None:
        fig.show()
    else:
        plt.tight_layout()
        plt.savefig(save)
        plt.close()


def time_evloution(data, params4, ylabel=[['1', '2'], ['3', '4']], burn=int(1e3), size=100, alpha=0.01):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=False)
    for counter0, aa in enumerate(ax):
        for counter1, a in enumerate(aa):
            for tmpsample in data['chain'][:, int(burn):, params4[counter0][counter1]]:
                tmp_data = general.thin(tmpsample, size, True)
                tmpx = np.linspace(0, len(tmpsample), size) * 1e-3
                a.plot(tmpx, tmp_data, 'gx-', alpha=alpha)
            a.set_ylabel(ylabel[counter0][counter1])

            if counter0==1:
                a.set_xlabel(r'$Steps \ (10^{3})$')
    fig.show()


def a_time_evolution(data, param, burn=int(1e4), interavls=10):
    data = data['chain'][:, int(burn):, param].flatten()
    fig, ax = plt.subplots()
    flags = np.linspace(0, len(data), interavls, dtype=int)
    te = np.logspace(0, np.log10(50), 20)
    for counter, tmp in enumerate(flags):
        if tmp != 0:
            kde = gaussian_kde(data[:tmp])
            ax.plot(te, kde(te), label=str(counter))

    ax.set_ylabel(r'$P(T_{e})$')
    ax.set_xlabel(r'$T_{e}$')

    leg = ax.legend()
    leg.draggable()
    fig.show()

def plot_p0_probs(posterior, p0):
    p0_probs = []
    for p in p0:
        p0_probs.append(posterior(p))
    general.plot(abs(np.array(p0_probs)), log=True)

from numpy import array


def plot_posterior_components(posterior, sample, alpha=1, reference=None):
    pc=[]
    labels=[str(type(posterior)).split('.')[-1][:-2]]
    for p in posterior.posterior_components:
        pc.append([])
        labels.append(str(type(p)).split('.')[-1][:-2])

    for s in sample:
        posterior(s)
        for l, p in zip(pc, posterior.posterior_components):
            l.append(p())

    linemarker='o-'
    plt.figure()
    plt.plot(array(pc).sum(0), linemarker, label=labels[0], alpha=alpha)
    for p, l in zip(pc, labels[1:]):
        plt.plot(p, linemarker, label=l, alpha=alpha)

    if reference is not None:
        for counter, ref in enumerate(reference):
            plt.plot(np.zeros(len(sample))+ref, '--', color='C'+str(counter))

    plt.ylabel(r'$logP$')
    plt.xlabel(r'$Steps$')

    legend=plt.legend()
    legend.draggable()

    plt.tight_layout()
    plt.show()

<<<<<<< HEAD
def plot_fit_old(posterior, sample, size=100, alpha=0.1, ylim=(1e10, 1e16)):

    fig, ax = plt.subplots(2, 1, sharex=True)
=======
import numpy as np


def plot_fit(posterior, sample, size=None, alpha=None, ylim=(1e10, 1e16),
             error_norm=True, plasma_ref=None, filename=None):
    if size is None:
        size = len(sample)
    if alpha is None:
        alpha=1/size
    if alpha < 0.02:
        alpha = 0.02

    fig = plt.figure()
    ax_fit = fig.add_axes([.15, .57, .32, .39]) # [x0, y0, width, height]
    ax_res = fig.add_axes([.15, .125, .32, .39]) # [x0, y0, width, height]
    # ax_plasma = fig.add_axes([.6, .125, .32, .825]) # [x0, y0, width, height]
    ax_plasma = fig.add_axes([.6, .37, .32, .59]) # [x0, y0, width, height]
    ax_plasma2 = fig.add_axes([.6, .125, .32, .19]) # [x0, y0, width, height]

    plasma_color = 'tab:red'
    te_color = 'tab:blue'
    ax_plasma.set_xlabel(r'$LOS \ / \ cm$')
    ax_plasma.set_ylabel(r'$n_{e} \ / \ cm^{-3}$', color=plasma_color)
    ax_plasma.tick_params(axis='y', labelcolor=plasma_color)
    ax_te = ax_plasma.twinx()  # instantiate a second axes that shares the same x-axis
    ax_te.set_ylabel(r'$T_{e} \ / \ eV$', color=te_color)  # we already handled the x-label with ax1
    ax_te.tick_params(axis='y', labelcolor=te_color)

    ax_plasma2.set_xlabel(r'$LOS \ / \ cm$')
    ax_plasma2.set_ylabel(r'$n_{e} \ / \ cm^{-3}$', color=plasma_color)
    ax_plasma2.tick_params(axis='y', labelcolor=plasma_color)
    ax_te2 = ax_plasma2.twinx()  # instantiate a second axes that shares the same x-axis
    ax_te2.set_ylabel(r'$T_{e} \ / \ eV$', color=te_color)  # we already handled the x-label with ax1
    ax_te2.tick_params(axis='y', labelcolor=te_color)
>>>>>>> dev

    # ax[0] plot data and fit
    spectra = posterior.posterior_components[0].y_data
    error = posterior.posterior_components[0].error
    waves = posterior.posterior_components[0].x_data

    ax_fit.plot(waves, spectra, label='Data')
    ax_res.plot(waves, spectra/max(spectra), color='C0', label='Data')
    ax_fit.fill_between(waves, spectra-error, spectra+error, alpha=0.2)
    ax_fit.plot(np.zeros(10), 'pink', label='Fit')
    ax_fit.set_ylim(ylim)
    ax_fit.set_yscale('log')

    # ax[1] plot normalies data and error normalised residuals
    ax_res.plot(waves, spectra/max(spectra))
    ax_res.plot(np.zeros(10), 'bx')

    error_lims = [1e-3, 1e-2, 1e-1, 2e-1, 5e-1, 1e0]
    for el in error_lims:
        ax_res.fill_between(waves, np.zeros(len(waves)), np.zeros(len(waves))+el,
                            color='k', alpha=0.1)

    # ax_plasma.set_ylim(bottom=0)
    # ax_te.set_ylim(bottom=0)

    ax_res.set_ylim([1e-3, 1e1])
    ax_res.set_yscale('log')
    ax_res.set_xlim([min(waves), max(waves)])

    ax_fit.set_xticklabels([])
    ax_fit.set_xlim([min(waves), max(waves)])

    # ax[1].set_ylabel(r'$\sigma - Normalised \ Residuals$')
    ax_res.set_xlabel(r'$Wavelength \ / \ \AA$')

    los = posterior.plasma.los
    los_ne_theta=posterior.plasma.profile_function.electron_density.x_points
    if posterior.plasma.profile_function.electron_density.zero_bounds:
        los_ne_theta=los_ne_theta[1:-1]
    los_te_theta=posterior.plasma.profile_function.electron_temperature.x_points
    if posterior.plasma.profile_function.electron_temperature.zero_bounds:
        los_te_theta=los_te_theta[1:-1]
    ax_plasma.set_xlim([min(los), max(los)])
    ax_plasma2.set_xlim([min(los), max(los)])

    if plasma_ref is not None:
        te = plasma_ref['electron_temperature']
        ne = plasma_ref['electron_density']
        ax_plasma.plot(los, ne, 'x--', color=plasma_color)
        ax_te.plot(los, te, 'x--', color=te_color)
        ax_plasma2.plot(los, ne, 'x--', color=plasma_color)
        ax_te2.plot(los, te, 'x--', color=te_color)

    if error_norm:
        k_res = error
    else:
        k_res = spectra

    for counter0 in np.linspace(0, len(sample)-1, size, dtype=int):
        posterior(sample[counter0])
        tmp_fit = posterior.posterior_components[0].forward_model()
        ax_fit.plot(waves, tmp_fit, 'pink', alpha=alpha)
        ax_res.plot(waves, abs(spectra-tmp_fit)/k_res, color='pink',
                    marker='x', alpha=alpha/3)

        te = posterior.plasma.plasma_state['electron_temperature']
        ne = posterior.plasma.plasma_state['electron_density']
        ne_theta = np.power(10, posterior.plasma.plasma_theta['electron_density'])
        te_theta = np.power(10, posterior.plasma.plasma_theta['electron_temperature'])
        ax_plasma.plot(los, ne, color=plasma_color, alpha=alpha)
        ax_te.plot(los, te, color=te_color, alpha=alpha)
        ax_plasma2.plot(los_ne_theta, ne_theta, 'o', color=plasma_color, alpha=alpha)
        ax_te2.plot(los_te_theta, te_theta, 'o', color=te_color, alpha=alpha)

    leg = ax_fit.legend()
    leg.draggable()

<<<<<<< HEAD
    fig.show()


def plot_fit(posterior, sample, size=100, alpha=0.1,
             ylim=(1e10, 1e16), error_norm=False,
             plasma_ref=None, filename=None, nitrogen=None, deuterium=None):

    # fig, ax = plt.subplots(2, 1, sharex=True)
    fig = plt.figure()

    ax_fit = fig.add_axes([.15, .57, .32, .39]) # [x0, y0, width, height]
    ax_res = fig.add_axes([.15, .125, .32, .39]) # [x0, y0, width, height]
    ax_plasma = fig.add_axes([.6, .125, .32, .825]) # [x0, y0, width, height]

    plasma_color = 'tab:red'
    ax_plasma.set_xlabel(r'$LOS \ / \ cm$')
    ax_plasma.set_ylabel(r'$n_{e} \ / \ cm^{-3}$', color=plasma_color)
    # ax_plasma.set_ylim([0, 10e14])
    # ax_plasma.plot(los, ne, color=color, label=r'$n_{e}$')
    # ax_plasma.plot(los, n0, linestyle='--', color=color, label=r'$n_{0}$')
    ax_plasma.tick_params(axis='y', labelcolor=plasma_color)

    ax_te = ax_plasma.twinx()  # instantiate a second axes that shares the same x-axis

    te_color = 'tab:blue'
    ax_te.set_ylabel(r'$T_{e} \ / \ eV$', color=te_color)  # we already handled the x-label with ax1
    # ax_te.set_ylim([0, int(max(te)) + 2])  # we already handled the x-label with ax1
    # ax_te.plot(los, te, color=color)
    ax_te.tick_params(axis='y', labelcolor=te_color)


    # ax[0] plot data and fit
    spectra = posterior.posterior_components[0].y_data
    error = posterior.posterior_components[0].error

    try:
        waves = posterior.posterior_components[0].x_data_exp
    except AttributeError:
        waves = posterior.posterior_components[0].x_data
    except:
        raise

    ax_fit.plot(waves, spectra, label='Data')
    ax_res.plot(waves, spectra/max(spectra), color='C0', label='Data')
    ax_fit.fill_between(waves, spectra-error, spectra+error, alpha=0.2)

    ax_fit.plot(np.zeros(10), 'pink', label='Fit')

    ax_fit.set_ylim(ylim)

    ax_fit.set_yscale('log')

    # ax_fit.set_ylabel(r'$ph/ cm^{2}/ sr^{1}/A^{1}/ s^{1}$')

    # ax[1] plot normalies data and error normalised residuals
    ax_res.plot(waves, spectra/max(spectra))
    ax_res.plot(np.zeros(10), 'bx')

    error_lims = [1e-3, 1e-2, 1e-1, 2e-1, 5e-1, 1e0]
    for el in error_lims:
        ax_res.fill_between(waves, np.zeros(len(waves)), np.zeros(len(waves))+el,
                            color='k', alpha=0.1)

    # ax_plasma.set_ylim(bottom=0)
    # ax_te.set_ylim(bottom=0)

    ax_res.set_ylim([1e-3, 1e1])
    ax_res.set_yscale('log')
    ax_res.set_xlim([min(waves), max(waves)])

    ax_fit.set_xticklabels([])
    ax_fit.set_xlim([min(waves), max(waves)])

    # ax[1].set_ylabel(r'$\sigma - Normalised \ Residuals$')
    ax_res.set_xlabel(r'$Wavelength \ / \ \AA$')

    los = posterior.plasma.plasma_state['los']

    ax_plasma.set_xlim([min(los), max(los)])

    if plasma_ref is not None:
        te = plasma_ref['electron_temperature']
        ne = plasma_ref['electron_density']

        ax_plasma.plot(los, ne, 'x--', color=plasma_color)
        ax_te.plot(los, te, 'x--', color=te_color)


    if error_norm:
        k_res = error
    else:
        k_res = spectra


    for counter0 in np.linspace(0, len(sample)-1, size, dtype=int):

        posterior(sample[counter0])

        if posterior.posterior_components[0].calibrated:
            tmp_fit = posterior.posterior_components[0].forward_model()
        else:
            tmp_fit = posterior.posterior_components[0].forward_model() * posterior.plasma.plasma_state['a_cal'][0]

        ax_fit.plot(waves, tmp_fit, 'pink', alpha=alpha)

        ax_res.plot(waves, abs(spectra-tmp_fit)/k_res, color='pink',
                    marker='x', alpha=alpha/3)

        te = posterior.plasma.plasma_state['electron_temperature']
        ne = posterior.plasma.plasma_state['electron_density']

        ax_plasma.plot(los, ne, color=plasma_color, alpha=alpha)
        ax_te.plot(los, te, color=te_color, alpha=alpha)



    leg = ax_fit.legend()
    leg.draggable()

    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)

=======
    if filename is None:
        fig.show()
    else:
        plt.tight_layout() # breaks the code
        plt.savefig(filename)
        plt.close()
>>>>>>> dev
