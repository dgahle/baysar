import numpy as np
from numpy import random

import scipy.io as sio
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import os, sys, io

from tulasa.data_processing import reduce_sample, hdi_estimator
from tulasa import fitting, general



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

    # print(x)


def plot_fm_old(posterior, lims=[1e11, 2e15]):

    fig, ax = plt.subplots(2, 1, sharey=True)

    linewidth = 3

    for counter in np.arange( len(posterior.posterior_components) ):
        ax[counter].plot(posterior.posterior_components[counter].x_data,
                         posterior.posterior_components[counter].forward_model(),
                         linewidth=linewidth)
        # , label=labels[counter], linewidth=linewidth)

        oplot_markers(ax[counter], posterior)

        ax[counter].set_xlim([min(posterior.posterior_components[counter].x_data),
                              max(posterior.posterior_components[counter].x_data)])


        ax[counter].set_ylabel(r'$I \ / \ phcm^{-2}sr^{-1}\AA^{-1}s^{-1}$')
    # ax[1].set_ylabel(r'$Spectral \ Radiance \ / \ phcm^{-2}sr^{-1}\AA^{-1}s^{-1}$')


    ax[1].set_xlabel(r'$Wavelength \ / \ \AA$')

    ax[0].set_yscale('log')
    ax[0].set_ylim(lims)

    fig.show()

    pass

def plot_fm(posterior, theta, error_res=True, ylim=[1e11, 1e15]):

    print( posterior(theta) )

    fig, ax = plt.subplots(3, 1)

    try:
        wave = posterior.posterior_components[0].x_data_exp
    except AttributeError:
        wave = posterior.posterior_components[0].x_data
    except:
        raise

    spectra = posterior.posterior_components[0].y_data
    error = posterior.posterior_components[0].error

    fm = posterior.posterior_components[0].forward_model()

    ax[0].plot(wave, spectra, label=r'$Spectra$')
    ax[0].fill_between(wave, spectra-error, spectra+error, alpha=0.2)

    ax[0].plot(wave, fm, 'pink', label=r'$Fit$')

    ax[0].set_ylim(ylim)

    ax[0].set_yscale('log')

    if error_res:
        res = (fm-spectra)/error
        res_label = r'$Residuals/Error$'
    else:
        res = (fm - spectra) / spectra
        res_label = r'$Residuals$'

    ax[1].plot(wave, np.zeros(len(wave)), 'k')
    ax[1].fill_between(wave, np.zeros(len(wave))-0.2, np.zeros(len(wave))+0.2, color='k', alpha=0.2)

    ax[1].plot(wave, spectra/max(spectra))
    ax[1].plot(wave, res, 'x', color='pink', label=res_label)

    te = posterior.plasma.plasma_state['electron_temperature']
    ne = posterior.plasma.plasma_state['electron_density']

    los = posterior.plasma.plasma_state['los']

    ax[2].plot(los, ne/1e13, 'r', label=r'$n_{e} \ / \ 10^{13} cm^{-3}$')
    ax[2].plot(los, te, 'b', label=r'$T_{e} \ / \ eV$')

    ax[2].set_xlabel(r'$LOS \ / \ cm$')

    ax[2].grid(True)

    leg = []

    for a in ax:

        tmp_l = a.legend()
        tmp_l.draggable()

        leg.append(tmp_l)

    fig.show()



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


def plot_guess(posterior, theta, pc_counter=0, chain=None, lines=None, y_lim=(1e10, 1e15),
               alpha=0.1, small=False, save=None, sample=None):

    if save is None: print( posterior(theta) )


    if small:
        # general.plot( [posterior.posterior_components[0].y_data,
        #                posterior.posterior_components[0].forward_model()],
        #               [posterior.posterior_components[0].x_data,
        #                posterior.posterior_components[0].x_data], multi='fake', log=True)

        pass

    else:
        plt.ion()

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        # fig, ax = plt.subplots(2, 1)

        try:
            x = posterior.posterior_components[pc_counter].x_data_exp
        except:
            x = posterior.posterior_components[pc_counter].x_data

        y = posterior.posterior_components[pc_counter].y_data
        y_err = posterior.posterior_components[pc_counter].error

        ax[0].set_title('$Fit \ Summary$')

        ax[0].plot(x, y, label='Exp')

        if chain is not None:

            # posterior( chain.mode() )
            # fm = posterior.posterior_components[0].forward_model()
            #
            # ax[0].plot(x, fm, label='Mode')  # , alpha=0.5)

            if sample is None:
                try:
                    sample = chain.get_sample()
                except ValueError:

                    sample = chain.theta[1::]

                except:
                    print('chain.get_sample() and chain.theta[1::] is not working')
                    raise


            if len(sample) < 100:
                num = len(sample)
            else:
                num = 100

            for counter in np.linspace(0, (len(sample) -1), num, dtype='int'):

                if lines is not None:

                    l_colours = ['yellow', 'purple', 'green']

                    for lconter, l in enumerate(lines):

                        try:
                            tmp_ems = posterior.posterior_components[pc_counter].lines[l].ems_profile
                        except AttributeError:
                            tmp_ems = posterior.posterior_components[pc_counter].lines[l].emission_profile
                        except:
                            raise

                        tmp_ems = 5 * tmp_ems / max(tmp_ems)

                        ax[2].plot(posterior.plasma.plasma_state['los'], tmp_ems, color=l_colours[lconter],
                                   linestyle='-', alpha=alpha)


                if counter == 0 and chain is not None:
                    posterior(chain.mode())
                    fm = posterior.posterior_components[pc_counter].forward_model()

                    ax[0].plot(x, fm, 'pink', label='95% Interval')

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_density'] / 1e13,
                               'red', label='$n_{e}[10^{13} \ cm^{-3}]$ 95% Interval')

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_temperature'],
                               'blue', label='$T_{e}[eV]$ 95% Interval')

                else:

                    posterior( sample[counter] )

                    fm = posterior.posterior_components[pc_counter].forward_model()

                    ax[0].plot(x, fm, 'pink', alpha=0.1)

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_density'] / 1e13,
                               'red', alpha=0.1)

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_temperature'],
                               'blue', alpha=0.1)

                pass

            posterior( chain.mode() )
            fm = posterior.posterior_components[pc_counter].forward_model()

            pass

        else:
            fm = posterior.posterior_components[pc_counter].forward_model()

            ax[0].plot(x, fm, label='Mode')  # , alpha=0.5)

            ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_density'] / 1e13,
                               'red', label='$n_{e}[10^{13} \ cm^{-3}]$ 95% Interval')

            ax[2].plot(posterior.plasma.plasma_state['los'],
                       posterior.plasma.plasma_state['electron_temperature'],
                       'blue', label='$T_{e}[eV]$ 95% Interval')


        ax[0].plot(x, y+y_err, 'r--', label='Exp error')
        ax[0].plot(x, y-y_err, 'r--')

        ax[0].set_ylim(y_lim)
        ax[0].set_yscale('log')

        ax[0].set_ylabel(r'$Spectral \ Radiance$')

        ax[1].plot(x, np.zeros( len(x) ), 'k-', label='0.0')
        ax[1].plot(x, np.zeros( len(x) ) + 0.2, 'k--', label='0.2')
        ax[1].plot(x, np.zeros( len(x) ) - 0.2, 'k--')
        ax[1].plot(x, (y - fm) / y, 'x')

        ax[1].set_ylim([-1, 1])

        ax[1].set_title(r'$Normalised \ Residuals$')
        # ax[1].set_ylabel(r'$Normalised \ Residuals$')
        ax[1].set_xlabel(r'$Wavelength \ / \ \AA$')

        ax[2].set_xlim([min(posterior.plasma.plasma_state['los']), max(posterior.plasma.plasma_state['los'])])

        # ax[2].set_ylabel(r'$Normalised \ Residuals$')
        ax[2].set_xlabel(r'$x\ / \ cm$')

        ax[2].grid(True)

        for tmp in ax[0:2]:
            tmp.set_xlim([min(x), max(x)])

        leg = []

        for counter in [0, 2]:
            tmp_leg = ax[counter].legend()
            tmp_leg.draggable()

            leg.append(tmp_leg)

        # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

        if save is not None:
            plt.tight_layout()

            plt.savefig(save)

            plt.close()

            pass
        else:
            fig.show()


def plot_guess2(posterior, sample=None, size=100, burn=int(1e4), pc_counter=0,
                lines=None, y_lim=(1e10, 1e15), alpha=0.1, small=False, save=None):

    # if save is None: print( posterior(theta) )


    if small:
        # general.plot( [posterior.posterior_components[0].y_data,
        #                posterior.posterior_components[0].forward_model()],
        #               [posterior.posterior_components[0].x_data,
        #                posterior.posterior_components[0].x_data], multi='fake', log=True)

        pass

    else:
        plt.ion()

        fig, ax = plt.subplots(3, 1) # , figsize=(10, 10))
        # fig, ax = plt.subplots(2, 1)

        try:
            x = posterior.posterior_components[pc_counter].x_data_exp
        except:
            x = posterior.posterior_components[pc_counter].x_data

        y = posterior.posterior_components[pc_counter].y_data
        y_err = posterior.posterior_components[pc_counter].error

        ax[0].set_title('$Fit \ Summary$')

        ax[0].plot(x, y, label='Exp')
        ax[1].plot(x, y/max(y))

        sample = sample[:, burn:, :]
        sample = sample.reshape(sample.shape[0] * sample.shape[1], sample.shape[2])

        indicies = np.linspace(burn, len(sample) - 1, size, dtype=int)

        for counter in indicies:

            test = posterior(sample[counter, :])

            if test > -1e50:

                if lines is not None:

                    l_colours = ['yellow', 'purple', 'green']

                    for lconter, l in enumerate(lines):

                        try:
                            tmp_ems = posterior.posterior_components[pc_counter].lines[l].ems_profile
                        except AttributeError:
                            tmp_ems = posterior.posterior_components[pc_counter].lines[l].emission_profile
                        except:
                            raise

                        tmp_ems = 5 * tmp_ems / max(tmp_ems)

                        if counter == 0:
                            ax[2].plot(posterior.plasma.plasma_state['los'], tmp_ems, color=l_colours[lconter],
                                       linestyle='-', label=r'$\epsilon - profile$')
                        else:
                            ax[2].plot(posterior.plasma.plasma_state['los'], tmp_ems, color=l_colours[lconter],
                                       linestyle='-', alpha=alpha)

                if counter == 0:

                    fm = posterior.posterior_components[pc_counter].forward_model()

                    ax[0].plot(x, fm, 'pink', label='95% Interval')

                    ax[1].plot(x, (y - fm) / y_err, 'x', color='pink', label='residuals')

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_density'] / 1e13,
                               'red', label='$n_{e}[10^{13} \ cm^{-3}]$ 95% Interval')

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_temperature'],
                               'blue', label='$T_{e}[eV]$ 95% Interval')

                else:


                    fm = posterior.posterior_components[pc_counter].forward_model()

                    ax[0].plot(x, fm, 'pink', alpha=alpha)

                    ax[1].plot(x, (y - fm) / y_err, 'x', color='pink', alpha=alpha)

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_density'] / 1e13,
                               'red', alpha=alpha)

                    ax[2].plot(posterior.plasma.plasma_state['los'],
                               posterior.plasma.plasma_state['electron_temperature'],
                               'blue', alpha=alpha)


        ax[0].fill_between(x, y-y_err, y+y_err, alpha=0.2)
        # ax[0].plot(x, y+y_err, 'r--', label='Exp error')
        # ax[0].plot(x, y-y_err, 'r--')

        ax[0].set_ylim(y_lim)
        ax[0].set_yscale('log')

        ax[0].set_ylabel(r'$Spectral \ Radiance$')

        ax[1].plot(x, np.zeros( len(x) ), 'k-')
        ax[1].fill_between(x, np.zeros( len(x) ) - 0.2,
                              np.zeros( len(x) ) + 0.2, alpha=0.2)

        # ax[1].plot(x, (y - fm) / y_err, 'x', alpha=alpha)

        ax[1].set_ylim([-1, 1])

        # ax[1].set_title(r'$Normalised \ Residuals$')
        ax[1].set_title(r'$Error \ Normalised \ Residuals$')
        # ax[1].set_ylabel(r'$Normalised \ Residuals$')
        ax[1].set_xlabel(r'$Wavelength \ / \ \AA$')

        ax[2].set_ylim([0, 100])
        ax[2].set_xlim([min(posterior.plasma.plasma_state['los']), max(posterior.plasma.plasma_state['los'])])

        # ax[2].set_ylabel(r'$Normalised \ Residuals$')
        ax[2].set_xlabel(r'$x\ / \ cm$')

        ax[2].grid(True)

        for tmp in ax[0:2]:
            tmp.set_xlim([min(x), max(x)])

        leg = []

        for counter in [0, 2]:
            tmp_leg = ax[counter].legend()
            tmp_leg.draggable()

            leg.append(tmp_leg)

        # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

        if save is not None:
            plt.tight_layout()

            plt.savefig(save)

            plt.close()

            pass
        else:
            fig.show()


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

                pass

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

        f_rec.append( posterior.posterior_components[0].lines[balmer].f_rec )

        tmp_nf = posterior.plasma.plasma_state['D']['0']['conc']
        tmp_te = posterior.plasma.plasma_state['electron_temperature']
        tmp_ne = posterior.plasma.plasma_state['electron_density']
        tmp_ems = posterior.posterior_components[0].lines[balmer].ems_profile

        stark_ne.append( sum(tmp_ems * tmp_ne) / sum(tmp_ems) )
        te_ems.append( sum(tmp_ems * tmp_te) / sum(tmp_ems) )
        # neutral_fraction.append( np.log10(tmp_nf / stark_ne[-1]) )
        neutral_fraction.append( tmp_nf / stark_ne[-1] )

    # general.plot([stark_ne, f_rec], multi=2)

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

        pass
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

                pass

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


    # general.plot([stark_ne, f_rec], multi=2)

    stark_ne = [t for t in stark_ne if ~np.isnan(t)]
    te_ems = [t for t in te_ems if ~np.isnan(t)]
    conc = [t for t in conc if ~np.isnan(t)]

    # print(tau)


    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # ne
    try:
        ax[0, 0].hist(stark_ne, bins=bins)
    except:
        print(stark_ne)

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

        pass
    else:
        fig.show()

    pass


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

                # print(los_sections)
                # print(los_sections_diff)

                if num_sections > 1:
                    dls = []

                    for counter in np.arange(num_sections):

                        counter = int(counter*2)

                        dls.append(los_sections_diff[counter])

                    tmp_dl = sum(dls)

                else:
                    tmp_dl = sum(los_sections_diff)

                # print(tmp_dl)

                dl.append(tmp_dl)

            else:

                print('num_hdi_indicies is odd')
                # los_indices = general.middle_section(tmp_ems, los, pc, True)
                #
                # tmp_dl = abs(los[max(los_indices)] - los[min(los_indices)])
                #
                # dl.append(tmp_dl)

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

        pass
    else:
        fig.show()


def ess_plot(data, params, times=50, burn = int(1.5e3)):

    fig, ax = plt.subplots(1, 2, sharex=True)

    ac_time = []

    landmarks = np.linspace(burn+500, len(data['lnprobs'][0, :]), times, dtype=int)

    # print(landmarks)

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

    pass


def diagnostics(data, burn=int(1e4), essticks=None):

    'fake chain.plot_diagnostics()'

    fig, ax = plt.subplots(2, 2)

    its = np.arange( len(data['lnprobs'][0, :]) )
    its = np.array(general.thin(its, 500, True))

    # print(type(its))

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

            # print('hello ', essticks)
        else:
            a.set_xticks(xticks_loc, ([str(l) for l in xticks_loc]))


    # fig.tight_layout()

    fig.show()


def hist1d(data, params4, xlabel=[['1', '2'], ['3', '4']], burn=int(1e3), bins=50):

    fig, ax = plt.subplots(2, 2, sharex=False, sharey=True)

    # if size is not None:
    #     sample = thin(sample, size, True)

    for counter0, aa in enumerate(ax):

        for counter1, a in enumerate(aa):

            a.hist(data['chain'][:, int(burn):, params4[counter0][counter1]].flatten(), bins=bins)

            a.set_xlabel(xlabel[counter0][counter1])

    # fig.tight_layout()

    fig.show()

    pass


from scipy.stats import gaussian_kde


def histmd(data, params, labels=None, burn=0, bins=20, save=None):

    l = len(params)

    fig, ax = plt.subplots(l, l, sharex=False, sharey=False)

    ax[0, 0].set_yticks([])

    for counter0 in np.arange(l):

        # tmpdata = data['chain'][:, int(burn):, params[counter0]].flatten()
        tmpdata = data[int(burn):, params[counter0]]

        # kde = gaussian_kde(tmpdata)
        #
        # ax[l, l].plot(xra[l], kde(xra[l]))

        ax[counter0, counter0].hist(tmpdata, bins)

        # ax[counter0, counter0].set_xticks([])
        # ax[counter0, counter0].set_yticks([])

        for counter1 in np.arange(l):

            if (counter0==counter1): # or (counter0<counter1):
                pass
            else:
                tmpdata2 = data[int(burn):, params[counter1]].flatten()

                ax[counter0, counter1].hist2d(tmpdata2, tmpdata, int(bins))

            if counter0 != (l-1):
                # if labels is not None:
                #     ax[counter1, 0].set_ylabel(labels[counter1])

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

    # fig.tight_layout()

    fig.show()


def a_time_evolution(data, param, burn=int(1e4), interavls=10):

    data = data['chain'][:, int(burn):, param].flatten()

    fig, ax = plt.subplots()

    flags = np.linspace(0, len(data), interavls, dtype=int)

    te = np.logspace(0, np.log10(50), 20)

    for counter, tmp in enumerate(flags):

        if tmp == 0:
            pass
        else:
            # tmpp, tmpdata = np.histogram(data[flags[counter-1]:tmp], bins)

            # kde = gaussian_kde(data[flags[counter-1]:tmp])
            kde = gaussian_kde(data[:tmp])

            ax.plot(te, kde(te), label=str(counter))



    ax.set_ylabel(r'$P(T_{e})$')
    ax.set_xlabel(r'$T_{e}$')

    # ax.set_yscale('log')

    leg = ax.legend()
    leg.draggable()

    fig.show()

def plot_p0_probs(posterior, p0):

    p0_probs = []

    for p in p0:
        p0_probs.append(posterior(p))

    general.plot(abs(np.array(p0_probs)), log=True)

def plot_fm_old2(posterior, theta, plasma=False):

    posterior(theta)

    data = [posterior.posterior_components[0].y_data,
            posterior.posterior_components[0].forward_model()]

    general.plot(data, multi='fake', log=True)

    if plasma:

        data = [posterior.plasma.plasma_state['electron_temperature'],
                posterior.plasma.plasma_state['electron_density'] / 1e13]

        general.plot(data, multi='fake')


def plot_fit(posterior, sample, size=100, alpha=0.1, ylim=(1e10, 1e16)):

    fig, ax = plt.subplots(2, 1, sharex=True)

    # ax[0] plot data and fit
    spectra = posterior.posterior_components[0].y_data
    error = posterior.posterior_components[0].error

    try:
        waves = posterior.posterior_components[0].x_data_exp
    except AttributeError:
        waves = posterior.posterior_components[0].x_data
    except:
        raise

    ax[0].plot(waves, spectra, label='Data')
    ax[0].plot(waves, spectra+error, 'r--', label='Error')
    ax[0].plot(waves, spectra-error, 'r--')

    ax[0].plot(np.zeros(10), 'pink', label='Fit')

    ax[0].set_ylim(ylim)

    ax[0].set_yscale('log')

    ax[0].set_ylabel(r'$ph/cm^{2}/ sr^{1}/A^{1}/ s^{1}$')

    # ax[1] plot normalies data and error normalised residuals
    ax[1].plot(waves, spectra/max(spectra))
    ax[1].plot(np.zeros(10), 'bx')

    ax[1].plot(waves, np.zeros(len(waves))+0.2, 'k--')
    ax[1].plot(waves, np.zeros(len(waves)), 'k--')
    ax[1].plot(waves, np.zeros(len(waves))-0.2, 'k--')

    ax[1].set_ylim([-30, 30])
    ax[1].set_xlim([min(waves), max(waves)])

    ax[1].set_ylabel(r'$\sigma - Normalised \ Residuals$')
    ax[1].set_xlabel(r'$Wavelength \ / \ \AA$')


    for counter0 in np.linspace(0, len(sample)-1, size, dtype=int):

        posterior(sample[counter0])

        if posterior.posterior_components[0].calibrated:
            tmp_fit = posterior.posterior_components[0].forward_model()
        else:
            tmp_fit = posterior.posterior_components[0].forward_model() * posterior.plasma.plasma_state['a_cal'][0]

        ax[0].plot(waves, tmp_fit, 'pink', alpha=alpha)

        ax[1].plot(waves, (spectra-tmp_fit)/error, 'bx', alpha=alpha)

    leg = ax[0].legend()
    leg.draggable()

    fig.show()


