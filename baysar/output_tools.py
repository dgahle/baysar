import matplotlib.pyplot as plt
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
    ax_fit = fig.add_axes([.15, .37, .32, .59]) # [x0, y0, width, height]
    ax_res = fig.add_axes([.15, .125, .32, .19]) # [x0, y0, width, height]
    ax_plasma = fig.add_axes([.6, .125, .32, .825]) # [x0, y0, width, height]
    # ax_plasma = fig.add_axes([.6, .37, .32, .59]) # [x0, y0, width, height]
    # ax_plasma2 = fig.add_axes([.6, .125, .32, .19]) # [x0, y0, width, height]

    plasma_color = 'tab:red'
    te_color = 'tab:blue'
    ax_plasma.set_xlabel(r'$LOS \ / \ cm$')
    ax_plasma.set_ylabel(r'$n_{e} \ / \ cm^{-3}$', color=plasma_color)
    ax_plasma.tick_params(axis='y', labelcolor=plasma_color)
    ax_te = ax_plasma.twinx()  # instantiate a second axes that shares the same x-axis
    ax_te.set_ylabel(r'$T_{e} \ / \ eV$', color=te_color)  # we already handled the x-label with ax1
    ax_te.tick_params(axis='y', labelcolor=te_color)

    # ax_plasma2.set_xlabel(r'$LOS \ / \ cm$')
    # ax_plasma2.set_ylabel(r'$n_{e} \ / \ cm^{-3}$', color=plasma_color)
    # ax_plasma2.tick_params(axis='y', labelcolor=plasma_color)
    # ax_te2 = ax_plasma2.twinx()  # instantiate a second axes that shares the same x-axis
    # ax_te2.set_ylabel(r'$T_{e} \ / \ eV$', color=te_color)  # we already handled the x-label with ax1
    # ax_te2.tick_params(axis='y', labelcolor=te_color)

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

    # error_lims = [1e-3, 1e-2, 1e-1, 2e-1, 5e-1, 1e0]
    # for el in error_lims:
    #     ax_res.fill_between(waves, np.zeros(len(waves)), np.zeros(len(waves))+el,
    #                         color='k', alpha=0.1)

    # ax_plasma.set_ylim(bottom=0)
    # ax_te.set_ylim(bottom=0)

    ax_res.set_ylim([1e-3, 1e1])
    ax_res.set_yscale('log')
    ax_res.set_xlim([min(waves), max(waves)])

    ax_fit.set_xticklabels([])
    ax_fit.set_xlim([min(waves), max(waves)])

    # ax[1].set_ylabel(r'$\sigma - Normalised \ Residuals$')
    ax_res.set_xlabel(r'$Wavelength \ / \ \AA$')

    los=posterior.plasma.los
    los_ne_theta=posterior.plasma.profile_function.electron_density.x_points
    if posterior.plasma.profile_function.electron_density.zero_bounds:
        los_ne_theta=los_ne_theta[1:-1]
    los_te_theta=posterior.plasma.profile_function.electron_temperature.x_points
    if posterior.plasma.profile_function.electron_temperature.zero_bounds:
        los_te_theta=los_te_theta[1:-1]
    ax_plasma.set_xlim([min(los), max(los)])
    # ax_plasma2.set_xlim([min(los), max(los)])

    if plasma_ref is not None:
        te = plasma_ref['electron_temperature']
        ne = plasma_ref['electron_density']

        if 'electron_density_los' in plasma_ref:
            ax_plasma.plot(plasma_ref['electron_density_los'], ne, 'x--', color=plasma_color)
            # ax_plasma2.plot(plasma_ref['electron_density_los'], ne, 'x--', color=plasma_color)
        else:
            ax_plasma.plot(los, ne, 'x--', color=plasma_color)
            # ax_plasma2.plot(los, ne, 'x--', color=plasma_color)

        if 'electron_temperature_los' in plasma_ref:
            ax_te.plot(plasma_ref['electron_temperature_los'], te, 'x--', color=te_color)
            # ax_te2.plot(plasma_ref['electron_temperature_los'], te, 'x--', color=te_color)
        else:
            ax_te.plot(los, te, 'x--', color=te_color)
            # ax_te2.plot(los, te, 'x--', color=te_color)

    if error_norm:
        k_res = error
    else:
        k_res = spectra

    te_all=[]
    ne_all=[]
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

        te_all.append(te)
        ne_all.append(ne)

        ax_plasma.plot(los, ne, color=plasma_color, alpha=alpha)
        ax_te.plot(los, te, color=te_color, alpha=alpha)
        ax_plasma.plot(los_ne_theta, ne_theta, 'o', color=plasma_color, alpha=alpha)
        ax_te.plot(los_te_theta, te_theta, 'o', color=te_color, alpha=alpha)
        # ax_plasma2.plot(los_ne_theta, ne_theta, 'o', color=plasma_color, alpha=alpha)
        # ax_te2.plot(los_te_theta, te_theta, 'o', color=te_color, alpha=alpha)

    te_all=np.array(te_all)
    ne_all=np.array(ne_all)
    ax_plasma.fill_between(los, ne_all.min(0), ne_all.max(0), color=plasma_color, alpha=alpha)
    ax_te.fill_between(los, te_all.min(0), te_all.max(0), color=te_color, alpha=alpha)


    leg=ax_fit.legend()
    leg.draggable()

    if filename is None:
        fig.show()
    else:
        plt.tight_layout() # breaks the code
        plt.savefig(filename)
        plt.close()

def plot_fit_demo(posterior, sample, size=None, alpha=None, ylim=(1e10, 1e16),
             error_norm=True, plasma_ref=None, filename=None, parameterised=True):

    if size is None:
        size = len(sample)
    if alpha is None:
        alpha=1/size
    if alpha < 0.02:
        alpha = 0.02

    fig = plt.figure()
    ax_fit = fig.add_axes([.15, .57, .32, .39]) # [x0, y0, width, height]
    ax_res = fig.add_axes([.15, .125, .32, .39]) # [x0, y0, width, height]
    ax_plasma = fig.add_axes([.6, .125, .32, .825]) # [x0, y0, width, height]
    # ax_plasma = fig.add_axes([.6, .37, .32, .59]) # [x0, y0, width, height]
    # ax_plasma2 = fig.add_axes([.6, .125, .32, .19]) # [x0, y0, width, height]

    plasma_color = 'tab:red'
    te_color = 'tab:blue'
    ax_plasma.set_xlabel(r'$LOS \ / \ cm$')
    ax_plasma.set_ylabel(r'$n_{e} \ / \ cm^{-3}$', color=plasma_color)
    ax_plasma.tick_params(axis='y', labelcolor=plasma_color)
    ax_te = ax_plasma.twinx()  # instantiate a second axes that shares the same x-axis
    ax_te.set_ylabel(r'$T_{e} \ / \ eV$', color=te_color)  # we already handled the x-label with ax1
    ax_te.tick_params(axis='y', labelcolor=te_color)

    # ax_plasma2.set_xlabel(r'$LOS \ / \ cm$')
    # ax_plasma2.set_ylabel(r'$n_{e} \ / \ cm^{-3}$', color=plasma_color)
    # ax_plasma2.tick_params(axis='y', labelcolor=plasma_color)
    # ax_te2 = ax_plasma2.twinx()  # instantiate a second axes that shares the same x-axis
    # ax_te2.set_ylabel(r'$T_{e} \ / \ eV$', color=te_color)  # we already handled the x-label with ax1
    # ax_te2.tick_params(axis='y', labelcolor=te_color)

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

    los=posterior.plasma.los
    ax_plasma.set_xlim([min(los), max(los)])

    if not parameterised:
        los_ne_theta=posterior.plasma.profile_function.electron_density.x_points
        if posterior.plasma.profile_function.electron_density.zero_bounds:
            los_ne_theta=los_ne_theta[1:-1]
        los_te_theta=posterior.plasma.profile_function.electron_temperature.x_points
        if posterior.plasma.profile_function.electron_temperature.zero_bounds:
            los_te_theta=los_te_theta[1:-1]

    if plasma_ref is not None:
        te = plasma_ref['electron_temperature']
        ne = plasma_ref['electron_density']

        if 'electron_density_los' in plasma_ref:
            ax_plasma.plot(plasma_ref['electron_density_los'], ne, 'x--', color=plasma_color)
            # ax_plasma2.plot(plasma_ref['electron_density_los'], ne, 'x--', color=plasma_color)
        else:
            ax_plasma.plot(los, ne, 'x--', color=plasma_color)
            # ax_plasma2.plot(los, ne, 'x--', color=plasma_color)

        if 'electron_temperature_los' in plasma_ref:
            ax_te.plot(plasma_ref['electron_temperature_los'], te, 'x--', color=te_color)
            # ax_te2.plot(plasma_ref['electron_temperature_los'], te, 'x--', color=te_color)
        else:
            ax_te.plot(los, te, 'x--', color=te_color)
            # ax_te2.plot(los, te, 'x--', color=te_color)

    if error_norm:
        k_res = error
    else:
        k_res = spectra

    te_all=[]
    ne_all=[]
    for counter0 in np.linspace(0, len(sample)-1, size, dtype=int):
        posterior(sample[counter0])
        tmp_fit = posterior.posterior_components[0].forward_model()
        ax_fit.plot(waves, tmp_fit, 'pink', alpha=alpha)
        ax_res.plot(waves, abs(spectra-tmp_fit)/k_res, color='pink',
                    marker='x', alpha=alpha/3)

        te = posterior.plasma.plasma_state['electron_temperature']
        ne = posterior.plasma.plasma_state['electron_density']

        te_all.append(te)
        ne_all.append(ne)

        ax_plasma.plot(los, ne, color=plasma_color, alpha=alpha)
        ax_te.plot(los, te, color=te_color, alpha=alpha)

        if not parameterised:
            ne_theta = np.power(10, posterior.plasma.plasma_theta['electron_density'])
            te_theta = np.power(10, posterior.plasma.plasma_theta['electron_temperature'])
            ax_plasma.plot(los_ne_theta, ne_theta, 'o', color=plasma_color, alpha=alpha)
            ax_te.plot(los_te_theta, te_theta, 'o', color=te_color, alpha=alpha)
        # ax_plasma2.plot(los_ne_theta, ne_theta, 'o', color=plasma_color, alpha=alpha)
        # ax_te2.plot(los_te_theta, te_theta, 'o', color=te_color, alpha=alpha)

    te_all=np.array(te_all)
    ne_all=np.array(ne_all)
    ax_plasma.fill_between(los, ne_all.min(0), ne_all.max(0), color=plasma_color, alpha=alpha)
    ax_te.fill_between(los, te_all.min(0), te_all.max(0), color=te_color, alpha=alpha)


    leg=ax_fit.legend()
    leg.draggable()

    if filename is None:
        fig.show()
    else:
        plt.tight_layout() # breaks the code
        plt.savefig(filename)
        plt.close()
