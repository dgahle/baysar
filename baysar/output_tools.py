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

def plot_fit_demo(posterior, sample, size=None, alpha=None, ylim=(1e10, 1e16),
                  error_norm=True, plasma_reference=None, filename=None, parameterised=True,
                  only_band=True, sort_te=False):

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

    ax_res.plot(waves, np.ones(len(waves)), '--', color='k', alpha=0.5) # , alpha=0.1)
    # grids for residuals
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

    if plasma_reference is not None:
        te = plasma_reference['electron_temperature'].copy()
        ne = plasma_reference['electron_density'].copy()
        los_ref=plasma_reference['electron_density_los'].copy()
        # sort plasma profiles
        if sort_te:
            indices=np.argsort(te)
            ne=ne[indices][::-1]
            te=te[indices][::-1]
            los_ref-=76.

        if 'electron_density_los' in plasma_reference:
            ax_plasma.plot(los_ref, ne, 'x--', color=plasma_color)
            # ax_plasma2.plot(plasma_reference['electron_density_los'], ne, 'x--', color=plasma_color)
        else:
            ax_plasma.plot(los, ne, 'x--', color=plasma_color)
            # ax_plasma2.plot(los, ne, 'x--', color=plasma_color)

        if 'electron_temperature_los' in plasma_reference:
            ax_te.plot(los_ref, te, 'x--', color=te_color)
            # ax_te2.plot(plasma_reference['electron_temperature_los'], te, 'x--', color=te_color)
        else:
            ax_te.plot(los, te, 'x--', color=te_color)
            # ax_te2.plot(los, te, 'x--', color=te_color)

    if error_norm:
        k_res = error
    else:
        k_res = spectra

    fit_all=[]
    wave_all=[]
    res_all=[]
    te_all=[]
    ne_all=[]
    for counter0 in np.linspace(0, len(sample)-1, size, dtype=int):
        posterior(sample[counter0])
        tmp_fit = posterior.posterior_components[0].forward_model()
        tmp_wave = posterior.posterior_components[0].cal_wave
        tmp_res = abs(spectra-tmp_fit)/k_res
        te = posterior.plasma.plasma_state['electron_temperature']
        ne = posterior.plasma.plasma_state['electron_density']

        fit_all.append(tmp_fit)
        wave_all.append(tmp_wave)
        res_all.append(tmp_res)
        te_all.append(te)
        ne_all.append(ne)

        if not only_band: # =True
            ax_fit.plot(waves, tmp_fit, 'pink', alpha=alpha)
            ax_res.plot(waves, tmp_res, color='pink',
                        marker='x', alpha=alpha/3)

            ax_plasma.plot(los, ne, color=plasma_color, alpha=alpha)
            ax_te.plot(los, te, color=te_color, alpha=alpha)

            if not parameterised:
                ne_theta = np.power(10, posterior.plasma.plasma_theta['electron_density'])
                te_theta = np.power(10, posterior.plasma.plasma_theta['electron_temperature'])
                ax_plasma.plot(los_ne_theta, ne_theta, 'o', color=plasma_color, alpha=alpha)
                ax_te.plot(los_te_theta, te_theta, 'o', color=te_color, alpha=alpha)
            # ax_plasma2.plot(los_ne_theta, ne_theta, 'o', color=plasma_color, alpha=alpha)
            # ax_te2.plot(los_te_theta, te_theta, 'o', color=te_color, alpha=alpha)

    fit_all=np.array(fit_all)
    res_all=np.array(res_all)
    te_all=np.array(te_all)
    ne_all=np.array(ne_all)

    waves=waves.astype(np.float64)
    # plot the band of the spectra fits
    if only_band:
        alpha=1
        ax_fit.plot(waves, fit_all.mean(0), color='pink', alpha=alpha)

    a, b=fit_all.min(0), fit_all.max(0)
    ax_fit.fill_between(waves, a, b, color='pink', alpha=alpha)
    if only_band:
        alpha=0.6
    # plot the band of residuals
    a, b=res_all.min(0), res_all.max(0)
    ax_res.fill_between(waves, a, b, color='pink', alpha=alpha)

    # sort plasma profiles
    if sort_te:
        indices=np.argsort(te_all, axis=1)
        for counter, index in enumerate(indices):
            ne_all[counter]=ne_all[counter, index]
        te_all=np.sort(te_all, axis=1)
    # plot the band of the inferred plasma profiles
    ax_plasma.fill_between(los, ne_all.min(0), ne_all.max(0), color=plasma_color, alpha=alpha)
    ax_te.fill_between(los, te_all.min(0), te_all.max(0), color=te_color, alpha=alpha)


    leg=ax_fit.legend()
    leg.draggable()

    if filename is None:
        fig.show()
    else:
        # plt.tight_layout() # breaks the code
        plt.savefig(filename)
        plt.close()

    # return te_all, ne_all
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
plt.ion()

from adas import read_adf11, run_adas406

from baysar.plasmas import get_meta

def reaction_rate(posterior, sample, neutral='D_ADAS', reaction='scd'):
    if reaction=='scd':
        file='/home/adas/adas/adf11/scd12/scd12_h.dat'
    elif reaction=='acd':
        file='/home/adas/adas/adf11/acd12/acd12_h.dat'
    else:
        raise ValueError('reaction must be scd (ionisation) or acd (recombination)')

    rates=[]
    for s in sample:
        posterior(s)

        te=posterior.plasma.plasma_state['electron_temperature']
        ne=posterior.plasma.plasma_state['electron_density']
        if reaction=='scd':
            reactant=float(posterior.plasma.plasma_state['D_ADAS_0_dens'])
        else:
            reactant=posterior.plasma.plasma_state['main_ion_density']

        rate=read_adf11(te=te, dens=ne, adf11type=reaction, file=file, is1=1)

        rates.append(reactant*ne*rate)

    return rates

def plot_sources_and_sinks(posterior, sample, log=False, data=None, chord=None, neutral='D_ADAS'):
    los=posterior.plasma.los
    plt.figure()

    rates=['acd', 'scd']
    labels=[r'$n_{i}n_{e}ACD$', r'$n_{0}n_{e}SCD$']
    colours=['blue', 'red']
    for rate, label, colour in zip(rates, labels, colours):
        rate=np.array( reaction_rate(posterior, sample, neutral, reaction=rate) )
        plt.fill_between(los, rate.min(0), rate.max(0), label=label, color=colour, alpha=0.5)
        plt.plot(los, rate.mean(0), color=colour)

    plt.xlim(los.min(), los.max())

    if log:
        plt.yscale('log')
        plt.ylim(1e12)

    plt.ylabel(r'$rate \ / \ cm^{-3}s^{-1}$')
    plt.xlabel(r'$LOS \ / \ cm$')

    leg=plt.legend()
    leg.draggable()

    plt.show()

def plot_impurtity_profiles_dense(posterior, sample, data, chord, alpha=0.3):

    elem='N'
    meta=get_meta(elem)
    ions=[1, 2, 3]
    species=['N_1', 'N_2', 'N_3']

    fig, ax = plt.subplots(1, 2, sharey=True)

    ax_profile, ax_balance = ax

    ax_profile.set_title(r'$Line \ of \ Sight \ Profiles$')
    ax_balance.set_title(r'$Ionisation \ Balance$')

    # reference SOLPS things
    solps_los=data.get('d_los')[::-1, chord]*100
    solps_los-=solps_los[np.argmax(data.get('Te_los')[:, chord])]
    solps_te=data.get('Te_los')[:, chord]
    for charge, az in enumerate(data.get('nz_los')[:, chord, :].T*1e-6):
        if (charge in ions): #  or (charge==max(ions)+1):
            ax_profile.plot(solps_los, az, 'x--', color='C'+str(charge))
            label='N'+str(charge)+r'$+ \ (SOLPS)$'
            ax_balance.plot(solps_te, az, 'x--', label=label, color='C'+str(charge))

    # formatting
    ax_profile.set_yscale('log')
    ax_profile.set_ylim(1e10, 1e13)

    ax_profile.set_ylabel(r'$n \ / \ cm^{-3}$')
    ax_profile.set_xlabel(r'$LOS \ / \ cm$')

    ax_balance.set_xlim(0, solps_te.max()+1)

    ax_balance.set_xlabel(r'$T_{e} \ / \ eV$')

    # inference stuff
    densities={}
    for i, s in zip(ions, species):
        densities[s]={i:[], i+1:[]}

    te_profiles=[]
    los_baysar=posterior.plasma.los
    for s in sample:
        posterior(s)
        tmp_te=posterior.plasma.plasma_state['electron_temperature']
        tmp_ne=posterior.plasma.plasma_state['electron_density']
        te_profiles.append(tmp_te)
        for i, s in zip(ions, species):
            tmp_tau=float(posterior.plasma.plasma_state[s+'_tau'])
            a_out, pow = run_adas406(year=96, elem=elem, te=tmp_te, dens=tmp_ne, tint=tmp_tau, meta=meta)
            # densities[s]
            tmp_dens=float(posterior.plasma.plasma_state[s+'_dens'])
            tmp_bal=a_out['ion']*tmp_dens
            for charge in [i, i+1]:
                tmp_indicies=np.argsort(tmp_te)
                # ax_balance.plot(tmp_te[tmp_indicies], tmp_bal[tmp_indicies, i], 'C'+str(i) , alpha=alpha)
                # ax_balance.plot(tmp_te, tmp_bal[:, i+1], 'C'+str(i+1), alpha=alpha)
                # ax_profile.plot(los_baysar, tmp_bal[:, i], 'C'+str(i))
                densities[s][charge].append(tmp_bal[:, charge])


    los=posterior.plasma.los
    te_profiles=np.array(te_profiles)
    te_grid=np.linspace(1., te_profiles.max(), 20)
    for i, s in zip(ions, species):
        # for charge, reaction in zip([i, i+1], [r'$N_{exc}$', r'$N_{rec}$']):
        for charge, reaction in zip([i], [r'$N_{exc}$']):
            colour='C'+str(charge)
            profiles=np.array( densities[s][charge] )

            if charge in (i, max(ions)+1):
                label=r'$N$'+str(charge)+r'$+ \ (BaySAR)$'
            else:
                label=None

            ax_profile.fill_between(los, profiles.min(0), profiles.max(0),
                             color=colour, alpha=alpha)
            ax_profile.plot(los, profiles.mean(0), label=label, color=colour)

            grided_profiles=[]
            for te, prof in zip(te_profiles, profiles):
                tmp_interp=interp1d(te, prof, bounds_error=False, fill_value='extrapolate')
                grided_profiles.append( tmp_interp(te_grid) )

            # ax_balance.fill_between(te_grid, np.array(grided_profiles).min(0), np.array(grided_profiles).max(0), 'C'+str(i), alpha=alpha)
            ax_balance.fill_between(te_grid, np.array(grided_profiles).min(0), np.array(grided_profiles).max(0), color='C'+str(charge), alpha=alpha)
            ax_balance.plot(te_grid, np.array(grided_profiles).mean(0), 'C'+str(charge))

    leg_bal=ax_balance.legend()
    leg_bal.draggable()
    leg_prof=ax_profile.legend()
    leg_prof.draggable()
    ax_profile.set_xlim(los.min(), los.max())

    fig.show()
