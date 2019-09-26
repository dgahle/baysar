import numpy as np
from numpy import random
import time as clock

import os, sys, io

from baysar.plasmas import PlasmaLine
from baysar.spectrometers import SpectrometerChord, within


def tmp_func(*args, **kwargs):
    return random.rand() * 1e1

class BaysarPosterior(object):

    """
    BaysarPosterior is a top level object within BaySAR which is passed an input dictionary (created by
    make_input_dict()) for the initialisation. During the initialisation the necessary plasma, spectrometer
    and line model objects are initialised and structure to all for the forward modeling of the spectra,
    calculation of the experimental errors and evaluation of the likelihood of a given/passed plasma state.
    """

    def __init__(self, input_dict, profile_function=None, priors=None, check_bounds=False,
                       temper=1, curvature=None, print_errors=False):

        'input_dict input has been checked if created with make_input_dict'

        self.check_inputs(priors, check_bounds, temper, curvature, print_errors)

        self.input_dict = input_dict

        self.plasma = PlasmaLine(input_dict=input_dict, profile_function=profile_function)

        self.build_posterior_components()
        self.priors = priors
        self.check_bounds = check_bounds

        self.temper = temper
        self.print_errors = print_errors
        self.curvature = curvature

        self.nan_thetas = []
        self.runtimes = []

        run_check = self(self.random_start())
        if not (-1e50 < run_check and run_check < 0):
            raise ValueError("Posterior is not evalauating correctly")

        print("Posterior successfully created!") #instantiated

    def check_inputs(self, priors, check_bounds, temper, curvature, print_errors):

        if priors is not None:
            if type(priors) is not list:
                raise TypeError("If priors are being passed then it must be a list of functions")

        if type(check_bounds) is not bool:
            raise TypeError("check_bounds must be True or False and is False by default")

        if type(temper) not in (int, float, np.int64, np.float64):
            raise TypeError("temper must be an int or float")
        elif temper < 0:
            raise ValueError("temper must be positive (should, but doesn't have to, be greater than 1)")
        else:
            pass

        if curvature is not None:
            if type(curvature) not in (int, float, np.int64, np.float64):
                raise TypeError("curvature must be an int or float")
            elif curvature < 1:
                raise ValueError("curvature must be greater than 1")

        if type(print_errors) is not bool:
            raise TypeError("print_errors must be True or False and is False by default")

    def __call__(self, theta, skip_error=True):

        theta = list(theta)
        self.last_proposal = theta
        if self.check_bounds:
            if not all(self.plasma.is_theta_within_bounds(theta)):
                prob = -1e50
                if self.print_errors:
                    print('Out of Bounds')
                return prob

        self.plasma(theta)
        prob = sum(p() for p in self.posterior_components)
        if self.priors is not None:
            prob += sum(p(self) for p in self.priors)

        if any(self.plasma.plasma_state['main_ion_density'] < 0):
            if self.print_errors:
                print('contains antiprotons')
            prob += sum(np.log10( abs(self.plasma.plasma_state['main_ion_density'].clip(max=-1e-20)) ))

        if self.curvature is not None:
            te_cost = self.curvature_cost(self.plasma.plasma_state['electron_temperature'])
            ne_cost = self.curvature_cost(self.plasma.plasma_state['electron_density'])

            prob -= self.curvature * (te_cost + ne_cost)

        if np.isnan(prob):
            self.nan_thetas.append(theta)
            if self.print_errors:
                print('logP = NaNs')
            return -1e50
        elif type(prob) in (np.float64, np.int64):
            return prob / self.temper
        else:
            return -1e50

    def curvature_cost(self, profile):
        grad1 = np.gradient(profile / max(profile))
        grad2 = np.gradient(grad1)
        curvature = sum(np.square(grad2) / np.power((1 + np.square(grad1)), 3))

        if curvature < 0:
            return 0
        else:
            return curvature

    def cost(self, theta):
        return - self.__call__(theta)

    def build_posterior_components(self):
        self.posterior_components = []
        for chord_num, refine in enumerate(self.input_dict['refine']):
            chord = SpectrometerChord(plasma=self.plasma, refine=refine, chord_number=chord_num)
            self.posterior_components.append(chord)

    def start_sample(self, num, min_logp=-1e10, high_prob=False, normal=False):
        if high_prob:
            old_num = num
            num = int(num * 10)

        sb_start = []
        for tmp in np.arange(num):
            tmp_logp = -1e50
            while tmp_logp < min_logp:
                tmp_start = self.random_start(normal=normal)
                tmp_logp = self(tmp_start)
            sb_start.append(tmp_start)

        sb_start = sorted(sb_start, key=self)
        if high_prob:
            sb_start = sb_start[-old_num:]

        return np.array(sb_start[::-1])

    def random_start(self, normal=False):

        start = []

        for bounds in self.plasma.theta_bounds:

            if not normal:
                start.append( np.random.uniform(min(bounds), max(bounds)) )
            else:
                sigma = abs(min(bounds) - max(bounds)) / 2
                nu = min(bounds) + sigma

                start.append(np.random.normal(nu, sigma/3))

        return np.array(start)

    # # TODO: This needs to be generalised and have flags so is optional
    # # TODO: Need to add a prior for main ion density
    # def prior_checks(self, theta):
    #
    #     self.plasma.update_plasma(theta)
    #
    #     total_impurity = 0
    #
    #     for tmp_element in list(self.plasma.input_dict.keys()):
    #
    #         if any( [tmp_element==t for t in ('D', 'X')] ):
    #             pass
    #         else:
    #             total_impurity += self.plasma.plasma_state[tmp_element]['conc']
    #
    #     if total_impurity > ( 0.5 * max(self.plasma.plasma_state['electron_density']) ):
    #
    #         print('total_impurity > ne / 2 ', total_impurity / 1e14, '>',
    #               0.5 * max(self.plasma.plasma_state['electron_density']))
    #
    #         return False
    #
    #     else:
    #
    #         return True
    #
    # # TODO: Needs tidying
    # def line_ratio(posterior, upper, lower, res=10, magical_tau=1):
    #
    #     upper_line = posterior.posterior_components[upper[0]].lines[upper[1]].tec406
    #     lower_line = posterior.posterior_components[lower[0]].lines[lower[1]].tec406
    #
    #     ratios = []
    #
    #     tau, ne = np.zeros(res) + magical_tau, np.linspace(1e13, 1e14, res)# np.logspace(12, 14, res)
    #
    #     long_te = np.linspace(2, 10, 5)
    #
    #     for tmp_te in long_te:
    #         te = np.zeros(res) + tmp_te
    #
    #         tec_in = np.array([tau, ne, te]).T
    #
    #         ratios.append( np.nan_to_num(upper_line(tec_in) / lower_line(tec_in)) )
    #
    #     fig, ax = plt.subplots(1, 1)
    #
    #     for counter, ratio in enumerate(ratios):
    #
    #         l = str(long_te[counter]) + r'$\ / \ eV$'
    #
    #         ax.plot(ne*1e6, ratio, label=l)
    #
    #     ax.set_xlim([1e19, 1e20])
    #
    #     ylabel = r'$N \ II: \ 462/399 \ nm \ | \ \tau \ = $' + \
    #              str(magical_tau*1e3) + r'$\ / \ ms$'
    #     ax.set_ylabel(ylabel)
    #     ax.set_xlabel(r'$n_{e} \ / \ m^{-3}$')
    #
    #     leg = []
    #     leg.append(ax.legend())
    #
    #     for l in leg:
    #         l.draggable()
    #
    #     fig.show()



# TODO: Write wrapper posteriorS to be able to do low dimentional fits - produce with a demo
# Todo: update to use with new BaysarPosterior
# Todo: add a check input function
class BaysarPosteriorFilterWrapper(BaysarPosterior):

    """
    BaysarPosteriorFilterWrapper is a wrapper for BaysarPosterior (and inherits that object) which allows
    for the sampling of a subset of parameters that compose of the full parameters of the forward model.
    """

    def __init__(self, input_dict, reference, indicies, profile_function=None, check_bounds=False):

        super().__init__(input_dict, profile_function=profile_function, check_bounds=False)

        self.reference = reference
        self.indicies = indicies

        self.calc_reduced_theta_bounds()
        self.check_reduced_bounds = check_bounds

    def __call__(self, theta, print_ref_new=False):

        if self.check_reduced_bounds:
            if not all(self.is_theta_within_bounds(theta)):
                prob = -1e50
                # print('Out of Bounds')
            else:
                prob = self.call(theta, print_ref_new=False)
        else:
            prob = self.call(theta, print_ref_new=False)

        return prob

    def call(self, theta, print_ref_new=False):

        new_theta = self.reference

        for counter, new_theta_index in enumerate(self.indicies):

            new_theta[new_theta_index] = theta[counter]

        if print_ref_new:
            print( theta )
            print( self.reference )
            print( new_theta )

        self.last_proposal = theta
        self.last_proposal_long = new_theta

        return super().__call__(new_theta)

    def negative_call(self, theta):

        return - self.__call__(theta)

    def calc_reduced_theta_bounds(self):

        self.reduced_theta_bounds = []
        self.reduced_theta_widths = []

        self.plasma.reduced_default_start = []

        for new_theta_index in self.indicies:

            self.reduced_theta_bounds.append(self.plasma.theta_bounds[new_theta_index])
            self.reduced_theta_widths.append(self.plasma.theta_widths[new_theta_index])

            self.plasma.reduced_default_start.append(self.plasma.default_start[new_theta_index])

    def is_theta_within_bounds(self, theta):

        out = []

        for counter, bound in enumerate(self.reduced_theta_bounds):
            out.append(within(theta[counter], bound))

        return out

    def random_start(self, normal=False):

        start = []

        for bounds in self.reduced_theta_bounds:

            if not normal:
                start.append( np.random.uniform(min(bounds), max(bounds)) )
            else:
                sigma = abs(min(bounds) - max(bounds)) / 2
                nu = min(bounds) + sigma

                start.append(np.random.normal(nu, sigma/3))

        return start

    # Building input dict



if __name__=='__main__':

    from baysar.input_functions import make_input_dict

    wavelength_axis = [np.linspace(3900, 4150, 512)]
    experimental_emission = [np.array([1e12*np.random.rand() for w in wavelength_axis[0]])]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ['D', 'N']
    ions = [ ['0'], ['1', '2', '3'] ]
    noise_region = [[4040, 4050]]
    mystery_lines = [ [ [4070], [4001, 4002] ],
                      [    [1],    [0.4, 0.6]]]

    input_dict = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                                instrument_function=instrument_function, emission_constant=emission_constant,
                                noise_region=noise_region, species=species, ions=ions,
                                mystery_lines=mystery_lines, refine=[0.05],
                                ion_resolved_temperatures=False, ion_resolved_tau=True)


    posterior = BaysarPosterior(input_dict=input_dict)

    rand_theta = posterior.random_start()
    print(posterior(rand_theta))

    from tulasa.general import plot
    from tulasa.plotting_functions import plot_fit

    sample_num = 20
    sample = posterior.start_sample(sample_num, min_logp=-500)

    plot([posterior(s) for s in sample])
    plot_fit(posterior, sample, size=int(sample_num/2), alpha=0.1, ylim=(1e10, 1e16),
             error_norm=True, plasma_ref=None)
    pass