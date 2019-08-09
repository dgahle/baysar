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

        self.input_dict = input_dict

        self.plasma = PlasmaLine(input_dict=input_dict, profile_fucntion=profile_function)

        self.posterior_components = self.build_posterior_components()
        self.priors = priors
        self.check_bounds = check_bounds

        self.temper = temper
        self.print_errors = print_errors
        self.curvature = curvature

        self.nan_thetas = []

        self.runtimes = []

    def __call__(self, theta, skip_error=True):

        theta = list(theta)

        self.last_proposal = theta

        if self.check_bounds:

            if not all(self.plasma.is_theta_within_bounds(theta)):
                prob = -1e50
                if self.print_errors:
                    print('Out of Bounds')
                return prob

        self.plasma.update_plasma(theta)

        try:
            if self.priors is None:
                prob = sum(p(theta) for p in self.posterior_components)
            else:
                prob = sum(p(theta) for p in self.posterior_components)

                prob += sum(p(self) for p in self.priors)
                       # sum(p(self.plasma.plasma_state) for p in self.priors)
        except:
            if self.print_errors:
                print('Error in evaluating posterior')
            prob = -1e50
            if skip_error:
                return prob
            else:
                raise

        if any(self.plasma.plasma_state['main_ion_density'] < 0):
            if self.print_errors:
                print('contains antiprotons')
            prob -= sum(np.log10(abs(self.plasma.plasma_state['main_ion_density'].clip(max=0))).clip(0))
            # return -1e50

        if self.curvature is not None:
            te_cost = self.curvature_cost(self.plasma.plasma_state['electron_temperature'])
            ne_cost = self.curvature_cost(self.plasma.plasma_state['electron_density'])

            prob -= self.curvature * (te_cost + ne_cost)


        if np.isnan(prob):

            self.nan_thetas.append(theta)

            if self.print_errors:
                print('logP = NaNs')

            return -1e50
        elif type(prob) == np.float64:
            return prob / self.temper
        else:
            return -1e50

    def curvature_cost(self, profile):
    # def curvature(profile):

        grad1 = np.gradient(profile / max(profile))
        grad2 = np.gradient(grad1)

        return sum(np.square(grad2) / np.power((1 + np.square(grad1)), 3))

    def cost(self, theta):

        return - self.__call__(theta)

    def build_posterior_components(self):

        components = []

        for chord_num in self.input_dict['chords'].keys():

            chord = SpectrometerChord(plasma=self.plasma, chord_number=chord_num)

            components.append(chord)

            pass

        return components

    def stormbreaker_start(self, num, min_logp=-1e10, high_prob=False, normal=False):

        sb_start = []

        if high_prob:
            tmp_num = int(num * 10)
        else:
            tmp_num = num

        for tmp in np.arange(tmp_num):

            tmp_logp = -1e50

            while tmp_logp < min_logp:

                tmp_start = self.random_start(normal=normal)

                tmp_logp = self(tmp_start)

            sb_start.append(tmp_start)

        sb_start = sorted(sb_start, key=self)

        # sb_start = np.array(sb_start).T

        if high_prob:
            sb_start = sb_start[-num:]

        self.sb_start = np.array(sb_start)

        return self.sb_start

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

    input_dict = {}
    num_chords = 3

    input_dict['number_of_chords'] = num_chords

    input_dict['chords'] = {}

    from BaySAR.BaySAR.lineshapes import Gaussian, GaussianNorm

    instrument_function = GaussianNorm(cwl=15, x=np.arange(31))
    instrument_function = instrument_function([10, 1])

    a_cal = 1e11

    continuum = [3960, 3990]
    x_data = np.arange(3950, 4150, 0.18)
    y_data = np.zeros( len( x_data ) )

    for tmp in np.arange( len(y_data) ):
        y_data[tmp] = a_cal + random.rand() * a_cal

    wavelength_axes = [x_data]


    experimental_emission = [y_data]

    instrument_function = [instrument_function]
    emission_constant = [a_cal]
    noise_region = [[3960, 3990]]

    for counter0, chord in enumerate(np.arange(num_chords)):

        # tmp = 'chord' + str(counter0)
        tmp = counter0
        counter0=0

        input_dict['chords'][tmp] = {}
        input_dict['chords'][tmp]['data'] = {}

        input_dict['chords'][tmp]['data']['wavelength_axis'] = wavelength_axes[counter0]
        input_dict['chords'][tmp]['data']['experimental_emission'] = experimental_emission[counter0]

        input_dict['chords'][tmp]['data']['instrument_function'] = instrument_function[counter0]
        input_dict['chords'][tmp]['data']['emission_constant'] = emission_constant[counter0]
        input_dict['chords'][tmp]['data']['noise_region'] = noise_region[counter0]

        pass

    '''
    n_ii_cwls = [3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92]
    n_ii_jjr = [1, 0.92, 0.08, 0.456, 0.211, 0.197, 0.026, 0.022]
    # n_ii_pec_keys = [0, 2, 2, 3, 3, 3, 3, 3]
    n_ii_pec_keys = [3, 6, 6, 7, 7, 7, 7, 7]
    
    # n_iii_cwls = [3998.63, 4003.58, 4097.33, 4103.34]
    # n_iii_jjr = [0.375, 0.625, 0.665, 0.335]
    # n_iii_pec_keys = [1, 1, 4, 4]    
    '''

    chord0_dict = {}

    species = ['D', 'N']
    ions = [['0'], ['1', '2']]

    cwl = [[[3968.99, 4100.58]],
           [[3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92],
            [3998.63, 4003.58, 4097.33, 4103.34]]]
    n_pec = [[[0,   0]],
             [[0,      0,      0,      0,      0,      0,      0,      0      ],
              [0,      0,      0,      0]]]  # TODO need the real values
    f_jj = [[[3968.99, 4100.58]],
            [[3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92],
             [3998.63, 4003.58, 4097.33, 4103.34]]]  # TODO need the real values

    atomic_charge = [1, 7]
    ma = [2, 14]

    for counter0, isotope in enumerate(species):

        # if counter0 == 0:
        #     input_dict['isotopes'] = species

        chord0_dict[isotope] = {}
        chord0_dict[isotope]['atomic_mass'] = ma[counter0]
        chord0_dict[isotope]['atomic_charge'] = atomic_charge[counter0]

        for counter1, ion in enumerate(ions[counter0]):

            if counter1 == 0:
                chord0_dict[isotope]['ions'] = ions[counter0]

            chord0_dict[isotope][ion] = {}

            for counter2, line in enumerate(cwl[counter0][counter1]):

                if counter2 == 0:
                    chord0_dict[isotope][ion]['lines'] = []

                chord0_dict[isotope][ion]['lines'].append(str(line))

                chord0_dict[isotope][ion][str(line)] = {}

                chord0_dict[isotope][ion][str(line)]['wavelength'] = line
                chord0_dict[isotope][ion][str(line)]['pec_key'] = n_pec[counter0][counter1][counter2]
                chord0_dict[isotope][ion][str(line)]['jj_frac'] = f_jj[counter0][counter1][counter2]

                chord0_dict[isotope][ion][str(line)]['tec'] = tmp_func
                chord0_dict[isotope][ion][str(line)]['tec_file'] = None

                pass

            pass

        pass

    input_dict['physics'] = chord0_dict



    # if __name__=='__main__':

    posterior = BaysarPosterior(input_dict=input_dict)

    theta = [1e12,
             -5, 30, 1e13,
             5, 10,
             0.8, 0.2,
             1e-4, 1e-4,
             5, 10]

    print(posterior(theta))

    pass