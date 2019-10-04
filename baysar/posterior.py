import os, sys, io
import numpy as np
from numpy import random
import time as clock
from scipy.optimize import fmin_l_bfgs_b

from baysar.plasmas import PlasmaLine
from baysar.spectrometers import SpectrometerChord, within, progressbar


def tmp_func(*args, **kwargs):
    return random.rand() * 1e1

def curvature(profile):
    grad1 = np.gradient(profile / max(profile))
    grad2 = np.gradient(grad1)
    return - sum(np.square(grad2) / np.power((1 + np.square(grad1)), 3)) # curvature


class CurvatureCost(object):
    def __init__(self, plasma, scale):
        self.plasma=plasma
        self.scale=scale
    def __call__(self):
        curves=0
        for p in ['electron_density', 'electron_temperature']:
            curves+=curvature(self.plasma.plasma_state[p])
        return curves*self.scale

class AntiprotonCost(object):
    def __init__(self, plasma):
        self.plasma=plasma
    def __call__(self):
        if any(self.plasma.plasma_state['main_ion_density'] < 0):
            anti_profile=self.plasma.plasma_state['main_ion_density'].clip(max=-1)
            return -np.log10(abs(anti_profile)).sum()
        else:
            return 0.


class BaysarPosterior(object):

    """
    BaysarPosterior is a top level object within BaySAR which is passed an input dictionary (created by
    make_input_dict()) for the initialisation. During the initialisation the necessary plasma, spectrometer
    and line model objects are initialised and structure to all for the forward modeling of the spectra,
    calculation of the experimental errors and evaluation of the likelihood of a given/passed plasma state.
    """

    def __init__(self, input_dict, profile_function=None, priors=[], check_bounds=False,
                       temper=1, curvature=None, print_errors=False, skip_test=False):

        'input_dict input has been checked if created with make_input_dict'

        self.check_inputs(priors, check_bounds, temper, curvature, print_errors)



        self.input_dict = input_dict
        self.plasma = PlasmaLine(input_dict=input_dict, profile_function=profile_function)

        self.build_posterior_components()
        self.check_bounds = check_bounds
        self.print_errors = print_errors
        self.temper = temper

        self.priors = priors
        self.curvature = curvature
        if self.curvature is not None:
            self.priors.append(CurvatureCost(self.plasma, self.curvature))
        self.priors.append(AntiprotonCost(self.plasma))

        self.posterior_components.extend(self.priors)

        self.nan_thetas = []
        self.runtimes = []

        start=self.random_start()
        run_check=self(start)
        if not (np.isreal(run_check) and -1e50 < run_check and run_check < 0):
            if skip_test:
                ValueError("Posterior is not evalauating correctly. logP =", run_check, 'from input=', start)
            else:
                raise ValueError("Posterior is not evalauating correctly. logP =", run_check, 'from input=', start)

        print("Posterior successfully created!") #instantiated

    def check_inputs(self, priors, check_bounds, temper, curvature, print_errors):

        if len(priors)>0:
            if type(priors) is not list:
                raise TypeError("If priors are being passed then it must be a list of functions")

        if type(check_bounds) is not bool:
            raise TypeError("check_bounds must be True or False and is False by default")

        if type(temper) not in (int, float, np.int64, np.float64):
            raise TypeError("temper must be an int or fa_calloat")
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
        # updating plasma state
        self.plasma(theta)
        prob = [p() for p in self.posterior_components]
        prob = sum(prob)
        self.check_output(prob)
        return prob/self.temper # temper default is 1

    def check_output(self, prob):
        if np.isnan(prob):
            raise ValueError('logP = NaNs')
        if np.isinf(prob):
            raise ValueError('logP = NaNs')
        if prob >= 0:
            raise ValueError('lopP is positive')

    def cost(self, theta):
        return -self.__call__(theta)

    def build_posterior_components(self):
        self.posterior_components = []
        for chord_num, refine in enumerate(self.input_dict['refine']):
            chord = SpectrometerChord(plasma=self.plasma, refine=refine, chord_number=chord_num)
            self.posterior_components.append(chord)

    def start_sample_old(self, num, min_logp=-1e10, high_prob=False, order=1, flat=False):
        if high_prob:
            old_num = num
            num = int(num * 10)

        sb_start = []
        for tmp in np.arange(num):
            tmp_logp = -1e50
            while tmp_logp < min_logp:
                tmp_start = self.random_start(order=order, flat=flat)
                tmp_logp = self(tmp_start)
            sb_start.append(tmp_start)

        sb_start = sorted(sb_start, key=self)
        if high_prob:
            sb_start = sb_start[-old_num:]

        return np.array(sb_start[::-1])

    def sample_start(self, number, order=1, flat=False):
        sample=[]
        for i in progressbar(np.arange(number), 'Building starting sample: ', 30):
            start, logp, info=fmin_l_bfgs_b(self.cost, self.random_start(order, flat), approx_grad=True, bounds=self.plasma.theta_bounds.tolist())
            sample.append((start, logp))
        return [s[0].tolist() for s in sorted(sample, key=lambda x:x[1])]

    def random_start(self, order=1, flat=False):
        start = [np.mean(np.random.uniform(bounds[0], bounds[1], size=order)) for bounds in self.plasma.theta_bounds]
        if flat:
            for param in ['electron_density', 'electron_temperature']:
                for index in np.arange(self.plasma.slices[param].start, self.plasma.slices[param].stop):
                    start[index] = start[self.plasma.slices[param]][0]

        for chord in [chord for chord in self.posterior_components if type(chord)==SpectrometerChord]:
            mean=np.mean(chord.y_data_continuum)
            std=np.std(chord.y_data_continuum)/10
            start[self.plasma.slices['background'+str(chord.chord_number)]][0]=np.log10(np.random.normal(mean, std))
        return np.array(start)



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
    experimental_emission = [np.zeros(len(wavelength_axis[0]))+1e12*np.random.normal(0, 1e10)]
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


    posterior = BaysarPosterior(input_dict=input_dict, curvature=1e2)

    rand_theta=posterior.random_start()
    start_sample=posterior.sample_start(number=3, order=1, flat=False)
    print(posterior(rand_theta))
    for start in start_sample:
        print(posterior(start), start)

    # from tulasa.general import plot
    # from tulasa.plotting_functions import plot_fit
    #
    # sample_num = 20
    # sample = posterior.start_sample(sample_num, min_logp=-500)
    #
    # plot([posterior(s) for s in sample])
    # plot_fit(posterior, sample, size=int(sample_num/2), alpha=0.1, ylim=(1e10, 1e16),
    #          error_norm=True, plasma_ref=None)
    pass
