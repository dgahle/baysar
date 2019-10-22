import os, sys, io
import numpy as np
from numpy import random
import time as clock
from scipy.optimize import fmin_l_bfgs_b

from baysar.plasmas import PlasmaLine
from baysar.spectrometers import SpectrometerChord, within, progressbar, clip_data
from baysar.linemodels import XLine


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

        self.los=plasma.profile_function.electron_density.x_points
        self.empty_array=plasma.profile_function.electron_density.empty_theta
        if plasma.profile_function.electron_density.zero_bounds is not None:
            self.slice=slice(1, -1)
        else:
            self.slice=slice(0, len(self.empty_array))
    def __call__(self):
        curves=0
        for tag in ['electron_density']: # , 'electron_temperature']:
            self.empty_array[self.slice]=self.plasma.plasma_theta.get(tag)
            curves+=curvature(self.empty_array)
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

        print("Building BaySAR posterior object")

        self.input_dict = input_dict
        self.plasma = PlasmaLine(input_dict=input_dict, profile_function=profile_function)

        self.build_posterior_components()
        self.check_bounds = check_bounds
        self.print_errors = print_errors
        self.temper = temper

        self.curvature = curvature
        if self.curvature is not None:
            self.posterior_components.append(CurvatureCost(self.plasma, self.curvature))
        self.posterior_components.append(AntiprotonCost(self.plasma))
        self.posterior_components.extend(priors)

        self.nan_thetas = []
        self.inf_thetas = []
        self.positive_thetas = []
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
        if not np.isreal(temper):
            raise TypeError("temper must be a real number")
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
        theta=list(theta)
        self.last_proposal=theta
        # updating plasma state
        self.plasma(theta)
        prob = sum( p() for p in self.posterior_components )
        self.check_output(prob)
        return prob/self.temper # temper default is 1

    def check_output(self, prob):
        if np.isnan(prob):
            self.nan_thetas.append(self.last_proposal)
            raise ValueError('logP is NaNs')
        if np.isinf(prob):
            self.inf_thetas.append(self.last_proposal)
            raise ValueError('logP is inf')
        if prob >= 0:
            self.positive_thetas.append(self.last_proposal)
            raise ValueError('lopP is positive')

    def cost(self, theta):
        return -self.__call__(theta)

    def build_posterior_components(self):
        self.posterior_components = []
        for chord_num, refine in enumerate(self.input_dict['refine']):
            chord = SpectrometerChord(plasma=self.plasma, refine=refine, chord_number=chord_num)
            self.posterior_components.append(chord)

    def random_start(self, order=1, flat=False):
        start = [np.mean(np.random.uniform(bounds[0], bounds[1], size=order)) for bounds in self.plasma.theta_bounds]
        # produce flat plasma profiles
        if flat:
            for param in ['electron_density', 'electron_temperature']:
                for index in np.arange(self.plasma.slices[param].start, self.plasma.slices[param].stop):
                    start[index]=start[self.plasma.slices[param]][0]
        # improved start for the background
        chords=[chord for chord in self.posterior_components if type(chord)==SpectrometerChord]
        half_width_range=0.1
        half_width_range=np.array([-half_width_range, half_width_range])
        for chord in chords:
            mean=np.log10(np.mean(chord.y_data_continuum)) # std=np.std(chord.y_data_continuum)/10
            start[self.plasma.slices['background'+str(chord.chord_number)].start]=np.random.uniform(*(mean+half_width_range))
            # improved start for mystery_lines
            for xline in [line for line in chord.lines if type(line)==XLine]:
                estemate_ems=0
                half_width_wave=np.diff(chord.x_data)[0]*2
                for cwl, fraction in zip(xline.line.cwl, xline.line.fractions):
                    _, y=clip_data(chord.x_data, chord.y_data, [cwl-half_width_wave, cwl+half_width_wave])
                    estemate_ems+=sum(y*fraction)
                start[self.plasma.slices[xline.line_tag].start]=np.random.uniform(*(np.log10(estemate_ems)-1+1e1*half_width_range))
                # np.log10(estemate_ems)-1+np.random.normal(0, 0.1)
        return np.array(start)

    def random_sample(self, number=1, order=1, flat=False):
        sample=[]
        for rstart in progressbar(np.arange(number), 'Building random sample: ', 30):
            sample.append(self.random_start(order, flat))
        # sample_and_prob=[]
        # for s in progressbar(sample, 'Sorting from high to low probability: ', 30):
        #     sample_and_prob.append((self.cost(s), s))
        print('Sorting random sample from high to low probability')
        return sorted(sample, key=lambda x:self.cost(x))

    def sample_start(self, number, scale=1, order=1, flat=False):
        sample=[]
        random_sample=self.random_sample(number*scale, order, flat)[:number]
        for rstart in progressbar(random_sample, 'Building starting sample: ', 30):
            start, logp, info=fmin_l_bfgs_b(self.cost, rstart, approx_grad=True, bounds=self.plasma.theta_bounds.tolist())
            sample.append((start, logp))
        return [s[0].tolist() for s in sorted(sample, key=lambda x:x[1])]


from copy import copy

# Todo: add a check input function
class BaysarPosteriorFilterWrapper(object):

    """
    BaysarPosteriorFilterWrapper is a wrapper for BaysarPosterior (and inherits that object) which allows
    for the sampling of a subset of parameters that compose of the full parameters of the forward model.
    """

    def __init__(self, posterior, reference, indicies=None, tags=None):
        self.posterior=posterior
        self.reference=reference
        if tags is None:
            self.filter_indices=list(indicies)
        else:
            self.filter_indices_from_tags(tags)

        reduce_indices=set(self.filter_indices).symmetric_difference( set(np.arange(len(self.reference)).tolist()) )
        self.reduced_indices=list(reduce_indices)

    def __call__(self, theta):
        return self.posterior(self.expand_theta(theta))

    def cost(self, theta):
        return -self.__call__(theta)

    def reduce_theta(self, theta):
        reduced_theta=[]
        for index in self.reduced_indices:
            reduced_theta.append(theta[index])
        return reduced_theta

    def expand_theta(self, reduced_theta):
        theta=copy(self.reference)
        for index, param in zip(self.reduced_indices, reduced_theta):
            theta[index]=param
        return theta

    def filter_indices_from_tags(self, tags):
        slices=[self.posterior.plasma.slices[t] for t in tags]
        self.filter_indices=np.concatenate([np.arange(s.start, s.stop) for s in slices]).tolist()



if __name__=='__main__':

    from baysar.input_functions import make_input_dict

    num_pixels=512
    wavelength_axis = [np.linspace(3900, 4150, num_pixels)]
    experimental_emission = [np.zeros(num_pixels)+1e12+np.random.normal(0, 1e10, num_pixels)]
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
                                mystery_lines=mystery_lines, refine=[0.25],
                                ion_resolved_temperatures=False, ion_resolved_tau=True)


    posterior = BaysarPosterior(input_dict=input_dict, curvature=1e2)

    from tulasa.general import plot
    from tulasa.plotting_functions import plot_fit

    # random_sample=posterior.random_sample(number=1000, order=3, flat=True)
    # # start_sample=posterior.sample_start(number=10, scale=1000, order=3, flat=True)
    # plot_fit(posterior, random_sample, alpha=0.3)
    # plot([posterior(t) for t in random_sample])

    for n in np.arange(100):
        posterior(posterior.random_start())

    pass
