import os, sys, io
import time as clock
from time import sleep
from copy import copy

import numpy as np
from numpy import random
from random import uniform

from scipy.optimize import fmin_l_bfgs_b

import concurrent.futures
from multiprocessing import Pool
# from baysar.priors import gaussian_low_pass_cost

from baysar.priors import CurvatureCost, AntiprotonCost, MainIonFractionCost
from baysar.priors import NeutralFractionCost, StaticElectronPressureCost, TauPrior
from baysar.priors import ChargeStateOrderPrior
from baysar.priors import WavelengthCalibrationPrior, CalibrationPrior, GaussianInstrumentFunctionPrior
from baysar.priors import gaussian_low_pass_cost, gaussian_high_pass_cost
from baysar.plasmas import PlasmaLine
from baysar.spectrometers import SpectrometerChord
from baysar.tools import within, progressbar, clip_data
from baysar.linemodels import XLine
from baysar.optimisation import evolutionary_gradient_ascent, sample_around_theta, optimise
from baysar.output_tools import plot_fit_demo


'''
# if you want to do profiling import profilehooks
from profilehooks import profile
# the above the relavent function (even in classes) place the following 'hook'
@profile
'''

def tmp_func(*args, **kwargs):
    return random.rand() * 1e1


class BaysarPosterior(object):

    """
    BaysarPosterior is a top level object within BaySAR which is passed an input dictionary (created by
    make_input_dict()) for the initialisation. During the initialisation the necessary plasma, spectrometer
    and line model objects are initialised and structure to all for the forward modeling of the spectra,
    calculation of the experimental errors and evaluation of the likelihood of a given/passed plasma state.
    """

    def __init__(self, input_dict, profile_function=None, priors=[], check_bounds=False,
                       temper=1., curvature=None, print_errors=False, skip_test=False):

        'input_dict input has been checked if created with make_input_dict'
        self.check_init_inputs(priors, check_bounds, temper, curvature, print_errors)

        print("Building BaySAR posterior object")

        self.input_dict = input_dict
        self.plasma = PlasmaLine(input_dict=input_dict, profile_function=profile_function)

        self.build_posterior_components()
        self.check_bounds = check_bounds
        self.print_errors = print_errors
        self.temper = temper

        self.passed_priors=priors
        self.curvature = curvature

        self.nan_thetas = []
        self.inf_thetas = []
        self.positive_thetas = []
        self.runtimes = []

        self.get_priors()
        self.set_sensible_bounds()

        self.init_test(skip_test)

    def get_priors(self):
        if self.curvature is not None:
            self.posterior_components.append(CurvatureCost(self.plasma, self.curvature))

        # calibration priors
        if self.plasma.calibrate_wavelength:
            inmean=self.posterior_components[0].x_data.mean()
            indisp=np.diff(self.posterior_components[0].x_data).mean()
            wcp_input=[self.plasma, [inmean, indisp], [2*indisp, indisp*0.05]]
            self.posterior_components.append(WavelengthCalibrationPrior(*wcp_input))
        if self.plasma.calibrate_intensity:
            self.posterior_components.append(CalibrationPrior(self.plasma))
        if self.plasma.calibrate_instrument_function:
            self.posterior_components.append(GaussianInstrumentFunctionPrior(self.plasma.plasma_state))

        # static Pe priors
        self.posterior_components.append(StaticElectronPressureCost(self.plasma))
        # neutral fraction priors
        if self.plasma.contains_hydrogen:
            n0_prior=NeutralFractionCost(self.plasma, species=self.plasma.hydrogen_species[0])
            self.posterior_components.append(n0_prior)
        # priors for impure plasmas (could use better phrasing)
        if len(self.plasma.impurities)>0:
            self.posterior_components.append(AntiprotonCost(self.plasma))
            self.posterior_components.append(MainIonFractionCost(self.plasma))
            self.posterior_components.append(ChargeStateOrderPrior(self))
            # self.posterior_components.append(TauPrior(self.plasma))

        self.posterior_components.extend(self.passed_priors)

    def check_init_inputs(self, priors, check_bounds, temper, curvature, print_errors):
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

    def init_test(self, skip_test):
        start=self.random_start()
        run_check=self(start, skip_test)
        if not (np.isreal(run_check) and -1e50 < run_check and run_check < 0):
            if skip_test:
                print(ValueError("Posterior is not evalauating correctly. logP =", run_check, 'from input=', start))
            else:
                raise ValueError("Posterior is not evalauating correctly. logP =", run_check, 'from input=', start)
        print("Posterior successfully created!") #instantiated

    def __call__(self, theta, skip_error=False):
        theta=list(theta)
        self.last_proposal=theta
        if not skip_error:
            self.check_call_input(theta, skip_error)
        # updating plasma state
        self.plasma(theta)

        # print('updated plasma')
        prob = sum( p() for p in self.posterior_components )

        if not skip_error:
            self.check_call_output(prob)
        # prob/=self.temper # temper default is 1
        self.last_proposal_logp=prob

        return prob

    def check_call_input(self, theta, skip_error):
        # check that theta doesn'contain NaNs
        if any(np.isnan(theta)):
            raise TypeError("Theta contains NaNs")
        # check that theta doesn't contain infs
        if any(np.isinf(theta)):
            raise TypeError("Theta contains infinities")

        # check that skip_error is a boulean
        if skip_error not in (True, False):
            raise TypeError("skip_error not in (True, False)")

    def check_call_output(self, prob):
        if np.isnan(prob):
            self.nan_thetas.append(self.last_proposal)
            raise ValueError('logP is NaNs')
        if np.isinf(prob):
            self.inf_thetas.append(self.last_proposal)
            raise ValueError('logP is inf')
        if prob >= 0.:
            self.positive_thetas.append(self.last_proposal)
            raise ValueError('lopP is positive')

    def cost(self, theta):
        return -self.__call__(theta)

    def build_posterior_components(self):
        self.posterior_components = []
        for chord_num, refine in enumerate(self.input_dict['refine']):
            chord = SpectrometerChord(plasma=self.plasma, refine=refine, chord_number=chord_num)
            self.posterior_components.append(chord)

    def set_sensible_bounds(self, dcwl=2, ddisp=0.005):
        import numpy as np
        # chordal bounds
        for chord_num in range(self.plasma.num_chords):
            tmp_chord=self.posterior_components[chord_num]
            # build new bounds
            # build cal bounds
            if tmp_chord.y_data.max() > 1e6:
                cal_bounds=[-0.3, 0.3]
            else:
                cal_bounds=[8, 15]

            calems_tag='cal_'+str(chord_num)
            if calems_tag in self.plasma.slices:
                self.plasma.theta_bounds[self.plasma.slices[calems_tag]]=cal_bounds
            else:
                ...

            # build wavelenth bounds
            estimate_cwl=tmp_chord.x_data.mean()
            cwl_bounds=[estimate_cwl-dcwl, estimate_cwl+dcwl]
            estimate_disp=np.diff( tmp_chord.x_data ).mean()
            disp_bounds=[estimate_disp-ddisp, estimate_disp+ddisp]

            calwave_tag='calwave_'+str(chord_num)
            if calwave_tag in self.plasma.slices:
                self.plasma.theta_bounds[self.plasma.slices[calwave_tag]][0]=cwl_bounds
                self.plasma.theta_bounds[self.plasma.slices[calwave_tag]][1]=disp_bounds
            else:
                ...

            # build background bounds
            estimate_background=tmp_chord.y_data_continuum.mean()
            d_background=tmp_chord.y_data_continuum.std()
            background_bounds=[estimate_background-d_background, estimate_background+d_background]
            background_limits=[estimate_background/10, estimate_background*10]
            background_bounds=[m(a, b) for m, a, b in zip([max, min], background_bounds, background_limits)]
            self.plasma.theta_bounds[self.plasma.slices['background_'+str(chord_num)]]=[np.log10(bb) for bb in background_bounds]

        # impurity density
        for s in self.plasma.slices:
            if s.endswith('_dens'):
                s_split=s.split('_')
                charge=float(s_split[-2])
                new_limit=self.plasma.theta_bounds[self.plasma.slices['electron_density']][0, 1]
                if not charge==0: # impurity ions
                    new_limit-=np.log10(charge)
                    nb_range=2
                else: # neutrals
                    new_limit-=0.7
                    nb_range=1
                self.plasma.theta_bounds[self.plasma.slices[s]][0]=[new_limit-nb_range, new_limit]
            if not self.plasma.thermalised:
                if s.endswith('_Ti'):
                    self.plasma.theta_bounds[self.plasma.slices[s]][0]=[0, 1.7]

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
            prandom_background=np.random.uniform(*(mean+half_width_range))
            start[self.plasma.slices['background_'+str(chord.chord_number)].start]=prandom_background
            # # improved start for mystery_lines
            # for xline in [line for line in chord.lines if type(line)==XLine]:
            #     estemate_ems=0
            #     half_width_wave=np.diff(chord.x_data)[0]*2
            #     for cwl, fraction in zip(xline.line.cwl, xline.line.fractions):
            #         _, y=clip_data(chord.x_data, chord.y_data, [cwl-half_width_wave, cwl+half_width_wave])
            #         estemate_ems+=sum(y*fraction)
            #     start[self.plasma.slices[xline.line_tag].start]=np.random.uniform(*(np.log10(estemate_ems)-1+1e1*half_width_range))
            #     # np.log10(estemate_ems)-1+np.random.normal(0, 0.1)
        return np.array(start)

    def _random_start(self, order=1, co=1, spectra=False, inform=True):
        seed=co*float(str(clock.time()).split('.')[-1])
        np.random.seed(int(seed))
        rs=[np.random.uniform(bounds[0], bounds[1], size=int(order)).mean() for bounds in self.plasma.theta_bounds]

        if inform:
            chords=[chord for chord in self.posterior_components if type(chord)==SpectrometerChord]
            for chord in chords:
                rs[self.plasma.slices['background_'+str(chord.chord_number)].start]=np.log10(np.mean(chord.y_data_continuum))
            # improved start for mystery_lines
            half_width_range=0.1
            half_width_range=np.array([-half_width_range, half_width_range])
            for xline in [line for line in chord.lines if type(line)==XLine]:
                estemate_ems=0
                half_width_wave=np.diff(chord.x_data)[0]*2
                for cwl, fraction in zip(xline.line.cwl, xline.line.fractions):
                    _, y=clip_data(chord.x_data, chord.y_data, [cwl-half_width_wave, cwl+half_width_wave])
                    estemate_ems+=sum(y*fraction)
                # rs[self.plasma.slices[xline.line_tag].start]=np.log10(estemate_ems)
                rs[self.plasma.slices[xline.line_tag].start]=np.random.uniform( *(np.log10(estemate_ems)-1+1e1*half_width_range) )

        if not all(self.plasma.is_theta_within_bounds(rs)):
            raise ValueError("Error in rs shape. {} should be {}.".format(rs.shape, self.plasma.n_params))

        out=[self(rs), rs]
        if spectra:
            out.append(self.posterior_components[0].forward_model())

        return out

    def _random_sample(self, number=1, order=1, flat=False, spectra=False, num_processes=1):
        number=int(number)
        seed_coefficients=np.random.uniform(0, 1, number)
        iterator=zip(np.zeros(number, dtype=int)+order, seed_coefficients, [spectra for i in range(number)])
        with Pool(processes=num_processes) as pool:
            sample=pool.starmap(self._random_start, iterator) # outputs a list of outputs

        sample=sorted(sample, key=lambda x:-x[0])
        if spectra:
            return [s[1:] for s in sample]
        else:
            return [s[1] for s in sample]

    def random_sample(self, number=1, order=1, flat=False):
        sample=[]
        for _ in progressbar(np.arange(number), 'Building random sample: ', 30):
            rstart=self.random_start(order, flat)
            sample.append((self(rstart), rstart))
        if not all(np.isreal([s[0] for s in sample])):
            raise ValueError("Blah")

        sample=sorted(sample, key=lambda x:-x[0])
        return [s[1] for s in sample]

    def sample_start(self, number=1, scale=100, order=1, flat=False, start_sample=None, return_info=False):
        sample=[]
        num_failed_grad_opt=0
        if start_sample is None:
            start_sample=self.random_sample(scale, order, flat)[:number]
        for rstart in progressbar(start_sample, 'Building sample: ', 30):
            # this try loop is not ideal
            # but is needed as parameters out of the bounds are sometimes being sampled and breaking the code
            # (its the taus)
            try:
                start, logp, info=fmin_l_bfgs_b(self.cost, rstart, approx_grad=True,
                                                bounds=self.plasma.theta_bounds.tolist(),
                                                maxfun=int(5e5))
                sample.append((-logp, start, info))
            except:
                num_failed_grad_opt+=1
                if num_failed_grad_opt==1:
                    self.broke_grad_opt=[self.last_proposal]
                else:
                    self.broke_grad_opt.append(self.last_proposal)

        if num_failed_grad_opt==number:
            raise ValueError("{} % of the gradient optimisation failed!".format(np.round(100.0, 1)))
        elif num_failed_grad_opt>0:
            pc=np.round(100*(1-num_failed_grad_opt/number), 2)
            self.grad_opt_performance='{} out of {} failed the gradient optimisation ({} % succeeded)'.format(num_failed_grad_opt, number, pc)
            print(self.grad_opt_performance)
        else:
            self.grad_opt_performance='{} % of the gradient optimisation succeeded!'.format(np.round(100.0, 1))

        sample=sorted(sample, key=lambda x:-x[0])

        if return_info:
            return sample
        else:
            return [s[1] for s in sample]

    def sample_around_theta(self, theta, num=11, pc=0.025, cal_pc=1e-3):
        cal_index=self.plasma.slices['background_0'].start
        s0=sample_around_theta(theta=theta, num=num, pc=pc, cal_pc=cal_pc, cal_index=cal_index)

        return s0

    def optimise(self, pop_size=12, num_eras=1, generations=3, threads=12, initial_population=None,
                 random_sample_size=int(1e3), random_order=3, perturbation=None, filename=None,
                 plot=False, plasma_reference=None, return_out=True, maxiter=100):

        if initial_population is None:
            initial_population=self.random_sample(number=random_sample_size, order=random_order)

        if perturbation is None:
            perturbation=np.logspace(-4, -2, 7)

        out, big_out=optimise(self, initial_population, pop_size=pop_size, num_eras=num_eras, generations=generations,
                              threads=threads, bounds=self.plasma.theta_bounds.tolist(), maxiter=maxiter,
                              cal_index=self.plasma.slices['background_0'].start, perturbation=perturbation, filename=filename)

        if plot:
            plot_fit_demo(self, [out['optimal_theta']], plasma_reference=plasma_reference, ylim=None)

        self.optimal_theta=out['optimal_theta']
        self.ega_out=out
        self.ega_big_out=big_out

        if return_out:
            return out, big_out


    def sample_start_p(self, number=1, scale=100, order=1, flat=False, start_sample=None, max_workers=None):
        sample=[]
        num_failed_grad_opt=0
        if start_sample is None:
            start_sample=self.random_sample(scale, order, flat)[:number]
        # for rstart in progressbar(start_sample, 'Building sample: ', 30):
        #     start, logp, info=fmin_l_bfgs_b(self.cost, rstart, approx_grad=True, bounds=self.plasma.theta_bounds.tolist())
        #     sample.append((-logp, start))
        # res = executor.map(lambda x,y:spam(x,params=y), urls, params)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            out=executor.map(lambda x, y: fmin_l_bfgs_b(x, y, approx_grad=True, bounds=self.plasma.theta_bounds.tolist()),
                             [self.sample_start for s in start_sample], start_sample)
            sample=concatenate([(-s[1], s[0]) for s in out])
        sample=sorted(sample, key=lambda x:-x[0])
        return [s[1] for s in sample]

    def differential_evolution(self, popsize=10, randsize=10000, gen_iter=10, mutation=(0.4, 0.9), tolerence=1.,
                               rand_order=3, flat=False, tolerence_gen=2, regression_gen=2, start_sample=None):
        # get intial population
        sample=self.sample_start(number=popsize, scale=randsize, order=rand_order, flat=flat, start_sample=start_sample) # , max_workers=4)
        gen_probs=[]
        gen_best=[]
        gen_best.append(sample[0])
        gen_probs.append(self.cost(gen_best[-1]))
        fail_evo_counter=0
        fail_tolerence_counter=0
        # evolve for gen_iter iterations
        tmp_format=(1, np.round(gen_probs[-1], 1))
        # print(f'differential evolution gen {} f(x)={}'.format(*tmp_format))
        print('differential evolution gen {} f(x)={}'.format(*tmp_format))
        for gen in np.arange(gen_iter):
        # for gen in progressbar(np.arange(gen_iter), 'Evolving Population!: ', 30):
            # mutate populations
            a=int( uniform(*(0, len(sample)-1)) )
            b=int( uniform(*(0, len(sample)-1)) )
            sample=[sample[0]+uniform(*mutation)*(sample[a]-sample[b]) for s in sample]
            # gradien optimse
            sample=self.sample_start(number=len(sample), start_sample=sample)
            # check that the evolution is progressive
            gen_best.append(sample)
            gen_probs.append(self.cost(gen_best[-1][0]))
            tmp_format=(gen+2, np.round(gen_probs[-1], 1))
            print('differential evolution gen {} f(x)={}'.format(*tmp_format))
            if gen_probs[-1] > gen_probs[-2]:
                fail_evo_counter+=1
                if fail_evo_counter < regression_gen:
                    tmp_format=(np.round(g, 2) for g in gen_probs[-2:])
                    UserWarning('Last evolution is regressive f(x)={}->{}!'.format(*tmp_format))
                else:
                    ValueError('Evolution is regressive for two generations returning the best found solution (f(X)={})'.format((min(gen_probs))))
                    return np.array(gen_best)[np.where([g==min(gen_probs) for g in gen_probs])][0]
            if abs(np.diff(gen_probs[-2:]))<tolerence:
                fail_tolerence_counter+=1
                if fail_tolerence_counter < tolerence_gen:
                    tmp_format=(tolerence, np.round(gen_probs[-2], 2), np.round(gen_probs[-1], 2))
                    UserWarning('Last evolution is was less than {}. f(x)={}->{}!'.format(*tmp_format))
                else:
                    ValueError('Evolutions was less then {} for two generations. Stopping evolution!'.format(tolerence))
                    return sample
        return sample

class Baysar2DWrapperPosterior:

    def __init__(self, baysar_posteriors, te_separatrix_err=1, pe_separatrix_err=1e13, pe_max=5e14, priors=None):
        self.chord_posteriors=baysar_posteriors

        self.posterior_components=[]
        self.posterior_components.extend(self.chord_posteriors)

        self.priors=[self.separatrix_prior]
        if priors is not None:
            self.priors.extend(*priors)

        self.te_separatrix_err=te_separatrix_err
        # self.ne_separatrix_err=ne_separatrix_err
        self.pe_separatrix_err=pe_separatrix_err
        self.pe_max=pe_max

        self.get_theta_bounds()
        self.get_separatrix_indicies()
        self.get_chord_slices()

    def __call__(self, theta):
        self.last_proposal=theta
        zip_list=[self.chord_posteriors, self.chord_slices]
        prob=sum(p(theta[t_slice]) for p, t_slice in zip(*zip_list))
        if len(self.priors) > 0:
            prob+=sum(p() for p in self.priors)
        return prob

    def cost(self, theta):
        return -self(theta)

    def random_start(self, order=1, flat_plasma=True):
        start=[chord.random_start(order=order) for chord in self.chord_posteriors]
        if flat_plasma:
            ne_profile=start[0][self.chord_posteriors[0].plasma.slices['electron_density']]
            te_profile=start[0][self.chord_posteriors[0].plasma.slices['electron_temperature']]
            for chord_num, chord in enumerate(self.chord_posteriors):
                tags=['electron_density', 'electron_temperature']
                for tag, profile in zip(tags, [ne_profile, te_profile]):
                    start[chord_num][chord.plasma.slices[tag]]=profile

        return np.array(start).flatten().tolist()

    def separatrix_prior(self):
        self.get_separatrix_values()

        # te must drop or stay constant
        self.te_prior_prob=sum([gaussian_low_pass_cost(d_te, 0, self.te_separatrix_err) for d_te in np.diff(self.te_separatrix)])

        # ne must be unimodal
        self.ne_prior_prob=0

        # pe must drop or stay constant
        self.pe_prior_prob=sum([gaussian_low_pass_cost(d_te, 0, self.pe_separatrix_err) for d_te in np.diff(self.pe_separatrix)])
        # max pe print_errors
        self.pe_max_prior_prob=gaussian_low_pass_cost(self.pe_separatrix[0], self.pe_max, self.pe_separatrix_err)
        # max pe drop
        self.pe_max_drop_prior_prob=gaussian_low_pass_cost(self.pe_separatrix[0]/self.pe_separatrix[0], 10., 1.)

        s_prior_probs=[self.te_prior_prob, self.ne_prior_prob,
                      self.pe_prior_prob, self.pe_max_prior_prob, self.pe_max_drop_prior_prob]

        return sum(s_prior_probs)

    def get_separatrix_values(self):
        self.te_separatrix=[np.power(10, te) for te in [self.last_proposal[i] for i in self.te_separatrix_indicies]]
        self.ne_separatrix=[np.power(10, ne) for ne in [self.last_proposal[i] for i in self.ne_separatrix_indicies]]
        self.pe_separatrix=[te*ne for ne, te in zip(self.ne_separatrix, self.te_separatrix)]


    def get_theta_bounds(self):
        self.theta_bounds=[]
        for chord in self.chord_posteriors:
            self.theta_bounds.extend(chord.plasma.theta_bounds.tolist())

    def get_separatrix_indicies(self):
        self.te_separatrix_indicies=[]
        self.ne_separatrix_indicies=[]
        shift=0
        for chord in self.chord_posteriors:
            self.te_separatrix_indicies.append(chord.plasma.slices['electron_temperature'].start+shift)
            self.ne_separatrix_indicies.append(chord.plasma.slices['electron_density'].start+shift)
            shift+=chord.plasma.n_params

    def get_chord_slices(self):
        self.chord_slices=[]
        start=0
        for chord in self.chord_posteriors:
            n_params=chord.plasma.n_params
            self.chord_slices.append(slice(start, start+n_params))


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

class PosteriorTanhTran:
    def __init__(self, posterior, bounds):
        """
        This need to be passed the posterior, posterior bounds
        """

        self.posterior=posterior
        self.bounds=np.array(bounds)
        self.bounds_widths=0.5*np.abs( np.diff(self.bounds).flatten() )
        self.bounds_centres=self.bounds.min(1)+self.bounds_widths

    def check_init(self):
        if any([b==0 for b in self.bounds_widths]):
            raise ValueError("Some of the upper and lower bounds are equal")

    def __call__(self, theta):
        # check that theta does not contain NaNs or infinities
        self.check_call_input(theta)
        # transform from the sigmoid space to the parameter
        theta=self.inverse_transform(theta)
        # return the posterior probability
        return self.posterior(theta)

    def check_call_input(self, theta):
        # check that theta doesn'contain NaNs
        if any(np.isnan(theta)):
            raise TypeError("theta contains NaNs")
        # check that theta doesn't contain infs
        if any(np.isinf(theta)):
            raise TypeError("theta contains infinities")

    def inverse_transform(self, theta):
        """
        going from the unbound/transformed space to the bound/desired space
        """
        k=self.bounds_widths
        c=self.bounds_centres
        return k*np.tanh(theta)+c

    def transform(self, theta):
        """
        going from the bound/desired space to the unbound/transformed space
        """
        k=self.bounds_widths
        c=self.bounds_centres
        return np.arctanh( (theta-c)/k )


def logistic(x, x0=0, k=1, L=1, c=0):
    """
    Implimentaion of the logistic sigmoid fucntion (https://en.wikipedia.org/wiki/Logistic_function).
    """
    return c + L / (1 + np.exp( -k*(x-x0) ))

def logistic_inverse(y, x0=0, k=1, L=1, c=0):
    """
    Implimentaion of the logistic sigmoid fucntion (https://en.wikipedia.org/wiki/Logistic_function).
    """
    z = L / (y - c) - 1
    return x0 - np.log(z)/k

def logistic_grad(x, x0=0, k=1, L=1, c=0, order=1):
    """
    Implimentaion of the logistic sigmoid fucntion (https://en.wikipedia.org/wiki/Logistic_function).
    Gradient of the transform - the validation of this formulation has only been checked upto order 2.
    """
    gradient_scaler_list=[[1-o*logistic(theta, x0, k, L)] for o in np.arange(order+1)]
    gradient_scaler=1
    for scalar in gradient_scaler_list:
        gradient_scaler*=scalar
    return gradient_scaler*logistic(theta, x0, k, L, c)

class PosteriorLogisticTran:
    def __init__(self, posterior, bounds):
        """
        This need to be passed the posterior, posterior bounds
        """

        self.posterior=posterior
        self.bounds=np.array(bounds)
        self.bounds_widths=np.abs( np.diff(self.bounds).flatten() )
        self.bounds_min=self.bounds.min(1)

    def check_init(self):
        if any([b==0 for b in self.bounds_widths]):
            raise ValueError("Some of the upper and lower bounds are equal")

    def __call__(self, theta):
        # check that theta does not contain NaNs or infinities
        self.check_call_input(theta)
        # transform from the sigmoid space to the parameter
        theta=self.inverse_transform(theta)
        # return the posterior probability
        return self.posterior(theta)

    def check_call_input(self, theta):
        # check that theta doesn'contain NaNs
        if any(np.isnan(theta)):
            raise TypeError("theta contains NaNs")
        # check that theta doesn't contain infs
        if any(np.isinf(theta)):
            raise TypeError("theta contains infinities")

    def inverse_transform(self, theta):
        """
        going from the unbound/transformed space to the bound/desired space
        """
        L=self.bounds_widths
        c=self.bounds_min
        return logistic_inverse(theta, x0=0, k=1, L=L, c=c)

    def transform(self, theta):
        """
        going from the bound/desired space to the unbound/transformed space
        """
        L=self.bounds_widths
        c=self.bounds_min
        return logistic_inverse(theta, x0=0, k=1, L=L, c=c)

    def grad_transform(self, theta, order=1):
        """
        gradient of the transform - the validation of this formulation has only been checked upto order
        """
        L=self.bounds_widths
        c=self.bounds_centres
        return logistic_grad(theta, x0=0, k=1, L=L, c=c, order=order)

    def transform_oddprimeprime(self, theta):
        """
        This is for hessien estimation
        """
        return 1-2*self.transform(theta)


from baysar.lineshapes import SimpleSeparatrix
from baysar.plasmas import PlasmaSimple2D
class BaysarSimple2DWrapper:
    def __init__(self, posteriors, chords, profile_function=None):
        if profile_function is None:
            self.profile_function=SimpleSeparatrix(chords)
        else:
            self.profile_function=profile_function

        self.priors=[]
        self.posteriors=posteriors
        self.plasma=PlasmaSimple2D(chords=chords, plasmas=[p.plasma for p in self.posteriors], profile_function=self.profile_function)

    def __call__(self, theta):
        # update 2d plasma
        self.plasma.update_plasma_state(theta)
        # need to go from 2D to a series of 1D thetas
        thetas=self.plasma.get_1d_thetas()
        # need to evaluate all the BaysarPosteriors
        logp=sum([p(thee) for p, thee in zip(self.posteriors, thetas)])
        # prior stuff?
        if len(self.priors) > 0:
            for p in self.priors:
                logp+=p()

        return logp

    def cost(self, theta):
        return -self(theta)


if __name__=='__main__':

    from baysar.input_functions import make_input_dict

    num_pixels=1024
    wavelength_axis = [np.linspace(3800, 4350, num_pixels)]
    experimental_emission = [np.zeros(num_pixels)+1e12+np.random.normal(0, 1e10, num_pixels)]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ['D', 'C', 'N']
    ions = [ ['0'] , ['1', '2'], ['1', '2', '3', '4'] ]
    noise_region = [[4040, 4050]]
    mystery_lines = [ [ [4070], [4001, 4002] ],
                      [    [1],    [0.4, 0.6]]]

    input_dict = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                                instrument_function=instrument_function, emission_constant=emission_constant,
                                noise_region=noise_region, species=species, ions=ions,
                                mystery_lines=mystery_lines, refine=[0.05],
                                ion_resolved_temperatures=False, ion_resolved_tau=True)

    from baysar.lineshapes import MeshPlasma, BowmanTeePlasma
    from baysar.plasmas import arb_obj

    x=np.linspace(-10, 35, 50)
    profile_function=BowmanTeePlasma(x=x, bounds=None, dr_bounds=[-5, 1], bounds_ne=[13, 15], bounds_te=[0, 1.7], background=False)
    posterior=BaysarPosterior(input_dict=input_dict, check_bounds=False, skip_test=False,
                                curvature=None, print_errors=False,
                                profile_function=profile_function)

    # posterior = BaysarPosterior(input_dict=input_dict, curvature=1e2)

    from tulasa.general import plot
    from tulasa.plotting_functions import plot_fit

    random_sample=posterior.random_sample(number=100, order=3, flat=True)
    chord = posterior.posterior_components[0]
    posterior(random_sample[-1])
    plot(chord.forward_model())
    print(chord.lines)
    posterior.plasma.impurity_tecs['N_2_4097.33_4103.43'][5]()
    # start_sample=posterior.sample_start(number=10, scale=1000, order=3, flat=True)
    # plot_fit(posterior, random_sample, alpha=0.3)
    # plot([posterior(t) for t in random_sample])



    pass
