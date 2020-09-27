
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from numpy import array, arange, zeros, ones, linspace, argsort, where, isfinite
from numpy import sqrt, exp, dot
from numpy import load, ceil, savez
from numpy.random import normal, random, choice, randint, seed
from scipy.optimize import fmin_l_bfgs_b
from multiprocessing import Process, Pipe, Event
from itertools import chain

from copy import copy
from time import time

from numpy import square, sqrt, mean, linspace, nan, log, diff, arange, zeros, concatenate, where

from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import sparse
import numpy as np
import time as clock

import os, sys, io, warnings


class LogWrapper(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, theta):
        return self.func(exp(theta))

    def gradient(self, theta):
        p = exp(theta)
        return p*self.func.gradient(p)

    def cost(self, theta):
        return -self.func(exp(theta))

    def cost_gradient(self, theta):
        p = exp(theta)
        return -p*self.func.gradient(p)




def bfgs_worker(posterior, bounds, maxiter, maxfun, connection, end):
    # main loop persists until shutdown event is triggered
    while not end.is_set():
        # poll the pipe until there is something to read
        while not end.is_set():
            if connection.poll(timeout = 0.05):
                theta_list = connection.recv()
                break

        # if read loop was broken because of shutdown event
        # then break the main loop as well
        if end.is_set(): break

        # run L-BFGS-B for each starting position in theta_list
        results = []
        for theta0 in theta_list:
            if hasattr(posterior, 'cost_gradient'):
                x_opt, fmin, D = fmin_l_bfgs_b(posterior.cost, theta0, fprime=posterior.cost_gradient, bounds = bounds, maxiter=maxiter, maxfun=maxfun)
            else:
                x_opt, fmin, D = fmin_l_bfgs_b(posterior.cost, theta0, bounds = bounds, approx_grad = True, maxiter=maxiter, maxfun=maxfun)

            # store results in a dictionary
            result = {
                'probability': -fmin,
                'solution' : x_opt,
                'flag' : D['warnflag'],
                'evaluations' : D['funcalls'],
                'iterations' : D['nit']
            }

            results.append(result)

        # send the results
        connection.send(results)



def evolutionary_gradient_ascent(posterior=None, initial_population = None, generations=20, threads=1,
                                 perturbation=0.075, mutation_probability = 0.5, bounds = None,
                                 maxiter = 1000, maxfun= int(3e4)):

    # initialise all the processes
    shutdown_evt = Event()
    processes = []
    connections = []
    for i in range(threads):
        parent_ctn, child_ctn = Pipe()
        p = Process(target=bfgs_worker, args=(posterior, bounds, maxiter, maxfun, child_ctn, shutdown_evt))
        p.start()
        processes.append(p)
        connections.append(parent_ctn)

    # set up the genetic algorithm
    pop = Population(n_params = len(initial_population[0]),
                     n_pop = len(initial_population),
                     perturbation = perturbation,
                     mutation_probability = mutation_probability,
                     bounds = bounds)

    # initialise the first generation
    pop.children = initial_population

    # create command-line arguments by dividing up population between threads
    batchsize = int(ceil(pop.popsize / threads))
    L = [i for i in range(pop.popsize)]
    index_groups = [L[i:i + batchsize] for i in range(0, pop.popsize, batchsize)]


    times = []
    iteration_history = []
    evaluation_history = []
    flag_history = []
    for iteration in range(generations):
        # chop up the population for each thread
        theta_groups = [[pop.children[i] for i in inds] for inds in index_groups]

        t1 = time()
        # send the starting positions to each process
        for starts, ctn in zip(theta_groups, connections):
            ctn.send(starts)

        # wait until results from all processes are received
        results = [ ctn.recv() for ctn in connections ]
        # join the results from each process into one list
        results = [ r for r in chain(*results) ]
        t2 = time()

        # unpack the results
        solutions = [ D['solution'] for D in results ]
        probabilities = [ D['probability'] for D in results ]
        gen_iterations = [ D['iterations'] for D in results ]
        gen_evaluations = [ D['evaluations'] for D in results ]
        gen_flags = [ D['flag'] for D in results ]

        iteration_history.append(gen_iterations)
        evaluation_history.append(gen_evaluations)
        flag_history.append(gen_flags)

        pop.give_fitness(probabilities, solutions)
        t3 = time()

        times.append((t1,t2,t3))
        msg =  '\n # generation ' + str(iteration)
        msg += '\n # best prob: ' + str(pop.elite_fitnesses[-1])
        m, s = divmod(t3 - t1, 60)
        h, m = divmod(m, 60)
        time_left = "%d:%02d:%02d" % (h, m, s)
        msg += '\n # time taken this gen: {} ({:1.1f}% in ascent)'.format( time_left, 100*(t2-t1)/(t3-t1))
        print(msg)

    # trigger the shutdown event and terminate the processes
    shutdown_evt.set()
    for p in processes: p.join()

    # build the results dictionary:
    result = {
        'generations' : generations,
        'population_size' : pop.popsize,
        'threads_used' : threads,
        'perturbation_size' : perturbation,
        'solutions' : solutions,
        'mutation_probability' : mutation_probability,
        'optimal_theta' : pop.elite_adults[-1],
        'max_log_prob' : pop.elite_fitnesses[-1],
        'log_prob_history' : pop.best_fitness_history,
        'solution_history' : pop.best_adult_history,
        'optimisation_times' : [ b-a for a,b,c in times ],
        'evolution_times' : [ c-b for a,b,c in times ],
        'generation_times' : [ c-a for a,b,c in times ],
        'population_log_probs' : pop.fitnesses,
        'iteration_history' : iteration_history,
        'evaluation_history' : evaluation_history,
        'flag_history' : flag_history
    }

    return result




def unzip(v):
    return [[x[i] for x in v] for i in range(len(v[0]))]

class Population(object):
    """
    Population management and generation class for evolutionary algorithms.

    :param initial_population:
    :param initial_fitnesses:
    :param scale_lengths:
    :param perturbation:
    """
    def __init__(self, n_params = None, n_pop = 20, scale_lengths = None,
                 perturbation = 0.1, mutation_probability = None, bounds = None, record_ancestors = True):

        if scale_lengths is None:
            self.L = 1
        else:
            self.L = array(scale_lengths)

        # unpack bounds into two arrays
        if bounds is not None:
            self.lwr_bounds = array([b[0] for b in bounds])
            self.upr_bounds = array([b[1] for b in bounds])
        else:
            self.lwr_bounds = None
            self.upr_bounds = None

        # settings
        self.n = n_params
        self.popsize = n_pop
        self.p = 6./self.popsize
        self.mutation_prob = mutation_probability
        self.mutation_str = perturbation
        self.fitness_only = False
        self.record_ancestors = record_ancestors

        # storage
        self.fitnesses = []
        self.ancestors = []
        self.children = []

        self.n_elites = 3
        self.elite_adults = []
        self.elite_fitnesses = []

        self.breeding_adults = []
        self.breeding_fitnesses = []

        self.best_adult_history = []
        self.best_fitness_history = []

    def get_genes(self):
        return self.adults

    def give_fitness(self, fitnesses, adults):
        # include elites in population
        adults.extend(self.elite_adults)
        fitnesses.extend(self.elite_fitnesses)

        # sort the population by fitness
        fitnesses, adults = unzip(sorted(zip(fitnesses, adults), key=lambda x : x[0]))

        # update elites
        self.elite_adults = adults[-self.n_elites:]
        self.elite_fitnesses = fitnesses[-self.n_elites:]

        # results of this generation
        self.best_adult_history.append(self.elite_adults[-1])
        self.best_fitness_history.append(self.elite_fitnesses[-1])
        self.fitnesses.append(fitnesses)
        if self.record_ancestors: self.ancestors.append(copy(adults))

        # specify the breeding population
        self.adults = adults
        self.adult_ranks = self.rank_prob(fitnesses)

        # breed until population is of correct size
        self.children.clear()
        while len(self.children) < self.popsize:
            self.breed()

    def breed(self):
        # pick one of the two breeding methods
        if random() < self.mutation_prob:
            self.mutation()
        else:
            self.crossover()

    def rank_prob(self, data):
        ranks = argsort(-1*array(data))
        probs = array([ self.p*(1 - self.p)**r for r in ranks ])
        return probs / sum(probs)

    def mutation(self):
        if hasattr(self.mutation_str, '__len__' ):
            strength = choice(self.mutation_str)
        else:
            strength = self.mutation_str

        child = self.select_member() + (strength * self.L) * normal(size=self.n)

        # apply bounds if they are specified
        if self.lwr_bounds is not None:
            lwr_bools = child < self.lwr_bounds
            upr_bools = child > self.lwr_bounds

            if any(lwr_bools):
                lwr_inds = where(lwr_bools)
                child[lwr_inds] = self.lwr_bounds[lwr_inds]

            if any(upr_bools):
                upr_inds = where(upr_bools)
                child[upr_inds] = self.lwr_bounds[upr_inds]
        self.children.append(child)

    def crossover(self):
        g1, g2 = self.select_pair()

        # create a random ordered list containing
        booles = zeros(self.n)
        booles[0::2] = True
        booles[1::2] = False
        booles = booles[ argsort(random(size=self.n)) ]

        child = []
        for i in range(self.n):
            if booles[i]:
                child.append(g1[i])
            else:
                child.append(g2[i])
        self.children.append(child)

    def select_member(self):
        weights = self.get_weights()
        i = choice(len(self.adults), p = weights)
        return self.adults[i]

    def select_pair(self):
        weights = self.get_weights()
        i, j = choice(len(self.adults), p=weights, size=2, replace=False)
        return self.adults[i], self.adults[j]

    def get_weights(self):
        if (len(self.children) is 0) or self.fitness_only:
            weights = self.adult_ranks
        else:
            weights = self.adult_ranks * self.rank_prob(self.get_diversity())
            weights /= weights.sum()
        return weights

    def get_diversity(self):
        div = []
        for adult in self.adults:
            dist = sum([ self.distance(adult, child) for child in self.children ])
            div.append(dist)
        return div

    def distance(self, g1, g2):
        z = (array(g1) - array(g2)) / self.L
        return sqrt(dot(z,z))




class DiffEv(object):
    def __init__(self, posterior = None, initial_population = None, initial_probabilities = None,
                 differential_weight = 0.3, crossover_prob = 0.85, show_status = True):

        # settings
        self.F = differential_weight
        self.CR = crossover_prob
        self.max_tries = 200
        self.show_status = show_status

        self.N = len(initial_population)
        self.L = len(initial_population[0])

        self.posterior = posterior
        self.population = initial_population
        self.probabilities = initial_probabilities
        self.max_prob = max(self.probabilities)

    def run_for(self, seconds = 0, minutes = 0, hours = 0, days = 0):
        # first find the runtime in seconds:
        run_time = ((days*24. + hours)*60. + minutes)*60. + seconds
        start_time = time()
        end_time = start_time + run_time

        while time() < end_time:
            self.take_step()

            # display the progress status message
            if self.show_status:
                seconds_remaining = end_time - time()
                m, s = divmod(seconds_remaining, 60)
                h, m = divmod(m, 60)
                time_left = "%d:%02d:%02d" % (h, m, s)
                msg = '\r   [ best probability: {}, time remaining: {} ]'.format(self.max_prob, time_left)
                sys.stdout.write(msg)
                sys.stdout.flush()

        # this is a little ugly...
        if self.show_status:
            sys.stdout.write('\r   [ best probability: {}, run complete ]           '.format(self.max_prob))
            sys.stdout.flush()
            sys.stdout.write('\n')

    def take_step(self):
        for i in range(self.max_tries):
            flag, P = self.breed()
            if flag:
                if P > self.max_prob: self.max_prob = P
                break

    def breed(self):
        ix, ia, ib, ic = self.pick_indices()
        X, A, B, C = self.population[ix], self.population[ia], self.population[ib], self.population[ic]
        inds = where(random(size=self.L) < self.CR)
        Y = copy(X)
        Y[inds] = A[inds] + self.F*(B[inds] - C[inds])
        P_Y = self.posterior(Y)

        if P_Y > self.probabilities[ix]:
            self.population[ix] = Y
            self.probabilities[ix] = P_Y
            return True, P_Y
        else:
            return False, P_Y

    def pick_indices(self):
        return choice(self.N, 4, replace=False)

    def get_dict(self):
        items = [
            ('F', self.F),
            ('CR', self.CR),
            ('population', self.population),
            ('probabilities', self.probabilities),
            ('max_prob', self.max_prob) ]

        D = {} # build the dict
        for key, value in items:
            D[key] = value
        return D

    def save(self, filename):
        D = self.get_dict()
        savez(filename, **D)

    @classmethod
    def load(cls, filename, posterior = None):
        data = load(filename)
        pop = data['population']
        C = cls(posterior = posterior,
                initial_population = [ pop[i,:] for i in range(pop.shape[0])],
                initial_probabilities = data['probabilities'],
                differential_weight = data['F'],
                crossover_prob = data['CR'])
        return C




def diff_evo_worker(proc_seed, initial_population, initial_probabilities, posterior, hyperpars, runtime, pipe):
    seed(proc_seed)
    DE = DiffEv(posterior = posterior,
                initial_population = initial_population,
                initial_probabilities = initial_probabilities,
                differential_weight = hyperpars[0],
                crossover_prob = hyperpars[1],
                show_status = False)

    DE.run_for(seconds = runtime)
    max_ind = array(DE.probabilities).argmax()
    pipe.send( [DE.probabilities[max_ind], DE.population[max_ind]] )




def parallel_evolution(initial_population = None, popsize = 30, posterior = None, threads = None, runtime = None):

    L = len(initial_population)
    if L < popsize: popsize = L
    initial_probabilities = [ posterior(t) for t in initial_population ]

    # randomly sample the hyperparameters
    cross_probs = random(size = threads)*0.5 + 0.5
    diff_weights = random(size = threads)*1.2 + 0.1
    hyperpars = [ k for k in zip(cross_probs, diff_weights) ]

    # generate random seeds for each process
    seeds = randint(3, high=100, size=threads).cumsum()

    # loop to generate the separate processes
    processes = []
    connections = []
    for proc_seed, hyprs in zip(seeds, hyperpars):
        # create a pipe to communicate with the process
        parent_ctn, child_ctn = Pipe()
        connections.append(parent_ctn)

        # sub-sample the population
        inds = choice(L, popsize, replace=False)
        proc_pop = [initial_population[k] for k in inds]
        proc_probs = [initial_probabilities[k] for k in inds]

        # collect all the arguments
        args = [ proc_seed, proc_pop, proc_probs,
                 posterior, hyprs, runtime, child_ctn ]

        # create the process
        p = Process(target = diff_evo_worker, args = args)
        p.start()
        processes.append(p)

    # wait for results to come in from each process
    results = [ c.recv() for c in connections ]
    # terminate the processes now we have the results
    for p in processes: p.join()
    # unpack the results
    probabilities, parameters = unzip(results)

    return parameters, probabilities, cross_probs, diff_weights


def sample_around_theta(theta, num=11, pc=0.025, cal_pc=1e-4, cal_index=4):
    theta=array(theta)
    n_params=len(theta)
    calcorrect=ones(n_params)
    if cal_index is not None:
        calcorrect[:cal_index]*=cal_pc
    # produce a 2D grid around theta (+-pc*calcorrect)
    s0_ordered=theta[None, :]*(1-linspace(-pc, pc, num)[:, None]*calcorrect)
    # randomise the combination of the the parameters
    s0_order=[]
    for param in s0_ordered.T:
        s0_order.append(choice(param, len(param), replace=False))
    s0_order=list( array(s0_order).T )
    # make sure that theta is in the returned sample
    s0=[theta]
    for s in s0_order:
        s0.append(s)

    return s0

def optimise(posterior, initial_population, pop_size=12, num_eras=3, generations=3,
             threads=12, bounds=None, cal_index=None, perturbation=0.075, filename=None,
             maxiter=1000, mutation_probability=0.7):

    big_out=[]
    print(' # gen s0 best prob: ', posterior(initial_population[0]))
    s0=copy(initial_population)
    indicies=arange(len(s0[0]))
    for era in arange(num_eras):
        print()
        print('# era {}.'.format(era))
        # further informative print statements are produced by evolutionary_gradient_ascent
        out=evolutionary_gradient_ascent(posterior=posterior, initial_population=s0[:pop_size], generations=generations,
                                         threads=threads, maxiter=maxiter, perturbation=perturbation,
                                         mutation_probability=mutation_probability, bounds=bounds)
        if era < (num_eras-1):
            s0=sample_around_theta(out['optimal_theta'], num=int(1e3), pc=0.1, cal_index=cal_index)
            while len(s0) < pop_size:
                random_index=choice(indicies)
                s0.append(initial_population[random_index])
            s1=sorted(s0, key=posterior.cost)
            s0=[out['optimal_theta']]
            s0.extend(s1)

        big_out.append(out)

    if filename is not None:
        savez(filename, **out)

    print()

    return out, big_out

def gradient_optimisation(function, start, max_iterations=10, stepsize=None, k_step=2, tolerence=0.1,
                          line_tolerence=1., res=10, epsilon=1e-4, callback=False, line_callback=False):
    answers=[start]
    meta_data=['stating postion. no meta data']
    # begin d_prob=inf to initate while loop
    d_prob=inf
    # record number of gradient line searches
    iter=0
    # iterate gradient line optimisation until a threshold/tolerence is reached
    while abs(d_prob)>tolerence:
        if iter>1:
            stepsize=meta_data[-1]['distance'][argmax(meta_data[-1]['probs'])]*0.3
        # run gradient_line_search
        ans, meta=gradient_line_search(function, start, stepsize=stepsize, k_step=k_step, tolerence=line_tolerence,
                                       res=res, epsilon=epsilon, callback=line_callback)
        # record solution of gradient_line_search
        meta_data.append(meta)
        answers.append(ans)
        # record improvement in solution
        d_prob=function(start)-meta_data[-1]['probs'][-1]
        # update stating position
        start=ans
        # update number of iterations
        iter+=1
        # print progress
        if callback:
            print('Line search iterations {}'.format(iter))
            print(ans, meta)
        # exit if the number of max iterations has been reached
        if iter > max_iterations:
            print(UserWarning('Reached max number of iteration in binary search ({})'.format(max_iterations)))
            break

        # exit if the solution has converged
        if (iter>3) and all(ans==answers[-3]):
            break

    return ans, answers, meta_data

def check_start_type(start):
    # check that theta doesn'contain NaNs
    if any(np.isnan(start)):
        raise TypeError("Start contains NaNs")
    # check that theta doesn't contain infs
    if any(np.isinf(start)):
        raise TypeError("Start contains infinities")
    # check that theta doesn't contain 0s
    if any([s==0 for s in start]):
        raise TypeError("Start contains 0s. This with break the finite difference calcualtion!")

def gradient_line_search(function, start, stepsize=None, k_step=2, tolerence=1.,
                         res=10, epsilon=0.01, max_iter=4, callback=False):
    # make sure the start is a numpy array
    start=array([start]).flatten()
    # check that start doesn't conatain NaNs, inf or 0s
    check_start_type(start)
    # record prob of starting position
    origin=start.copy()
    probs=[function(start)]
    # estimate gradient
    grad=approx_fprime(start, function, start*epsilon)
    # check that the gradient does not contain NaNs
    if any(np.isnan(grad)):
        print("grad={}, epsilon={}, start={}".format(grad, epsilon, start))
        raise TypeError("k_grad contains NaNs")
    # normalise to make a unit vector
    k_grad=normalise(grad)
    grad/=k_grad
    # recod the stepsizes
    distance=[0] # stepsize]
    if stepsize is None:
        stepsize=.1*k_grad
    # find the peak
    d_prob=1
    iter=1
    while d_prob > 0:
        print('Iteration {} (Last logP = {})'.format(iter, probs[-1]))
        # take step along the line
        start=start+grad*stepsize
        # record prob of new postion
        probs.append(function(start))
        # upadte d_prob
        d_prob=probs[-1]-probs[-2]
        # recod the stepsizes
        distance.append(distance[-1]+stepsize)
        # # what is going on?!
        # print(probs[-1], distance[-1], stepsize, d_prob)
        # print(probs)
        # update stepsize
        stepsize*=k_step
        # print progress
        if callback:
            print('Line search step {}: {}'.format(iter, probs[-1]), flush=True)
        # record the number of iterations
        iter+=1
        tmp_max_iter=50
        if len(probs)>tmp_max_iter:
            raise ValueError('gradient_line_search never found a peak | max iter = {}'.format(tmp_max_iter))

        # print('Iteration {} (Last logP = {})'.format(iter, probs[-1]))

    # if the probility drops immediately then search within intervul unitil
    # the maxima is not at the edges (0 or len(probs))
    if len(distance)==2:
        # resort distances and probs
        indices=argsort(distance)
        distance=array(distance)[indices].tolist()
        probs=array(probs)[indices].tolist()
        # check that the maxima is not at the edges (0 or len(probs))
        while probs[0]>probs[1]:
            print('Iteration {} (Last logP = {})'.format(iter, probs[-1]))
        # while argmax(probs)==0 or argmax(probs)==len(probs):
            # calculate new point
            d=distance[0]+(distance[1]-distance[0])*.5
            # take step along the line
            start=origin+grad*d
            # record prob of new postion
            probs.append(function(start))
            # recod the stepsizes
            distance.append(d)
            # resort distances and probs
            indices=argsort(distance)
            distance=array(distance)[indices].tolist()
            probs=array(probs)[indices].tolist()
            # escape if runs for too ave_long
            tmp_max_iter=50
            if len(probs)>tmp_max_iter:
                # print('gradient_line_search never found a peak | max iter = {}'.format(tmp_max_iter))
                # break
                print(probs, distance)
                raise ValueError('gradient_line_search never found a peak | max iter = {}'.format(tmp_max_iter))
                # print( ValueError('gradient_line_search never found a peak | max iter = {}'.format(tmp_max_iter)) )
                # print("returning (probs, distance)")
                # return (probs, distance)



    # the maxima location is in the last three step_sizes
    # need to initiate a binary search of the space
    #
    # record number of binary searchs done
    iter_binary=1
    # binary search needs to be repeated until optimised to the threshold/tolerence
    # limit the total number of iterations to 10
    for i in range(max_iter):
    # while abs(d_prob)>tolerence:
        print('Iteration {} (Last logP = {})'.format(iter+i+1, probs[-1]))
        # get the indices of the most likely three points
        maxima_indicies=argmax(probs)
        # get the distances of the last three points
        last_three_distacnces=distance[maxima_indicies-1:maxima_indicies+2].copy()
        # first calculate the two mid points of the last three points
        diff_last_three_distacnces=diff(last_three_distacnces)/2
        points=[]
        for p, dp in zip(last_three_distacnces[:-1], diff_last_three_distacnces):
            points.append(p+dp)
        # print(points)
        # evaluate function at points
        for d in points:
            # take step along the line
            start=origin+grad*d
            # record prob of new postion
            probs.append(function(start))
            # upadte d_prob
            d_prob=probs[-1]-probs[-2]
            # recod the stepsizes
            distance.append(d)
            # print progress
            if callback:
                print('Binary search step {}: {}'.format(iter_binary, probs[-1]), flush=True)
            # if the solution is oscilating then break
        if len(probs)>3 and probs[-1]==probs[-3]:
            print(UserWarning('Solution is oscilating so aborting binary search!'))
            break

        # record the number of iterations
        iter_binary+=1
        # resort distances and probs
        indices=argsort(distance)
        distance=array(distance)[indices].tolist()
        probs=array(probs)[indices].tolist()

    # return optimisation estimate and search history
    return start, {'probs':array(probs).flatten(), 'distance':array(distance)}
