from numpy import square, sqrt, mean, linspace, nan, log, diff, arange, zeros, concatenate, where

from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import sparse
import numpy as np
import time as clock

import os, sys, io, warnings

def clip_data(x_data, y_data, x_range):

    '''
    returns sections of x and y data that is within the desired x range

    :param x_data:
    :param y_data:
    :param x_range:
    :return:
    '''

    assert len(x_data)==len(y_data), 'len(x_data)!=len(y_data)'

    x_index = where((min(x_range) < x_data) & (x_data < max(x_range)))

    if not np.isreal(x_data[x_index]).all():
        print('not isreal(x_data[x_index])')
        print(x_data)
    if not np.isreal(y_data[x_index]).all():
        print('not isreal(Y_data[x_index])')
        print(y_data)

    return x_data[x_index], y_data[x_index]

def calibrate_wavelength(spectra, waves, points, references, width=0.5):
    pixels=[]
    for p in points:
        tmpx, tmpy = clip_data(waves, spectra, [p-width, p+width])
        com=center_of_mass(tmpy)[0]
        shift=where(waves==tmpx.min())[0][0]
        pixels.append(com+shift)

    newaxis=interp1d(pixels, references, bounds_error=False, fill_value="extrapolate")

    return newaxis(np.arange(len(spectra)))

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

from baysar.input_functions import within
from scipy.ndimage.measurements import center_of_mass

def centre_peak0(peak, places=7):
    com=center_of_mass(peak)
    x=np.arange(len(peak))
    centre=np.mean(x)
    shift=com-centre
    interp=interp1d(x, peak, bounds_error=False, fill_value=0.)
    return interp(x+shift)

def centre_peak(peak, places=12):
    x=np.arange(len(peak))
    centre=np.mean(x)
    while not np.round(abs(centre-center_of_mass(peak))[0], places)==0:
        peak=centre_peak0(peak)
    return peak

def within(stuff, boxes):

    """
    This function checks is a value or list/array of values (the sample) are 'within' an array or values.

    :param stuff: a list of strings, ints or floats
    :param box: Array of values to check the sample(s) against
    :return: True or False if all the samples are in the array
    """

    if type(stuff) in (tuple, list):
        return any([any([(box.min() < float(s)) and (float(s) < box.max()) for s in stuff])
                    for box in boxes])
    else:
        if type(stuff) is str:
            stuff = float(stuff)
        elif type(stuff) in (int, float, np.float):
            pass
        else:
            raise TypeError("Input stuff is not of an appropriate type.")

        return any((box.min() < stuff) and (stuff < box.max()) for box in boxes)


from numpy import array, linspace, diff, argsort, argmax, isclose, inf
from numpy.linalg import norm as normalise

from scipy.optimize import approx_fprime

from copy import copy

def check_input():
    """
    write a fucntion for generalised type/value checking
    """
    pass

def type_check():
    pass

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
