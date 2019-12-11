import sys, warnings

import numpy as np

from scipy import special
from scipy.constants import pi
from scipy.integrate import trapz
from scipy.interpolate import interp1d

from scipy import interpolate
from scipy.interpolate import UnivariateSpline # , BSpline

def reduce_wavelength_check_input(wavelengths, cwl, half_range, return_indicies, power2):
    # :param 1D ndarray wavelengths: Input array to be reduced
    if type(wavelengths) is not np.ndarray:
        raise TypeError("type(wavelengths) is not np.ndarray")
    if not any(np.isreal(wavelengths)):
        raise TypeError("wavelengths must only contain real scalars") # this also checks that the array is 1D
    # :param list or real scalar cwl: Point in the array which will be the centre of the new reduced array
    if not any([np.isreal(cwl), type(cwl) is not list]):
        raise TypeError("cwl must be a list or a real scalar")
    # :param real scalar half_range: Half the range of the new array.
    if not np.isreal(half_range):
        raise TypeError("half_range must be a real scalar")
    # :param boul return_indicies: Boulean (False by default) which when True the function returns the indicies
    #                              of 'wavelengths' that match the beginning and end of the reduced array
    if type(return_indicies) is not bool:
        raise TypeError("return_indicies must be a Boolean")
    if type(power2) is not bool:
        raise TypeError("power2 must be a Boolean")


def reduce_wavelength(wavelengths, cwl, half_range, return_indicies=False, power2=False):


    """
    This function returns an array which contains a subsection of input array ('wavelengths') which is
    between and inclucing the points 'cwl' +- 'half_range'. If the end of this range is outside of the
    'wavelength' array then the end of the reduced array is the end of the 'wavelength'.

    :param 1D ndarray wavelengths: Input array to be reduced
    :param list or real scalar cwl: Point in the array which will be the centre of the new reduced array
    :param real scalar half_range: Half the range of the new array.
    :param boul return_indicies: Boulean (False by default) which when True the function returns the indicies
                            of 'wavelengths' that match the beginning and end of the reduced array
    :return: Returns an subset of 'wavelengths' which is centred around 'cwl' with a range of 'cwl' +-
             'half_range'
    """

    reduce_wavelength_check_input(wavelengths, cwl, half_range, return_indicies, power2)

    if type(cwl) is list:
        cwl=cwl[0]

    upper_cwl = cwl + half_range
    lower_cwl = cwl - half_range

    if upper_cwl > max(wavelengths):
        upper_index = len(wavelengths) - 1
    else:
        tmp_wave = abs(wavelengths - upper_cwl)
        upper_index = np.where(tmp_wave == min(tmp_wave))[0][0]

    if lower_cwl < max(wavelengths):
        lower_index = 0
    else:
        tmp_wave = abs(wavelengths - lower_cwl)
        lower_index = np.where(tmp_wave == min(tmp_wave))[0][0]

    new_waves=wavelengths[lower_index:upper_index+1]
    if power2:
        new_len=2**int(np.log2(len(new_waves))//1)
        diff_len=len(new_waves)-new_len
        lower_index += diff_len // 2
        upper_index -= diff_len // 2
        if diff_len % 2 == 1:
            upper_index-=1
        new_waves=wavelengths[lower_index:upper_index+1]

        print(len(new_waves))

    if return_indicies:
        return new_waves, [lower_index, upper_index]
    else:
        return new_waves

def gaussian_check_input(x, cwl, fwhm, intensity):
    # :param 1D np.ndarray x: Axis to evaluate gaussian
    if type(x) is not np.ndarray:
        raise TypeError("type(x) is not np.ndarray")
    if not any(np.isreal(x)):
        raise TypeError("x must only contain real scalars") # this also checks that the array is 1D
    # :param scalar cwl: Mean of the gaussian
    if not np.isreal(cwl):
        raise TypeError("cwl must be a real scalar")
    # :param non-negative scalar fwhm: FWHM of the gaussian
    if not np.isreal(fwhm):
        raise TypeError("fwhm must be a real scalar")
    if fwhm<=0:
        raise ValueError("fwhm must be a positive scalar")
    # :param scalar intensity: Height of the gaussian
    if not np.isreal(intensity):
        raise TypeError("fwhm must be a real scalar")
    if intensity<=0:
        raise ValueError("fwhm must be a positive scalar")

fwhm_to_sigma=1/np.sqrt(8*np.log(2))
def gaussian(x, cwl, fwhm, intensity):
    '''
    Function for calculating a height normalised gaussian
    :param 1D np.ndarray x: Axis to evaluate gaussian
    :param scalar cwl: Mean of the gaussian
    :param positive scalar fwhm: FWHM of the gaussian
    :param scalar intensity: Height of the gaussian
    :return: 1D np.ndarray containing the gaussian
    '''
    gaussian_check_input(x, cwl, fwhm, intensity)
    sigma = fwhm*fwhm_to_sigma
    return intensity*np.exp(-0.5*((x-cwl)/sigma)**2)

root_half_steradian=np.sqrt(2*np.pi)
def gaussian_norm(x, cwl, fwhm, intensity):
    gaussian_check_input(x, cwl, fwhm, intensity)
    sigma=fwhm*fwhm_to_sigma
    k= intensity/(root_half_steradian*sigma)
    peak=np.exp(-0.5*((x-cwl)/sigma)**2)
    return k*peak

# def gaussian_norm(x, cwl, fwhm, intensity):
#     k = np.sqrt(half_steradian*np.square(fwhm*fwhm_to_sigma))
#     return gaussian(x, cwl, fwhm, intensity)/k

def put_in_iterable(input):
    if type(input) not in (tuple, list, np.ndarray):
        return [input]
    else:
        return input

class Gaussian(object):
    def __init__(self, x=None, cwl=None, fwhm=None, fractions=None, normalise=True, reduced_range=None):
        self.x=x
        self.cwl=np.array(put_in_iterable(cwl))
        self.fwhm=fwhm
        self.normalise=normalise

        if reduced_range is None:
            self.reducedx = [x for c in self.cwl]
            self.reducedx_indicies = [[0, len(x)-1] for c in self.cwl]
        else:
            self.reducedx = []
            self.reducedx_indicies = []
            for c in self.cwl:
                rx, rxi = reduce_wavelength(x, c, reduced_range/2, return_indicies=True)
                self.reducedx.append(rx)
                self.reducedx_indicies.append(rxi)

        if fractions is None:
            self.fractions = [1/len(self.cwl) for c in self.cwl]
        else:
            self.fractions = fractions

        if self.normalise:
            self.func = gaussian_norm
        else:
            self.func = gaussian

        if self.cwl is not None and self.fwhm is not None:
            self.peak=self.peak_1d
        elif self.cwl is not None:
            self.peak=self.peak_2d
        else:
            self.peak=self.peak_3d

    def __call__(self, theta):
        return self.peak(theta)

    def peak_1d(self, intensity):
        return self.make_peak(self.cwl, self.fwhm, intensity)

    def peak_2d(self, theta):
        fwhm, intensity = theta
        return self.make_peak(self.cwl, fwhm, intensity)

    def peak_3d(self, theta):
        cwl, fwhm, intensity = theta
        cwl = list(cwl)
        return self.make_peak(self.cwl, fwhm, intensity)

    def make_peak(self, cwl, fwhm, intensity):
        fwhm = put_in_iterable(fwhm)
        if len(fwhm)<len(cwl) and len(fwhm)==1:
            fwhm = [fwhm[0] for c in cwl]

        peak = np.zeros(len(self.x))
        for f, c, fw, rx, rxi in zip(self.fractions, cwl, fwhm, self.reducedx, self.reducedx_indicies):
            peak[rxi[0]:rxi[1]+1] += self.func(rx, c, fw, f)

        return intensity*peak


class SuperGaussian(object):
    def __init__(self, mean, sigma, half_power):
        self.mean = mean
        self.sigma = sigma
        self.half_power = half_power

    def __call__(self, theta, log=True):
        inside = (theta - self.mean) / self.sigma
        log_peak = -0.5 * np.power(inside, 2*self.half_power)
        if log:
            return log_peak
        else:
            return np.exp(log_peak)


class MeshLine(object):
    def __init__(self, x, bounds=[-10, 10], zero_bounds=None, resolution=0.1, log=False, kind='linear'):
        self.x = np.arange(min(bounds), max(bounds), resolution)
        self.log = log
        self.kind = kind
        self.zero_bounds = zero_bounds
        if self.zero_bounds is not None:
            self.slice=slice(1, -1)
            self.empty_theta=np.zeros(len(x)+2)
            self.empty_theta[0]=zero_bounds
            self.empty_theta[-1]=zero_bounds
            self.x_points=np.concatenate([np.array([min(bounds)]), x, np.array([max(bounds)])])
        else:
            self.empty_theta = np.zeros(len(x))
            self.slice=slice(0, len(self.empty_theta))
            self.x_points=x


        self.check_init()

    def check_init(self):
        if len(self.x_points) != len(self.empty_theta):
            print('self.x_points', self.x_points)
            print('self.empty_theta', self.empty_theta)
            raise ValueError("len(self.x_points) != len(self.empty_theta)")

    def __call__(self, theta, *args, **kwargs):
        self.empty_theta[self.slice]=theta
        if self.log:
            get_new_profile = interp1d(self.x_points, np.power(10, self.empty_theta), self.kind, bounds_error=False, fill_value='extrapolate')
            return get_new_profile(self.x)
        else:
            get_new_profile = interp1d(self.x_points, self.empty_theta, self.kind, bounds_error=False, fill_value='extrapolate')
            return get_new_profile(self.x)


class PlasmaMeshLine(object):

    def __init__(self, x, separatix_index, bounds=[-10, 10], zero_bounds=False, resolution=0.1, kind='quadratic'):

        self.meshprofile = MeshLine(x, bounds, zero_bounds, resolution, kind)

        self.separatix_index = separatix_index

    def electron_temperature(self, theta):

        meshin = np.zeros(len(theta)) + theta[self.separatix_index]

        theta[self.separatix_index] = 1

        if type(theta) == list:
            theta = np.array(theta)

        meshin = theta * meshin

        return self.meshprofile(meshin)

    def electron_density(self, theta):

        return self.meshprofile(theta)

def bowman_tee_distribution(x, theta):
    A, x0, sigma0, q, nu, k, f, b = theta

    p0=f*np.tanh(k*(x-x0))
    sigma=sigma0*np.exp(p0)
    z=(x-x0)/sigma
    z=np.power(abs(z), q)
    p1=np.power((1+z)/nu, -nu)

    return A*p1+b

def bowman_tee_distribution_centred(x, theta):
    A, sigma0, q, nu, k, f, b = theta
    return bowman_tee_distribution(x, [A, 0, sigma0, q, nu, k, f, b])

from copy import copy

class BowmanTeeNe(object):
    def __init__(self, x):
        self.x=x

    def __call__(self, theta):
        theta=copy(theta)
        for i in [0, -1]:
            theta[i]=np.power(10, theta[i])
        return bowman_tee_distribution(self.x, theta)

class BowmanTeeTe(object):
    def __init__(self, x):
        self.x=x

    def __call__(self, theta):
        theta=copy(theta)
        for i in [0, -1]:
            theta[i]=np.power(10, theta[i])
        return bowman_tee_distribution_centred(self.x, theta)

from itertools import product

class BowmanTeePlasma(object):
    def __init__(self, x=None, bounds=None, dr_bounds=[-2, 2], bounds_ne=[11, 16], bounds_te=[-1, 2]):
        if x is None:
            self.x=np.linspace(-15, 25, 500)
        else:
            self.x=x

        if bounds is None:
            self.bounds=[[1e-3, 10], [0.1, 10], [1, 2],
                         [1, 10],  [0.5, 3]]
        else:
            self.bounds=bounds

        self.electron_density=BowmanTeeNe(self.x)
        self.electron_temperature=BowmanTeeTe(self.x)

        self.electron_density.number_of_variables=8
        self.electron_temperature.number_of_variables=7

        self.construct_bounds(dr_bounds, bounds_ne, bounds_te)

    def construct_bounds(self, dr_bounds,bounds_ne, bounds_te):
        """
        A > 0
        sigma0 > 0
        0.5 < q < 20
        nu > 1
        1 < k < 50
        1 < f < 10
        0 < b
        """

        self.bounds_ne=[bounds_ne, dr_bounds]
        self.bounds_te=[bounds_te]

        for param, bound in product([self.bounds_ne, self.bounds_te], self.bounds):
            param.append(bound)

        for param, bound in zip([self.bounds_ne, self.bounds_te], [bounds_ne, bounds_te]):
            param.append(bound)

        self.electron_density.bounds=self.bounds_ne
        self.electron_temperature.bounds=self.bounds_te

if __name__=='__main__':

    from tulasa.general import plot, close_plots
    import time as clock

    profile_function=BowmanTeePlasma()

    theta0=np.ones(7)+1
    theta0[1]=5
    peaks=[]
    for i in np.linspace(0., 20, 5): # .5*(np.arange(5)+1):
        theta0[1]=5
        theta0[2]=i
        # theta0[-2]=i
        print(i, ': ', *theta0)
        peaks.append(profile_function.electron_temperature(theta0))

    plot(peaks, x=[profile_function.electron_temperature.x for p in peaks], multi='fake') #, log=True)
