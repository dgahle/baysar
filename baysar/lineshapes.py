import sys, warnings

import numpy as np

from scipy import special
from scipy.constants import pi
from scipy.integrate import trapz
from scipy.interpolate import interp1d

from scipy import interpolate
from scipy.interpolate import UnivariateSpline # , BSpline

def reduce_wavelength_check_input(wavelengths, cwl, half_range, return_indicies):
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


def reduce_wavelength(wavelengths, cwl, half_range, return_indicies=False):


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

    reduce_wavelength_check_input(wavelengths, cwl, half_range, return_indicies)

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

    # print(lower_index, upper_index)

    if return_indicies:
        return wavelengths[lower_index:upper_index+1], [lower_index, upper_index]
    else:
        return wavelengths[lower_index:upper_index + 1]

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
    sigma = fwhm/np.sqrt(8*np.log(2))
    return intensity*np.exp(-0.5*((x-cwl)/sigma)**2)

def gaussian_norm(x, cwl, fwhm, intensity):
    k = np.sqrt(2*np.pi*np.square(fwhm/np.sqrt(8*np.log(2))))
    return gaussian(x, cwl, fwhm, intensity)/k

def put_in_iterable(input):
    if type(input) not in (tuple, list, np.ndarray):
        return [input]
    else:
        return input

class Gaussian(object):
    def __init__(self, x=None, cwl=None, fwhm=None, fractions=None, normalise=True, reduced_range=None):
        self.x=x
        self.cwl=put_in_iterable(cwl)
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

    def __call__(self, theta):
        if self.cwl is not None and self.fwhm is not None:
            intensity = theta
            fwhm = self.fwhm
            cwl = self.cwl
        elif self.cwl is not None:
            fwhm, intensity = theta
            cwl = self.cwl
        else:
            cwl, fwhm, intensity = theta
            cwl = list(cwl)

        fwhm = put_in_iterable(fwhm)
        if len(fwhm)<len(cwl) and len(fwhm)==1:
            fwhm = [fwhm[0] for c in cwl]

        peak = np.zeros(len(self.x))
        for f, c, fw, rx, rxi in zip(self.fractions, cwl, fwhm, self.reducedx, self.reducedx_indicies):
            peak[min(rxi):max(rxi)+1] += f*self.func(rx, c, fw, 1)

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
        self.x_points = x
        self.x = np.arange(min(bounds), max(bounds), resolution)
        self.empty_theta = np.zeros(len(self.x_points))        
        self.log = log
        self.kind = kind

        self.zero_bounds = zero_bounds
        if self.zero_bounds is not None:
            tmp_x = np.zeros(len(self.x_points)+2)
            tmp_x[0] = min(bounds)
            tmp_x[-1] = max(bounds)
            tmp_x[1:-1] = self.x_points
            self.x_points = tmp_x
            self.empty_theta[0] = zero_bounds
            self.empty_theta[-1] = zero_bounds

    def __call__(self, theta, *args, **kwargs):
        if self.zero_bounds is not None:
            self.empty_theta[1:-1] = theta
            theta = self.empty_theta
        assert len(theta) == len(self.x_points), 'len(theta) != len(self.x)' + \
                                             str(len(theta)) + ' ' + str(len(self.x_points))

        # get_new_profile = InterpolatedUnivariateSpline(self.x, theta)
        # get_new_profile = UnivariateSpline(self.x_points, theta)
        get_new_profile = interp1d(self.x_points, theta, self.kind, bounds_error=False, fill_value='extrapolate')

        if self.log:
            return np.power(10, get_new_profile(self.x))
        else:
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


if __name__=='__main__':

    from tulasa.general import plot
    import time as clock

    # gaussian check
    x = np.linspace(400, 402, 500)
    cwl = [401, 401.5]
    fwhm = 0.2
    intensity = 1

    peak = Gaussian(x=x, cwl=cwl, fwhm=fwhm, fractions=None, normalise=True)

    plot(peak(1), x=x)
