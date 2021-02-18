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
        upper_index = len(wavelengths)
    else:
        tmp_wave = abs(wavelengths - upper_cwl)
        upper_index = np.where(tmp_wave == min(tmp_wave))[0][0]+1

    if lower_cwl < min(wavelengths):
        lower_index = 0
    else:
        tmp_wave = abs(wavelengths - lower_cwl)
        lower_index = np.where(tmp_wave == min(tmp_wave))[0][0]

    new_waves_slice=slice(lower_index, upper_index)
    new_waves=wavelengths[new_waves_slice]

    if return_indicies:
        return new_waves, new_waves_slice
    else:
        return new_waves

def gaussian_check_input(x, cwl, fwhm, intensity):
    # :param 1D np.ndarray x: Axis to evaluate gaussian
    if type(x) is not np.ndarray:
        raise TypeError("type(x) is not np.ndarray")
    if not any(np.isreal(x)):
        print('x =', x)
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
            self.reducedx_indicies = [slice(0, len(x)) for c in self.cwl] # TODO: here
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
        return self.make_peak(cwl, fwhm, intensity)

    def make_peak(self, cwl, fwhm, intensity):
        fwhm = put_in_iterable(fwhm)
        if len(fwhm)<len(cwl) and len(fwhm)==1:
            fwhm = [fwhm[0] for c in cwl]

        peak = np.zeros(len(self.x))
        for f, c, fw, rx, rxi in zip(self.fractions, cwl, fwhm, self.reducedx, self.reducedx_indicies):
            peak[rxi] += self.func(rx, c, fw, f)

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
    def __init__(self, x, x_ends=[-10, 10], zero_bounds=None, resolution=0.1,
                       log=False, kind='linear', **kwargs):
        self.__dict__.update(kwargs)
        self.log = log
        self.kind = kind
        self.zero_bounds = zero_bounds
        if self.zero_bounds is None:
            self.empty_theta = np.zeros(len(x))
            self.slice=slice(0, len(self.empty_theta))
            self.x_points=x
        else:
            self.slice=slice(1, -1)
            self.empty_theta=np.zeros(len(x)+2)
            self.empty_theta[0]=zero_bounds
            self.empty_theta[-1]=zero_bounds
            self.x_points=np.concatenate([np.array([min(x_ends)]), x, np.array([max(x_ends)])])
        self.x_points=self.x_points.astype(float)
        self.x = np.arange(min(x_ends), max(x_ends), resolution)

        self.check_init()

    def check_init(self):
        if len(self.x_points) != len(self.empty_theta):
            print('self.x_points', self.x_points)
            print('self.empty_theta', self.empty_theta)
            raise ValueError("len(self.x_points) != len(self.empty_theta)")
        if any([t==0 for t in np.diff(self.x_points)]):
            raise ValueError("Some of self.x_points are the same. Zero bounds = {}".format(self.zero_bounds))

    def __call__(self, theta, *args, **kwargs):
        self.empty_theta[self.slice]=theta
        if self.log:
            get_new_profile = interp1d(self.x_points, np.power(10, self.empty_theta), self.kind, bounds_error=False, fill_value='extrapolate')
            return get_new_profile(self.x)
        else:
            get_new_profile = interp1d(self.x_points, self.empty_theta, self.kind, bounds_error=False, fill_value='extrapolate')
            return get_new_profile(self.x)


class MeshPlasma(object):
    def __init__(self, x=None, bounds=None, zero_bounds_ne=11, zero_bounds_te=-2, bounds_ne=[11, 16], bounds_te=[-1, 2]):
        self.electron_density=MeshLine(x=x, zero_bounds=zero_bounds_ne, x_ends=bounds, log=True, number_of_variables=len(x), bounds=[bounds_ne for n in np.arange(len(x))])
        self.electron_temperature=MeshLine(x=x, zero_bounds=zero_bounds_te, x_ends=bounds, log=True, number_of_variables=len(x), bounds=[bounds_te for n in np.arange(len(x))])

def bowman_tee_distribution(x, theta):
    A, x0, sigma0, q, nu, k, f, b = theta

    p0=f*np.tanh(k*(x-x0))
    sigma=sigma0*np.exp(p0)
    z=(x-x0)/sigma
    z=np.power(abs(z), q)
    p1=np.power(1+z/nu, -(nu+1)*0.5)

    peak=A*p1+b

    return peak

def bowman_tee_distribution_centred(x, theta):
    A, sigma0, q, nu, k, f, b = theta
    return bowman_tee_distribution(x, [A, 0, sigma0, q, nu, k, f, b])

from copy import copy

class BowmanTeeNe:
    def __init__(self, x, background=True):
        self.x=x
        self.background=background

    def __call__(self, theta):
        theta=copy(theta)
        if self.background:
            for i in [0, -1]:
                theta[i]=np.power(10, theta[i])
        else:
            theta[0]=np.power(10, theta[0])
            theta=np.concatenate((theta, np.zeros(1)))
        return self.profile(theta)

    def profile(self, theta):
        return bowman_tee_distribution(self.x, theta)

class BowmanTeeTe(BowmanTeeNe):
    def profile(self, theta):
        return bowman_tee_distribution_centred(self.x, theta)

from itertools import product

class BowmanTeePlasma(object):
    def __init__(self, x=None, bounds=None, dr_bounds=[-2, 2], bounds_ne=[11, 16], bounds_te=[-1, 2], background=False):
        if x is None:
            self.x=np.linspace(-15, 25, 500)
        else:
            self.x=x

        if bounds is None:
            self.bounds=[[1e-1, 10], [1.2, 3], [1, 5],
                         [1, 50],  [0, 2]]
        else:
            self.bounds=bounds

        self.background=background

        self.electron_density=BowmanTeeNe(self.x, background=self.background)
        self.electron_temperature=BowmanTeeTe(self.x, background=self.background)

        self.electron_density.number_of_variables=7
        self.electron_temperature.number_of_variables=6

        if self.background:
            self.electron_density.number_of_variables+=1
            self.electron_temperature.number_of_variables+=1

        self.construct_bounds(dr_bounds, bounds_ne, bounds_te)

    def construct_bounds(self, dr_bounds,bounds_ne, bounds_te):
        """
        A > 0
        sigma0 > 0
        1.2 < q < 3
        nu > 1
        1 < k < 50
        0 < f < 2
        0 < b
        """

        self.bounds_ne=[bounds_ne, dr_bounds]
        self.bounds_te=[bounds_te]

        for param, bound in product([self.bounds_ne, self.bounds_te], self.bounds):
            param.append(bound)

        if self.background:
            for param, bound in zip([self.bounds_ne, self.bounds_te], [[bounds_ne[0], 13.], [bounds_te[0], 0]]):
                param.append(bound)

        self.electron_density.bounds=self.bounds_ne
        self.electron_temperature.bounds=self.bounds_te


class SimplePlasma:
    def __init__(self, x=None): # , bounds=None, dr_bounds=[0, 2], bounds_ne=[11, 16], bounds_te=[-1, 2]):
        if x is None:
            self.x=np.linspace(0, 60, 100)
        else:
            self.x=x

        self.electron_density=Poisson(self.x)
        self.electron_temperature=ExpDecay(self.x)

class SimplePlasma:
    def __init__(self, x=None): # , bounds=None, dr_bounds=[0, 2], bounds_ne=[11, 16], bounds_te=[-1, 2]):
        if x is None:
            self.x=np.linspace(0, 60, 100)
        else:
            self.x=x

        self.electron_density=ExpDecay(self.x)
        self.electron_density.bounds[0]=[12, 16]
        self.electron_temperature=ExpDecay(self.x)

class ExpDecay:
    def __init__(self, x):
        self.x=x
        self.x_len=len(self.x)
        self.number_of_variables=2
        self.bounds=[ [-1, 2], # peak range
                      [-2, 1] ] # base range

    def __call__(self, theta):
        a, b=[np.power(10., t) for t in theta]
        base=np.ones(self.x_len)+b

        return (a*np.power(base, -self.x)).clip(0.01)

class ExpDecay:
    def __init__(self, x):
        self.x=x
        self.x_len=len(self.x)
        self.number_of_variables=2
        self.bounds=[ [-1, 2], # peak range
                      [-2, 1] ] # base range

    def __call__(self, theta):
        a, b=[np.power(10., t) for t in theta]
        base=np.ones(self.x_len)+b

        return (a*np.power(base, -self.x)).clip(0.01)

class ADoubleExpDecay:
    def __init__(self, x):
        self.x=x
        self.x_len=len(self.x)
        self.number_of_variables=5
        self.bounds=[ [12, 16], # peak range
                      [-2, 1], # base range
                      [0, 50], # peak center
                      [-3, 1], # asymmetry size
                      [-5, 5] ] # asymmetry gradient ()

    def __call__(self, theta):
        # a, b=[np.power(10., t) for t in theta]
        a, b, x0, f, k=theta

        base=np.ones(self.x_len)+np.power(10., b)
        base*=np.exp( np.power(10., f)*np.tanh(k*(self.x-x0)) )

        return (np.power(10., a)*np.power(base, x0-self.x)).clip(0.01)

class LessSimplePlasma:
    def __init__(self, x=None): # , bounds=None, dr_bounds=[0, 2], bounds_ne=[11, 16], bounds_te=[-1, 2]):
        if x is None:
            self.x=np.linspace(0, 60, 100)
        else:
            self.x=x

        self.electron_density=ADoubleExpDecay(self.x)
        self.electron_temperature=ExpDecay(self.x)

from scipy.special import factorial
class Poisson:
    def __init__(self, x):
        self.x=x
        self.x_len=len(self.x)
        self.number_of_variables=2
        self.bounds=[ [12, 16], # peak range
                      [-3, 2] ] # base range

    def __call__(self, theta):
        a, nu=[np.power(10., t) for t in theta]
        peak=np.power(np.zeros(self.x_len)+nu, self.x)
        peak*=np.exp(-nu)/factorial(self.x)
        peak*=(a/peak.max())

        return peak

    # def __call__(self, theta):
    #     a, nu=theta

class GaussianPlasma:
    def __init__(self, x=None, cwl=None): # , bounds=None, dr_bounds=[0, 2], bounds_ne=[11, 16], bounds_te=[-1, 2]):
        self.x=x
        self.cwl=cwl

        self.get_attributes()

    def __call__(self, theta):
        if self.cwl is None:
            intensity, fwhm, cwl=theta
        else:
            intensity, fwhm=theta
            cwl=self.cwl

        intensity=np.power(10, intensity)
        fwhm=np.power(10, fwhm)
        peak=gaussian(self.x, cwl, fwhm, intensity)
        return peak.clip(0.01)

    def get_attributes(self):
        x_bounds0=[self.x.min(), self.x.max()]
        x_bounds1=[-1, np.log10(self.x.max()/2)]
        if self.cwl is None:
            self.number_of_variables=3
            self.bounds=[ [12, 16], # peak range
                           x_bounds1, # fwhm range
                           x_bounds0 ] # cwl range
        else:
            self.number_of_variables=2
            self.bounds=[ [12, 16], # peak range
                          x_bounds1 ] # fwhm range




class SimpleGaussianPlasma:
    def __init__(self, x=None): # , bounds=None, dr_bounds=[0, 2], bounds_ne=[11, 16], bounds_te=[-1, 2]):
        if x is None:
            self.x=np.linspace(0, 60, 100)
        else:
            self.x=x

        self.electron_density=GaussianPlasma(x=self.x)
        self.electron_temperature=GaussianPlasma(x=self.x, cwl=0)


from scipy.interpolate import interp1d
def centre_peak(x, y, centre=0):
    ycentre=x[y.argmax()]
    y=interp1d(x, y, fill_value="extrapolate")(x+ycentre-centre)
    return y

class ReducedBowmanTProfile:
    def __init__(self, x, log_peak_bounds, centre=None, dr_bounds=None,
                       sigma_bounds=None, asdex=False, asdex_c2=0.7, **kwargs):

        self.x=x
        self.bounds=[log_peak_bounds]
        self.centre=centre
        self.dr_bounds=dr_bounds
        self.sigma_bounds=sigma_bounds

        self.asdex=asdex
        self.asdex_c2=asdex_c2

        self.get_bounds()

    def __call__(self, theta):
        # if self.centre is None and self.asdex:
        #     A, c, sigma, f, B=theta
        # elif self.centre is None:
        #     A, c, sigma, f=theta
        # else:
        #     A, sigma, f=theta
        #     c=self.centre
        if self.asdex:
            if self.centre is None:
                A, c, sigma, f, B=theta
            else:
                A, sigma, f, B=theta
                c=self.centre
        else:
            if self.centre is None:
                A, c, sigma, f=theta
            else:
                A, sigma, f=theta
                c=self.centre

        # btheta=[np.power(10, A), c, np.power(10, sigma), self.q, self.nu, self.k, f, 0]
        if self.asdex:
            btheta=[np.power(10, A)-np.power(10, B), c, sigma, self.q, self.nu, self.k, f, np.power(10, B)]
        else:
            btheta=[np.power(10, A), c, sigma, self.q, self.nu, self.k, f, 0]

        peak=bowman_tee_distribution(self.x, btheta)
        peak=centre_peak(self.x, peak, centre=c)
        return peak


    def get_bounds(self):

        if self.centre is None:
            if self.dr_bounds is None:
                self.bounds.append([-5, 5])
            else:
                self.bounds.append(self.dr_bounds)


        if self.sigma_bounds is None:
            self.bounds.append([1, 10])
        else:
            self.bounds.append(self.sigma_bounds)

        self.bounds.append([0, 2])
        if self.asdex:
            self.bounds.extend([self.bounds[0]])


        self.number_of_variables=len(self.bounds)

        self.q=2. # gaussian setting
        self.nu=1.
        self.k=1.

class ReducedBowmanTPlasma(object):
    def __init__(self, x=None, sigma_bounds=[1, 10], dr_bounds=[-5, 5], bounds_ne=[11, 16], bounds_te=[-1, 2], asdex=False):
        if x is None:
            self.x=np.linspace(-15, 35, 50)
        else:
            self.x=x

        self.electron_density=ReducedBowmanTProfile(self.x, bounds_ne, centre=None, dr_bounds=dr_bounds, sigma_bounds=sigma_bounds, asdex=asdex)
        self.electron_temperature=ReducedBowmanTProfile(self.x, bounds_te, centre=0, sigma_bounds=sigma_bounds, asdex=asdex)

class LinearSeparatrix:
    def __init__(self, x, peak_bounds=[0, 1.7]):
        self.x=x
        self.bounds=[peak_bounds]
        self.bounds.append([0.5, 1.5])
        self.number_of_variables=len(self.bounds)

    def __call__(self, theta):
        Te, grad=theta
        profile=np.power(10, Te)-grad*self.x
        profile=profile.clip(0.1)
        return np.log10(profile)


class CauchySeparatrix:
    def __init__(self, x, peak_bounds=[12.7, 16], centre=None):
        self.x=x
        self.centre=centre
        # build bounds
        self.bounds=[peak_bounds, [12.7, 14]]
        if self.centre is None:
            self.bounds.append([-5, 5])
        self.bounds.append([0.5, 1.5])

    def __call__(self, theta):
        if self.centre is None:
            ne, ne_min, centre, sigma=theta
        else:
            ne, ne_min, sigma=theta
            centre=self.centre

        chi_squared=np.square( (centre-self.x)/sigma )
        profile=np.power(10, ne_min)+np.power(10, ne)/(1+chi_squared)
        profile.clip(1e11)
        return np.log10(profile)

class SimpleSeparatrix(object):
    def __init__(self, chords, bounds_ne=[11, 16], bounds_te=[-1, 2]):
        self.x=np.array(chords)

        self.electron_density=CauchySeparatrix(self.x, bounds_ne) # , centre=self.x[-1])
        self.electron_temperature=LinearSeparatrix(self.x, bounds_te)

def esymmtric_gaussian(x, ems, cwl, sigma, efactor):
    # dx=cwl-x
    dx=x-cwl
    sigma+=efactor/(1+np.exp(-dx))
    mean_square=np.square(dx/sigma)

    return ems*np.exp(-0.5*mean_square)

def esymmtric_cauchy(x, ems, cwl, sigma, efactor):
    dx=x-cwl
    sigma+=efactor/(1+np.exp(-dx))
    mean_square=np.square(dx/sigma)

    return ems/(1+mean_square)

class EsymmtricProfile:
    def __init__(self, x, log_peak_bounds, centre=None, ptype='cauchy'):

        self.x=x
        self.bounds=[log_peak_bounds]
        self.centre=centre

        self.get_bounds()

        if ptype not in ('cauchy', 'gaussian'):
            raise ValueError("ptype not in ('cauchy', 'gaussian')")

        if ptype is 'cauchy':
            self.peak=esymmtric_cauchy
        if ptype is 'gaussian':
            self.peak=esymmtric_gaussian

    def __call__(self, theta):
        if self.centre is None:
            # A, c, sigma, f=theta
            A, c, f=theta
        else:
            # A, sigma, f=theta
            A, f=theta
            c=self.centre

        # sigma=1
        # btheta=[np.power(10, A), c, sigma, f]
        # btheta=[np.power(10, A), c, f, sigma]
        btheta=[np.power(10, A), c, 0.2*f, f]
        return self.peak(self.x, *btheta).clip(1e-3)

    def get_bounds(self):

        if self.centre is None:
            self.bounds.append([-5, 2])

        shape_bounds=[[1., 10.],
                      [0, 10]] # asymmetry bounds

        # self.bounds.extend(shape_bounds)
        self.bounds.append([1, 6])
        self.number_of_variables=len(self.bounds)

class EsymmtricCauchyPlasma:
    def __init__(self, x=None, dr_bounds=[-5, 2], bounds_ne=[13, 15], bounds_te=[0., 1.7], dr=None, ptype='cauchy', **args):
        if x is None:
            self.x=np.linspace(-10, 25, 50)
        else:
            self.x=x

        self.dr = dr

        self.electron_density=EsymmtricProfile(self.x, bounds_ne, centre=self.dr, ptype=ptype)
        self.electron_temperature=EsymmtricProfile(self.x, bounds_te, centre=0, ptype=ptype)


if __name__=='__main__':

    from tulasa.general import close_plots, plot

    x=np.linspace(-20, 50, 50)
    l=ReducedBowmanTPlasma(x=x)
    peaks=[l.electron_temperature([0, 0, 1]), l.electron_density([0, -2, .5, 1.5])]
    plot(peaks, x=[x, x], multi='fake')
