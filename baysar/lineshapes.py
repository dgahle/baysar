import sys, warnings

import numpy as np

from scipy import special
from scipy.constants import pi
from scipy.integrate import trapz
from scipy.interpolate import interp1d

from scipy import interpolate
from scipy.interpolate import UnivariateSpline # , BSpline


class Gaussian(object):

    def __init__(self, x=None, cwl=None, vectorise=1):

        self.check_input(x, cwl, vectorise)

        # Useful constants
        self.fwhm_to_sigma = 1 / np.sqrt(8 * np.log(2))

        self.vectorise = vectorise

        if self.vectorise > 1:
            self.x = np.tile(x, (vectorise, 1))
        else:
            self.x = x

        self.cwl = cwl

    def check_input(self, x, cwl, vectorise):
        if type(x)!=np.ndarray:
            raise TypeError("x must be an numpy array")
        if type(cwl) not in (int, float, np.int64, np.float64):
            raise TypeError("cwl must be an int or float")
        if type(vectorise) not in (int, np.int64):
            raise TypeError("vectorise must be an int")

    def __call__(self, theta, *args, **kwargs):

        if self.cwl is not None:
            fwhm, intensity = theta
            cwl = self.cwl
        else:
            cwl, fwhm, intensity = theta

        if self.vectorise > 1:
            if any([tmp_fwhm != np.mean(fwhm) for tmp_fwhm in fwhm]):
                # tiling
                fwhm = np.array([np.zeros(len(self.x[0])) + t for t in fwhm])
                intensity = np.array([np.zeros(len(self.x[0])) + t for t in intensity.flatten()])
                # intensity = np.array([np.zeros(len(self.x[0])) + t for t in intensity[0]])
            else:
                intensity = sum(intensity)
                sigma = np.mean(fwhm) * self.fwhm_to_sigma
                peak = np.exp(-0.5 * ( (self.x[0] - cwl) / sigma) ** 2)

                return sum( np.array(intensity).flatten() )  * peak.flatten()

        # guassian function
        if type(fwhm)==list:
            fwhm = np.array(fwhm)
        sigma = fwhm * self.fwhm_to_sigma


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peak = np.exp(-0.5 * ( (self.x - cwl) / sigma) ** 2)

        return intensity * peak


class Gaussians(object):

    def __init__(self, x=None, cwl=None, vectorise=1):

        'type checking done by Gaussian object'

        self.cwl = cwl

        self.peaks = []

        try:
            self.many = True
            for centre in cwl:
                self.peaks.append(Gaussian(x, centre, vectorise=vectorise))
        except TypeError:
            self.peaks.append(Gaussian(x, cwl, vectorise=vectorise))
        except:
            raise

    def __call__(self, theta, give_list=False, *args, **kwargs):

        try:
            if self.many:
                # if not give_list:
                #     return [t(theta) for t in self.peaks]
                # else:
                #     return sum([t(theta) for t in self.peaks])
                return sum([t(theta) for t in self.peaks])
        except NameError:
            return self.peaks[0](theta)
        except:
            raise


class GaussianNorm(Gaussian): # TODO: Needs fixing/making ?

    def __init__(self, x=None, cwl=None, vectorise=1, scaler=1):

        'type checking done by Gaussian object apart from scaler'

        if any([scaler < 0, scaler > 1, type(scaler) not in (int, float, np.int64, np.float64)]):
            raise AssertionError("scaler must be an int or a float greater than 0 and no greater than 1")

        self.x = x
        self.cwl = cwl

        self.scaler = scaler
        self.vectorise = vectorise

        if self.vectorise > 1:
            self.x = np.tile(x, (vectorise, 1))
        else:
            self.x = x

        super(GaussianNorm, self).__init__(x, cwl, vectorise=self.vectorise)

        self.fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))

        self.sigma_to_norm_k = np.sqrt(2 * np.pi)

    def __call__(self, theta, *args, **kwargs):

        peak = super(GaussianNorm, self).__call__(theta) * self.scaler #, give_list=True)

        if self.cwl is not None:
            fwhm = theta[0]
        else:
            fwhm = theta[1]

        # tiling
        if self.vectorise > 1:
            fwhm = np.array([np.zeros(len(self.x[0])) + t for t in fwhm])

        try:
            const = np.sqrt(2 * np.pi * np.square(fwhm * self.fwhm_to_sigma) )
        except TypeError:
            fwhm = fwhm[0]
            const = np.sqrt(2 * np.pi * np.square(fwhm * self.fwhm_to_sigma) )
            # print( *fwhm, self.fwhm_to_sigma )
            # raise
        except:
            raise

        try:
            return peak / const
        except ValueError:
            print(self.x.shape)
            print(peak.shape)
            print(fwhm.shape)
            raise
        except:
            raise


class GaussiansNorm(object): # TODO: Needs fixing/making

    def __init__(self, x=None, cwl=None, fwhm=None, fractions=[], vectorise=1):

        if type(cwl) == list:
            self.cwl = cwl
        else:
            self.cwl = [cwl]

        self.fwhm = fwhm

        self.fractions = fractions

        self.peaks = []

        self.vectorise = vectorise

        try:
            self.many = True
            for counter, centre in enumerate(self.cwl):
                self.peaks.append( GaussianNorm( x, centre, vectorise=self.vectorise,
                                                 scaler=self.fractions[counter] ) )
        except IndexError:
            org_sum_fractions = sum(self.fractions)
            if all([ (len(self.fractions) == 0) and (len(self.cwl) > 1) ]):
                for tmp in self.cwl:
                    self.fractions.append( 1 / len(self.cwl) )
            elif all([ (len(self.fractions) < len(self.cwl)) and (org_sum_fractions < 1) ]):
                len_extra_fractions = len(self.cwl) - len(self.fractions)
                while len(self.fractions) < len(self.cwl):
                    self.fractions.append( ( 1 - org_sum_fractions ) / len_extra_fractions )
            else:
                self.fractions = np.zeros( len(self.cwl) ) + ( 1 / len(self.cwl) )

            try:
                self.many = True

                self.peaks = []
                for counter, centre in enumerate(self.cwl):
                    self.peaks.append(GaussianNorm(x, centre, vectorise=vectorise,
                                                   scaler=self.fractions[counter]))
            except:
                raise
        except TypeError:
            self.many = False
            self.peaks = []
            self.peaks.append(GaussianNorm(x, self.cwl, vectorise=vectorise))
        except:
            raise

        if all([type(self.cwl) != t for t in (int, float)]):
            assert len(self.cwl) == len(self.peaks), str(len(self.cwl)) + ' ' + str(len(self.peaks))

    def __call__(self, theta, give_list=False, *args, **kwargs):

        try:
            if self.many:
                if give_list:
                    return [t(theta) for t in self.peaks]
                else:
                    return sum([t(theta) for t in self.peaks])
            else: pass
        except NameError:
            return self.peaks[0](theta)
        except:
            raise


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

    def __init__(self, x, bounds=[-10, 10], zero_bounds=None, resolution=0.1, log=False, kind='quadratic'):

        self.x_points = x
        self.x = np.arange(min(bounds), max(bounds), resolution)

        self.log = log
        self.kind = kind
        self.zero_bounds = zero_bounds

        if self.zero_bounds is not None:
            tmp_x = np.zeros(len(self.x_points)+2)
            tmp_x[0] = min(bounds)
            tmp_x[-1] = max(bounds)

            tmp_x[1:-1] = self.x_points
            self.x_points = tmp_x

            self.empty_theta = np.zeros(len(self.x_points))
            self.empty_theta[0] = zero_bounds
            self.empty_theta[-1] = zero_bounds

    def __call__(self, theta, *args, **kwargs):

        if self.zero_bounds is not None:

            self.empty_theta[1:-1] = theta

            theta = self.empty_theta

        assert len(theta) == len(self.x_points), 'len(theta) != len(self.x)' + \
                                             str(len(theta)) + ' ' + str(len(self.x_points))

        # get_new_profile = InterpolatedUnivariateSpline(self.x, theta)
        # get_new_profile = interp1d(self.x_points, theta, self.kind)
        get_new_profile = UnivariateSpline (self.x_points, theta)

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

    from tulasa import general

    import time as clock

    x = [-5, -2, -1, -0.6, 0, 1, 3]

    linear = MeshLine(x, zero_bounds=True, kind='linear')
    quad = MeshLine(x, zero_bounds=True, kind='quadratic')

    theta = np.ones(len(x))

    start_time = clock.time()

    i = 0
    while i < 100:
        linear(theta)
        i+=1

    second_time = clock.time()

    i = 0
    while i < 100:
        quad(theta)
        i += 1

    third_time = clock.time()

    lin_time = second_time - start_time
    quad_time = third_time - second_time

    print(lin_time, quad_time)
