import sys, warnings

import numpy as np

from scipy import special
from scipy.constants import pi
from scipy.integrate import trapz
from scipy.interpolate import interp1d




class Eich(object):

    def __init__(self, x=None, cwl=None, vectorise=1, height_norm=True, reduced=False, centre=True):

         # Useful constants
        # self.fwhm_to_sigma = 1 / np.sqrt(8 * np.log(2))

        self.vectorise = vectorise

        if self.vectorise > 1:
            self.x = np.tile(x, (vectorise, 1))
        else:
            self.x = x

        self.cwl = cwl

        self.height_norm = height_norm
        self.reduced = reduced
        self.centre = centre

    def __call__(self, theta, *args, **kwargs):

        if self.reduced:
            if self.cwl is not None:
                spreding, hieght = theta
                lambda_q, s_factor, flux_expansion = spreding, spreding, 1 # spreding

                centre = self.cwl
            else:
                centre, spreding, hieght = theta
                lambda_q, s_factor, flux_expansion = spreding, spreding, 1 # spreding
                pass
        else:
            if self.cwl is not None:
                lambda_q, s_factor, hieght = theta
                centre = self.cwl
                flux_expansion = 1
                pass
            else:
                centre, lambda_q, s_factor, hieght = theta
                flux_expansion = 1

        if self.centre:
            exp_in = np.square(s_factor / (2 * lambda_q)) - \
                     (0 - self.x) / (lambda_q * flux_expansion)
            exp_part = np.exp(exp_in)

            erfc_in = s_factor / (2 * lambda_q) - (0 - self.x) / (s_factor * flux_expansion)
            erfc_part = special.erfc(erfc_in)

            tmp_eich = exp_part * erfc_part

            try:
                centre_index = np.where(tmp_eich==max(tmp_eich))
            except:
                print(tmp_eich)
                raise

            dr = centre - self.x[centre_index]

            try:
                x_prime = self.x + dr
            except:
                print(dr, centre_index, any(np.isnan(tmp_eich)))

            eich_interp1d = interp1d(x_prime, tmp_eich, fill_value='extrapolate')

            tmp_eich = eich_interp1d(self.x)

            # old_centre = self.x[]

        else:
            exp_in = np.square(s_factor / (2 * lambda_q)) - \
                     (centre - self.x) / (lambda_q * flux_expansion)
            exp_part = np.exp(exp_in)

            erfc_in = s_factor / (2 * lambda_q) - (centre - self.x) / (s_factor * flux_expansion)
            erfc_part = special.erfc(erfc_in)

            tmp_eich = exp_part * erfc_part


        if self.height_norm:
            return hieght * tmp_eich / max(tmp_eich)
        else:
            # This is area normalised
            return hieght * tmp_eich / sum(tmp_eich)

        ...


class Gaussian(object):

    def __init__(self, x=None, cwl=None, vectorise=1):

        # Useful constants
        self.fwhm_to_sigma = 1 / np.sqrt(8 * np.log(2))

        self.vectorise = vectorise

        if self.vectorise > 1:
            self.x = np.tile(x, (vectorise, 1))
        else:
            self.x = x

        self.cwl = cwl



    def __call__(self, theta, *args, **kwargs):

        if self.cwl is not None:
            fwhm, intensity = theta
            cwl = self.cwl
            pass
        else:
            try:
                cwl, fwhm, intensity = theta
            except ValueError:
                print(theta)
                raise
            except:
                raise

        # try:
        #     fwhm /= 2
        # except:
        #     fwhm = [f/2 for f in fwhm]


        if self.vectorise > 1:

            if any([tmp_fwhm != np.mean(fwhm) for tmp_fwhm in fwhm]):
                # tiling
                fwhm = np.array([np.zeros(len(self.x[0])) + t for t in fwhm])

                try:
                    intensity = np.array([np.zeros(len(self.x[0])) + t for t in intensity])
                except ValueError:
                    intensity = np.array([np.zeros(len(self.x[0])) + t for t in intensity[0]])
                except:

                    print(intensity)

                    raise

                # print(self.x.shape)
                # print(self.x)
                #
                # intensity = np.array([np.zeros(len(self.x[0])) + t for t in intensity])

            else:

                # print( len(intensity), len(intensity[0]) )

                intensity = sum(intensity)

                sigma = np.mean(fwhm) * self.fwhm_to_sigma

                peak = np.exp(-0.5 * ( (self.x[0] - cwl) / sigma) ** 2)

                # print('hello')

                try:
                    final_peak = intensity * peak.flatten() # np.array([ peak for counter in np.arange(self.vectorise) ])
                    return final_peak # .flatten()
                except ValueError:
                    final_peak = sum(intensity) * peak.flatten() # np.array([peak for counter in np.arange(self.vectorise)])
                    return final_peak# .flatten()
                except:
                    raise

        else: pass

        # guassian function
        try:
            sigma = fwhm * self.fwhm_to_sigma
        except TypeError:
            sigma = np.array(fwhm) * self.fwhm_to_sigma
        except TypeError:
            print('fwhm', fwhm)
            print('self.fwhm_to_sigma', self.fwhm_to_sigma)
            raise
        except:
            print("Unexpected error:", sys.exc_info())  # [0])
            raise

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            peak = np.exp(-0.5 * ( (self.x - cwl) / sigma) ** 2)

        try:
            return intensity * peak
        except ValueError:
            print( len(intensity), len(peak) )
            raise
        except:
            raise

    def grad(self, theta):

        if self.cwl is not None:
            fwhm, intensity = theta
            cwl = self.cwl
            pass
        else:
            try:
                cwl, fwhm, intensity = theta
            except ValueError:
                print(theta)
                raise
            except:
                raise

        sigma = fwhm * self.fwhm_to_sigma

        d_by_d_sigm = -2 * np.square(self.x)

        pass


class Gaussians(object):

    def __init__(self, x=None, cwl=None, vectorise=1):

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


class Lorentzian(object): # TODO: This has not been thuroughly debugged


    def __init__(self, x, cwl):

        self.x = x

        self.x_shape = x.shape

        self.cwl = np.tile(cwl, self.x_shape[1])

    def __call__(self, fwhm, intensity):

        # tiling
        fwhm = np.tile(fwhm, self.x_shape[1])
        intensity = np.tile(intensity, self.x_shape[1])

        hwhm = fwhm / 2.0

        return intensity / (pi * hwhm * (1 + ((self.x - self.cwl) / hwhm) ** 2))


class LorentzianNorm(Lorentzian):

    def __init__(self, x, cwl):

        super(LorentzianNorm, self).__init__(x, cwl)

    def __call__(self, fwhm, intesity):

        peak = super(LorentzianNorm, self).__call__(fwhm, intesity)

        norm_k = trapz(peak, self.x)

        return peak / norm_k


class TmpLorentzian(object):

    def __init__(self, x, cwl=None, fwhm=None):

        self.x = x

        self.cwl = cwl
        self.fwhm = fwhm

        pass

    def __call__(self, theta):

        if all([ c == None for c in [self.fwhm, self.cwl] ]):

            cwl, fwhm, intensity = theta

        elif self.fwhm is not None and self.cwl is None:

            cwl, intensity = theta

            fwhm = self.fwhm

        elif self.cwl is not None and self.fwhm is None:

            fwhm, intensity = theta

            cwl = self.cwl

        else:
            intensity = theta

            fwhm = self.fwhm

        try:
            hwhm = fwhm / 2.0
        except:
            print(fwhm)
            raise

        peak = 1 / (pi * hwhm * (1 + ((self.x - cwl) / (hwhm)) ** 2))

        return (intensity / max(peak)) * peak


class TmpLorentzianNorm(TmpLorentzian):

    def __call__(self, theta):

        peak = super().__call__(theta)

        k = trapz(peak, self.x)

        peak_norm = peak / k

        return theta[1] * peak_norm


class DoubleTmpLorentzian(TmpLorentzian):

    def __call__(self, theta):

        if self.cwl is None:
            cwl, w0, w1, tune, h = theta

            l0_theta = [cwl, w0, h]
            l1_theta = [cwl, w1, h]
        else:
            w0, w1, tune, h = theta

            l0_theta = [w0, h]
            l1_theta = [w1, h]

        peak0 = super().__call__(l0_theta)
        peak1 = super().__call__(l1_theta)

        return (peak0 + tune*peak1) / (1 + tune)

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

def supergaussian(x, theta):

    mean, sigma, half_power = theta

    inside = (x - mean) / sigma

    log_peak = -0.5 * np.power(inside, 2 * half_power)

    return np.exp(log_peak)

from scipy import interpolate

from scipy.interpolate import InterpolatedUnivariateSpline


class MeshLine(object):

    def __init__(self, x, bounds=[-10, 10], zero_bounds=None, resolution=0.1, kind='quadratic'):

        self.x_points = x
        self.x = np.arange(min(bounds), max(bounds), resolution)

        self.kind = kind
        self.zero_bounds = zero_bounds

        if self.zero_bounds is not None:
            tmp_x = np.zeros(len(self.x_points)+2)
            tmp_x[0] = min(bounds)
            tmp_x[-1] = max(bounds)

            tmp_x[1:-1] = self.x_points
            self.x_points = tmp_x

            self.empty_theta = np.zeros(len(self.x_points))
            self.empty_theta[0] = self.zero_bounds
            self.empty_theta[-1] = self.zero_bounds

    def __call__(self, theta, *args, **kwargs):

        if self.zero_bounds is not None:

            self.empty_theta[1:-1] = theta

            theta = self.empty_theta

        assert len(theta) == len(self.x_points), 'len(theta) != len(self.x) ' + \
                                             str(len(theta)) + ' ' + str(len(self.x_points))

        # get_new_profile = InterpolatedUnivariateSpline(self.x, theta)
        get_new_profile = interp1d(self.x_points, theta, self.kind)

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
