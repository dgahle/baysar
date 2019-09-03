from numpy import square, sqrt, mean, linspace, nan, log, diff, arange, zeros, concatenate, where

from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import sparse
# import numpy as np
import time as clock

import os, sys, io, warnings


from baysar.linemodels import XLine, NoADASLines, ADAS406Lines, BalmerHydrogenLine
from baysar.lineshapes import GaussiansNorm

def clip_data(x_data, y_data, x_range):

    '''
    returns sections of x and y data that is within the desired x range

    :param x_data:
    :param y_data:
    :param x_range:
    :return:
    '''

    x_index = where((min(x_range) < x_data) & (x_data < max(x_range)))


    try:
        return x_data[x_index], y_data[x_index]
    except:
        print(x_index)

        raise


def within(points, bounds):

    if type(points) == str:
        try:
            float(points)
        except ValueError:
            # print(ValueError)
            return False
        except:
            raise
    elif type(points) == list:
        try:
            _ = [float(p) for p in points]
        except ValueError:
            # print(ValueError)
            return False
        except:
            raise
        pass
    else:
        pass


    try:
        try:
            return any([min(bounds) < p and p < max(bounds) for p in points])
        except TypeError:
            return any([min(bounds) < float(p) and float(p) < max(bounds) for p in points])
        except:
            raise
    except TypeError:
        return any([min(bounds) < points and points < max(bounds)])
    except:
        raise


class SpectrometerChord(object):

    """
    The SpectrometerChord object is meant to be a soft simulation of a spectrometer with wa single
    line of sight. From the input dictionary the error (Guassian noise and Poisson) of the emission
    is calculated and allows for the calculation of the likelihood of a give set of plasma parameters

    """

    def __init__(self, plasma, refine=None, chord_number=None):

        self.chord_number = chord_number

        self.plasma = plasma

        self.input_dict = self.plasma.input_dict['chords'][self.chord_number]

        if 'refine' in list(self.input_dict['data'].keys()):
            self.refine = self.input_dict['data']['refine']
        else:
            self.refine = refine

        self.calibrated = not self.plasma.is_chord_not_calibrated[self.chord_number]

        self.y_data = self.input_dict['data']['experimental_emission']

        if self.refine is not None:

            assert any([type(self.refine) == t for t in (int, float)]), 'self.refine is not a int or float'

            self.x_data_exp = self.input_dict['data']['wavelength_axis']

            low, up = min(self.x_data_exp), max(self.x_data_exp)
            self.x_data = linspace(low, up, int( abs(low - up) / self.refine ))
        else:
            self.x_data = self.input_dict['data']['wavelength_axis']

        self.instrument_function = self.input_dict['data']['instrument_function']
        self.instrument_function_matrix = self.int_func_sparce_matrix() # self.instrument_function, len(self.y_data))

        self.noise_region = self.input_dict['data']['noise_region']

        self.a_cal = self.input_dict['data']['emission_constant']

        self.get_error()

        self.get_lines()

        pass

    def __call__(self, theta, *args, **kwargs):
         # TODO: do this in the posterior not in the spectrometer chord

        return self.likelihood()

    def get_noise_spectra(self):

        if self.refine is None:
            self.x_data_continuum, self.y_data_continuum = clip_data(self.x_data, self.y_data, self.noise_region)
        else:
            self.x_data_continuum, self.y_data_continuum = clip_data(self.x_data_exp, self.y_data, self.noise_region)

    def get_error(self, fake_cal=False):

        self.get_noise_spectra()

        try:
            if fake_cal:
                self.error = sqrt( square( mean( self.y_data_continuum ) ) + # noise
                                   self.y_data * self.a_cal + # poisson
                                   square(self.y_data * 0.2) ) # fake & TEMP calibration error
            else:
                self.error = sqrt(square(mean(self.y_data_continuum)) +  # noise
                                  self.y_data * self.a_cal) # poisson
        except TypeError:
            print(self.y_data_continuum)
            raise
        except:
            raise

    def likelihood(self, cauchy=True):

        # try:
        fm = self.forward_model() # * calibration_fractor

        assert len(fm) == len(self.y_data), 'len(fm) != len(self.y_data)'

        if cauchy:
            likeli_in = 1 + ( (self.y_data - fm) / self.error ) ** 2
            # likeli_in = 1 / (pi * self.error * likeli_in)
            likeli_in = 1 / likeli_in

            likeli = sum( log(likeli_in) )

        else:
            likeli = - 0.5 * sum( ( (self.y_data - fm) / self.error) ** 2 )

        assert likeli != nan, 'likeli == nan'

        return likeli
        # except ValueError:
        #     print(self.y_data.shape, self.forward_model().shape, self.error.shape)
        #     raise
        # except:
        #     raise

    def forward_model(self, dont_interpolate=False):

        # tmp_time = clock.time()

        spectra = [ l().flatten() for l in self.lines ] # TODO: This is what takes all the time

        # print( (clock.time() - tmp_time) * 1e3, ' ms' )
        # print( 'Number of lines: ', len(self.lines) )


        # spectra = sum( sum( spectra ) )
        spectra = sum(spectra)

        # assert all([s != None for s in spectra]) , 'spectra == None'
        # assert type( spectra[0] ) != list, ''
        # print(spectra)

        # z_effective = ones( len(self.plasma.plasma_state['electron_density']) )
        #
        # continuum = continuo(electron_temperature=self.plasma.plasma_state['electron_temperature'],
        #                      electron_density=self.plasma.plasma_state['electron_density'],
        #                      wavelength=self.x_data, z_effective=z_effective,
        #                      path_length=diff(self.plasma.profile_fucntion.x)[0])

        self.continuum = self.plasma.plasma_state['intercept'] # + continuum


        if self.calibrated:
            spectra = spectra + self.continuum # + sum(continuum]
        else:
            tmp_a_cal = self.plasma.plasma_state['a_cal'][self.chord_number]

            spectra = (spectra / tmp_a_cal) + self.continuum


        try:
            spectra = self.instrument_function_matrix.dot(spectra)

            # print('self.refine', self.refine)

            if self.refine is not None:

                if dont_interpolate:
                    return spectra

                spectra = self.instrument_function_matrix.dot(spectra)

                spectra_interp = interp1d(self.x_data, spectra)

                spectra = spectra_interp(self.x_data_exp)

                assert len(spectra) == len(self.y_data), 'len(spectra) != len(self.y_data)'

                return spectra

            else:
                return self.instrument_function_matrix.dot(spectra)

        except (ValueError, AttributeError):
            try:
                with warnings.catch_warnings():

                    warnings.simplefilter("ignore")

                    spectra = fftconvolve(spectra, self.instrument_function, mode='same')

                if self.refine is not None:

                    spectra_interp = interp1d(self.x_data, spectra)

                    spectra = spectra_interp(self.x_data_exp)

                if any([t < 0 for t in spectra]):
                    return abs(spectra)
                else:
                    return spectra
            except ValueError:
                print(ValueError)
                print('in1 and in2 shape', spectra.shape, self.instrument_function.shape)
            except FutureWarning:
                print(FutureWarning)
                print('There was a Future warning about '
                      'scipy.signal.fftconvolve() that is being ignored')
        except:
            raise

    def get_lines(self):

        self.lines = []

        for counter0, isotope in enumerate(self.plasma.input_dict['physics'].keys()):

            if isotope == 'X':

                # for counter1, line in enumerate(self.plasma.input_dict[isotope].keys()):  # ['lines']):
                for line in self.plasma.input_dict['physics'][isotope].keys():

                    try:
                        tmp_check = float(line)
                    except ValueError:
                        tmp_check = line.split()
                    except TypeError:
                        print(line)
                        raise
                    except:
                        raise

                    if within(tmp_check , self.x_data):

                        try:
                            tmp_cwl = self.plasma.input_dict['physics'][isotope][line]['wavelength']
                        except (KeyError, TypeError):
                            print( self.plasma.input_dict['physics'][isotope].keys() )
                            print( isotope, line )
                            print( type(isotope), type(line) )
                            raise
                        except:
                            raise

                        try:
                            tmp_fractions = self.plasma.input_dict['physics'][isotope][line]['fractions']
                        except KeyError:
                            tmp_fractions = []
                        except:
                            raise

                        tmp_wavelengths = self.x_data
                        tmp_lineshape = GaussiansNorm

                        tmp_line = XLine(cwl=tmp_cwl, fwhm=diff(tmp_wavelengths)[0],
                                         plasma=self.plasma.plasma_state, wavelengths=tmp_wavelengths,
                                         lineshape=tmp_lineshape, species=isotope, fractions=tmp_fractions)

                        self.lines.append(tmp_line)

                    else:

                        pass

                    pass

                pass

            elif any([isotope == i for i in ('H', 'D', 'T')]):

                for counter1, ion in enumerate(self.plasma.input_dict['physics'][isotope]['ions']):

                    for counter2, line in enumerate(self.plasma.input_dict['physics'][isotope][ion].keys()): # ['lines']):

                        try:
                            tmp_check = float(line)
                        except ValueError:
                            tmp_check = line.split()
                        except:
                            raise

                        if within(tmp_check, self.x_data):

                            tmp_cwl = self.plasma.input_dict['physics'][isotope][ion][line]['wavelength']

                            tmp_wavelengths = self.x_data

                            n_upper = self.plasma.input_dict['physics'][isotope][ion][line]['n_upper']
                            tmp_ma = self.plasma.input_dict['physics'][isotope]['atomic_mass']

                            tmp_pec = [ self.plasma.input_dict['physics'][isotope][ion][line]['exc_tec'] , \
                                        self.plasma.input_dict['physics'][isotope][ion][line]['rec_pec'] ]

                            tmp_line = BalmerHydrogenLine(cwl=tmp_cwl, wavelengths=tmp_wavelengths, n_upper=n_upper,
                                                          atomic_mass=tmp_ma, pec=tmp_pec, species=isotope, ion=ion,
                                                          plasma=self.plasma.plasma_state)

                            # tmp_line = HydrogenLineShape(cwl, wavelengths, n_upper, n_lower, atomic_mass, zeeman=True)

                            self.lines.append(tmp_line)

                            pass

                        pass

                    pass

            else:

                for counter1, ion in enumerate(self.plasma.input_dict['physics'][isotope]['ions']):

                    for counter2, line in enumerate(self.plasma.input_dict['physics'][isotope][ion].keys()): # ['lines']):

                        try:

                            try:
                                tmp_check = float(line)
                            except ValueError:
                                tmp_check = line.replace(',', '')[1:-1]
                                tmp_check = tmp_check.split()
                            except:
                                raise

                            if within(tmp_check, self.x_data):

                                '''
                                cwl, wavelengths, lineshape, atomic_mass, tec406, length
                                '''

                                tmp_cwl = self.plasma.input_dict['physics'][isotope][ion][line]['wavelength']

                                tmp_wavelengths = self.x_data
                                tmp_lineshape = GaussiansNorm


                                tmp_ma = self.plasma.input_dict['physics'][isotope]['atomic_mass']
                                tmp_jj_frac = self.plasma.input_dict['physics'][isotope][ion][line]['jj_frac']

                                try:
                                    tmp_tec = self.plasma.input_dict['physics'][isotope][ion][line]['tec']


                                    # tmp_line = ADAS406Line(cwl=tmp_cwl, wavelengths=tmp_wavelengths, lineshape=tmp_lineshape,
                                    #                        atomic_mass=tmp_ma, tec406=tmp_tec, species=isotope,
                                    #                        ion=ion, plasma=self.plasma.plasma_state, jj_frac=tmp_jj_frac)

                                    tmp_line = ADAS406Lines(cwls=tmp_cwl, wavelengths=tmp_wavelengths, lineshape=tmp_lineshape,
                                                            atomic_mass=tmp_ma, tec406=tmp_tec, species=isotope,
                                                            ion=ion, plasma=self.plasma.plasma_state, jj_frac=tmp_jj_frac)


                                except KeyError:
                                    tmp_line = NoADASLines(cwl=tmp_cwl, wavelengths=tmp_wavelengths, lineshape=tmp_lineshape,
                                                           atomic_mass=tmp_ma, species=isotope, ion=ion,
                                                           plasma=self.plasma.plasma_state, jj_frac=tmp_jj_frac)

                                except:
                                    raise

                                self.lines.append(tmp_line)

                            else:

                                pass

                        except ValueError:

                            if line == 'lines':

                                pass

                            else:
                                print(self.plasma.input_dict['physics'][isotope][ion].keys())
                                print(isotope, ion, line)
                                print("Unexpected error:", sys.exc_info())  # [0])
                                raise



                        else:

                            pass

                    pass

                pass

            pass

        pass

    def int_func_sparce_matrix(self, theta=None):

        if theta is not None:
            int_func, res = theta
        else:
            if self.refine is not None:

                self.instrument_function_x = arange( len(self.instrument_function) )
                int_func_interp = interp1d(self.instrument_function_x, self.instrument_function)

                k = abs( mean( diff(self.x_data_exp) ) / mean( diff(self.x_data) ) )
                self.instrument_function_inter_x = arange(min(self.instrument_function_x),
                                                             max(self.instrument_function_x), 1/k)

                int_func = int_func_interp(self.instrument_function_inter_x)

                self.instrument_function_inter_y = int_func

                res = len(self.x_data)

            else:
                int_func, res = (self.instrument_function, len(self.y_data))

        # res = len(data)
        shape = (res, res)
        matrix = zeros(shape)
        # matrix = sparse.lil_matrix(shape)
        buffer = zeros(res)

        long_int_func = concatenate( (buffer, int_func[::-1], buffer) )
        # long_int_func = concatenate((buffer + int_func[0], int_func, buffer + int_func[-1]))

        rang = arange(res)[::-1] + int( (len(int_func) + 1) / 2 ) # 16

        # self.k_matrix =

        for i, key_i in enumerate(rang):
            # # print(i, key_i)
            # try:
            #     matrix_i = long_int_func[key_i:key_i + res]  # matrix_i # [::-1]
            # except:
            #     print(long_int_func.shape)
            #     print(i, res, key_i)
            #     raise

            matrix_i = long_int_func[key_i:key_i + res]  # matrix_i # [::-1]
            k = sum(matrix_i)

            # print('k', k)

            if k == 0: k = 1

            # print(len(matrix_i))
            matrix[i, :] = matrix_i / k

        matrix = sparse.csc_matrix(matrix)
        # matrix = matrix.tocsc()

        # return matrix.T

        # to do convoutions output = matrix.dot(input)

        return matrix

    def make_matrix(self, array):

        matrix = array

        return matrix



if __name__=='__main__':

    pass

    # pass
    #
    # # input files
    # input_dict = {}
    # num_chords = 1
    #
    # input_dict['number_of_chords'] = num_chords
    #
    # input_dict['chords'] = {}
    #
    # wavelength_axes = [[]]
    # experimental_emission = [[]]
    #
    # instrument_function = [[]]
    # emission_constant = [...]
    # noise_region = [[]]
    #
    # for counter0, chord in enumerate(arange(num_chords)):
    #     # tmp = 'chord' + str(counter0)
    #     tmp = counter0
    #
    #     input_dict['chords'][tmp] = {}
    #     input_dict['chords'][tmp]['meta'] = {}
    #
    #     input_dict['chords'][tmp]['meta']['wavelength_axis'] = wavelength_axes[counter0]
    #     input_dict['chords'][tmp]['meta']['experimental_emission'] = experimental_emission[counter0]
    #
    #     input_dict['chords'][tmp]['meta']['instrument_function'] = instrument_function[counter0]
    #     input_dict['chords'][tmp]['meta']['emission_constant'] = emission_constant[counter0]
    #     input_dict['chords'][tmp]['meta']['noise_region '] = noise_region[counter0]
    #
    #     pass
    #
    # '''
    # n_ii_cwls = [3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92]
    # n_ii_jjr = [1, 0.92, 0.08, 0.456, 0.211, 0.197, 0.026, 0.022]
    # # n_ii_pec_keys = [0, 2, 2, 3, 3, 3, 3, 3]
    # n_ii_pec_keys = [3, 6, 6, 7, 7, 7, 7, 7]
    #
    # # n_iii_cwls = [3998.63, 4003.58, 4097.33, 4103.34]
    # # n_iii_jjr = [0.375, 0.625, 0.665, 0.335]
    # # n_iii_pec_keys = [1, 1, 4, 4]
    # '''
    #
    # chord0_dict = {}
    #
    # species = ['D', 'N']
    # ions = [['0'], ['1', '2']]
    #
    # cwl = [[[3968.99, 4100.58]],
    #        [[3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92],
    #         [3998.63, 4003.58, 4097.33, 4103.34]]]
    # n_pec = [[[0, 0]],
    #          [[0, 0, 0, 0, 0, 0, 0, 0],
    #           [0, 0, 0, 0]]]  # TODO need the real values
    # f_jj = [[[3968.99, 4100.58]],
    #         [[3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92],
    #          [3998.63, 4003.58, 4097.33, 4103.34]]]  # TODO need the real values
    #
    # atomic_charge = [1, 7]
    # ma = [2, 14]
    #
    # for counter0, isotope in enumerate(species):
    #
    #     # if counter0 == 0:
    #     #     input_dict['isotopes'] = species
    #
    #     chord0_dict[isotope] = {}
    #     chord0_dict[isotope]['atomic_mass'] = ma[counter0]
    #     chord0_dict[isotope]['atomic_charge'] = atomic_charge[counter0]
    #
    #     for counter1, ion in enumerate(ions[counter0]):
    #
    #         if counter1 == 0:
    #             chord0_dict[isotope]['ions'] = ions[counter0]
    #
    #         chord0_dict[isotope][ion] = {}
    #
    #         for counter2, line in enumerate(cwl[counter0][counter1]):
    #
    #             if counter2 == 0:
    #                 chord0_dict[isotope][ion]['lines'] = []
    #
    #             chord0_dict[isotope][ion]['lines'].append(str(line))
    #
    #             chord0_dict[isotope][ion][str(line)] = {}
    #
    #             chord0_dict[isotope][ion][str(line)]['wavelength'] = line
    #             chord0_dict[isotope][ion][str(line)]['pec_key'] = n_pec[counter0][counter1][counter2]
    #             chord0_dict[isotope][ion][str(line)]['jj_frac'] = f_jj[counter0][counter1][counter2]
    #
    #             chord0_dict[isotope][ion][str(line)]['tec'] = tmp_func
    #             chord0_dict[isotope][ion][str(line)]['tec_file'] = None
    #
    #             pass
    #
    #         pass
    #
    #     pass
    #
    # input_dict['chords'][0]['physics'] = chord0_dict
    #
    # # from tulasa import general
    # #
    # # general.save(input_dict, filename='./input_dict.p')
    #
    # # interpolaters = {}
    # # interpolaters[0] = tmp_func()
    #
    # # make plasma object
    # from BaySAR.BaySAR.plasmas import PlasmaLine
    #
    # # from BaySAR.BaySAR.lineshapes import Gaussian, GaussianNorm
    #
    # profile_funciton = Gaussian(x=arange(-50., 50., 3))
    #
    # plasma = PlasmaLine(input_dict=input_dict['chords'][0]['physics'],
    #                     profile_fucntion=profile_funciton, profile_fucntion_num_varriables=3)
    # # plasma()
    #
    # instrument_function = GaussianNorm(cwl=15, x=arange(31))
    # instrument_function = instrument_function([10, 1])
    #
    # a_cal = 1e11
    #
    # continuum = [3960, 3990]
    # x_data = arange(3950, 4150, 0.18)
    # y_data = zeros( len( x_data ) )
    #
    # for tmp in arange( len(y_data) ):
    #     y_data[tmp] = a_cal + random.rand() * a_cal
    #
    # theta = [1e12,
    #          -5, 30, 1e13,
    #          5, 15,
    #          0.8, 0.2,
    #          1e-4, 1e-4,
    #          5, 10]
    #
    # plasma(theta)
    #
    # chord = SpectrometerChord(plasma=plasma, input_file=input_dict['chords'][0],
    #                           instrument_function=instrument_function, a_cal=a_cal,
    #                           x_data=x_data, y_data=y_data, continuum=continuum)
    #
    # from tulasa import general
    #
    # general.plot([chord.y_data, chord.forward_model()], [chord.x_data, chord.x_data], multi='fake')
    #
    # print( chord(theta=theta) )
    #
    # def time_eval(num=500):
    #
    #     from time import time
    #
    #     tmp_time = time()
    #
    #     for tmp in arange(num):
    #         chord(theta=theta)
    #
    #     print(str(num) + ' chord evaluations', time()-tmp_time)
    #     print('average time', (time()-tmp_time) / num)
    #
    # time_eval()
    #
    # pass