import numpy as np

import os, sys, io
import copy

from baysar.lineshapes import Gaussian, Eich
from baysar.linemodels import build_tec406

from baysar.spectrometers import within

class PlasmaLine():

    """
    PlasmaLine is a object that holds the simulated 1D plasma profiles and the function for updating the
    plamsa profiles. The plasma paramaters and profiles are also organised into a dictionary which are
    accessible to the line objects in the initialised spectrometer objects. This in essence is the parameter
    handler which handles the interface between the sampler and the forward model.
    """

    def __init__(self, input_dict, profile_function=None, conc=True, logte=False, netau=False):

        self.netau = netau
        self.conc = conc
        self.logte = logte

        self.input_dict = input_dict

        self.plasma = input_dict

        self.inference_resolution = self.input_dict['inference_resolution']

        if profile_function is None:
            # TODO: x range for arange needs to be a top level variable
            los_x = np.arange(-15., 15., 0.2)
            self.profile_function = Eich(x=los_x, reduced=True)
            self.profile_function.number_of_varriables = 3
            self.profile_function.dr = True

            self.profile_function_num_varriables = self.profile_function.number_of_varriables

            # self.profile_function = Gaussian( x )
            # self.profile_function = Gaussian( x=np.arange(-30., 30., 3) )

        else:
            self.profile_function = profile_function # ...
            self.profile_function_num_varriables = self.profile_function.number_of_varriables

        self.count_species()

        self.no_data_lines = self.where_the_no_data_lines()
        self.build_impurity_average_charge()

        self.plasma_state = {}

        self.hydrogen_isotopes = ['H', 'D', 'T']
        self.hydrogen_isotope = any(['H' == t or 'D' == t or 'T' == t
                                     for t in list(self.input_dict['physics'].keys())])

        self.build_plasma()
        # self.update_plasma(np.zeros(int(1e3)))

        self.plasma_state['los'] = self.profile_function.x

        # self.theta_bounds = self.chain_bounds()
        self.chain_bounds()

    def __call__(self, theta):

        return self.update_plasma(theta)

    def build_plasma(self):


        self.all_the_indicies = {}
        self.theta_slices = {}

        # check is the chords are calibrated
        self.calibration_constants = []

        for tmp_key in self.input_dict['chords']:

            tmp_a_cal = self.input_dict['chords'][tmp_key]['data']['emission_constant']

            self.calibration_constants.append(tmp_a_cal)

        cc_condition = np.array([(cc == 1) for cc in self.calibration_constants])

        self.is_chord_not_calibrated = cc_condition

        if any(cc_condition):

            self.all_the_indicies['calibration_index'] = 0

            intercept_index = len( np.where(cc_condition == True)[0] )

            self.theta_slices['calibration'] = slice(0, intercept_index)

        else:
            intercept_index = 0

        self.all_the_indicies['intercept_index'] = intercept_index
        num_chords = 1 #TODO: needs to be updated so there is one per chord
        self.theta_slices['intercept'] = slice(intercept_index, intercept_index+num_chords)

        # defining indicies for electron density and temperature profiles
        self.all_the_indicies['ne_index'] = intercept_index + 1
        self.all_the_indicies['te_index'] = self.all_the_indicies['ne_index'] + \
                                            self.profile_function_num_varriables

        self.theta_slices['electron_density'] = slice(self.theta_slices['intercept'].stop,
                                                      self.theta_slices['intercept'].stop +
                                                      self.profile_function_num_varriables)

        self.theta_slices['electron_temperature'] = slice(self.theta_slices['electron_density'].stop,
                                                          self.theta_slices['electron_density'].stop +
                                                          self.profile_function_num_varriables)

        # defining indicies for viewing-angle, b-field and concentration
        if self.hydrogen_isotope:
            if self.profile_function.dr:
                k_dr = 1
            else:
                k_dr = 0

            self.theta_slices['b-field'] = slice(self.theta_slices['electron_temperature'].stop,
                                                 self.theta_slices['electron_temperature'].stop + 1)
            self.theta_slices['viewangle'] = slice(self.theta_slices['b-field'].stop,
                                                   self.theta_slices['b-field'].stop + 1)
            # self.theta_slices['conc'] = slice(self.theta_slices['viewangle'].stop,
            #                                        self.theta_slices['b-field'].stop + 1)


            self.all_the_indicies['b_index'] = self.all_the_indicies['te_index'] + \
                                               self.profile_function_num_varriables - k_dr

            self.all_the_indicies['view_angle_index'] = self.all_the_indicies['b_index'] + 1
            self.all_the_indicies['conc_index'] = self.all_the_indicies['view_angle_index'] + 1

            self.all_the_indicies['upper_te_index'] = self.all_the_indicies['b_index']

            self.plasma_state['B-field'] = None # theta[b_index]
            self.plasma_state['view_angle'] = None # theta[view_angle_index]
        else:
            if self.profile_function.dr:
                k_dr = 1
            else:
                k_dr = 0

            self.all_the_indicies['conc_index'] = self.all_the_indicies['te_index'] + \
                                                  self.profile_function_num_varriables - k_dr

            self.all_the_indicies['upper_te_index'] = self.all_the_indicies['conc_index']

        self.theta_slices['conc'] = slice(self.theta_slices['viewangle'].stop, ...)


        # defining indicies for taus and ion temperatures
        if self.inference_resolution['ion_resolved_tau']:
            self.all_the_indicies['ti_index'] = self.all_the_indicies['conc_index'] + \
                                                self.number_of_ions
        else:
            self.all_the_indicies['ti_index'] = self.all_the_indicies['conc_index'] + \
                                                self.number_of_isotopes

        if self.inference_resolution['ion_resolved_temperatures']:
            self.all_the_indicies['tau_index'] =  self.all_the_indicies['ti_index'] + \
                                                  self.number_of_ions
        else:
            self.all_the_indicies['tau_index'] =  self.all_the_indicies['ti_index'] + \
                                                  self.number_of_isotopes

        # defining indicies for no data lines
        if 'D' in list(self.input_dict['physics'].keys()):
            k_d_exists = 1
        else:
            k_d_exists = 0

        if self.inference_resolution['ion_resolved_tau']:
            num_taus = self.number_of_ions
        else:
            num_taus = self.number_of_isotopes

        self.all_the_indicies['no_data_index'] = self.all_the_indicies['tau_index'] + \
                                                 num_taus - k_d_exists

        self.all_the_indicies['x_index'] = self.all_the_indicies['no_data_index'] + \
                                           self.number_of_no_data_lines

        if any(cc_condition):
            self.plasma_state['a_cal'] = []

        self.plasma_state['intercept'] = []

        self.plasma_state['electron_density'] = None
        self.plasma_state['electron_temperature'] = None

        x_param = 'conc*tec*jj_frac'

        for tmp_isotope in self.input_dict['physics'].keys():

            if tmp_isotope == 'X':
                self.plasma_state[tmp_isotope] = {}
                for countx, tmpx in enumerate(self.plasma['physics'][tmp_isotope].keys()): # ['lines']:
                    self.plasma_state[tmp_isotope][tmpx] = {}
                    self.plasma_state[tmp_isotope][tmpx][x_param] = None

            elif any([tmp_isotope == iso for iso in self.hydrogen_isotopes]):
                self.plasma_state[tmp_isotope] = {'0' : {'conc': None, 'ti': None} }

            else:
                self.plasma_state[tmp_isotope] = {}
                for tmp_ion in self.input_dict['physics'][tmp_isotope]['ions']:
                    self.plasma_state[tmp_isotope] = {tmp_ion: {'conc': None, 'ti': None, 'tau': None} }

            # update non-physics parameters
            #
            # check if the species has a nodata line
            if tmp_isotope in self.no_data_lines.keys():

                # check if the ion has a nodata line
                for tmp_ion in self.no_data_lines[tmp_isotope].keys():

                    # check that the ion has a subdict
                    if tmp_ion not in self.plasma_state[tmp_isotope].keys():
                        self.plasma_state[tmp_isotope][tmp_ion] = {}

                    # update the 'conc*tec*jj_frac' parameter
                    for tmp_line in self.no_data_lines[tmp_isotope][tmp_ion].keys():
                        self.plasma_state[tmp_isotope][tmp_ion][tmp_line] = {'conc*tec*jj_frac': None}

            pass

        self.total_impurity_electrons = None
        self.plasma_state['main_ion_density'] = None

        pass

    def update_plasma(self, theta):

        theta = copy.copy(theta) # avoids theta changing external varriables

        # check is the chords are calibrated
        if any(self.is_chord_not_calibrated):
            tmp_a_cal_array = theta[self.all_the_indicies['calibration_index']:
                                    self.all_the_indicies['intercept_index']]

            self.plasma_state['a_cal'] = np.power(10, tmp_a_cal_array)

        self.plasma_state['intercept'] = np.power(10, theta[self.all_the_indicies['intercept_index']])

        if self.hydrogen_isotope:
            self.plasma_state['B-field'] = theta[self.all_the_indicies['b_index']]
            self.plasma_state['view_angle'] = theta[self.all_the_indicies['view_angle_index']] * 180

        ne_theta = theta[self.all_the_indicies['ne_index']:
                         self.all_the_indicies['te_index']]

        ne_theta = theta[self.theta_slices['electron_density']]

        te_theta = []

        if self.profile_function.dr:
            te_theta.append([0])

        for tmp in theta[self.all_the_indicies['te_index']:
                         self.all_the_indicies['upper_te_index']]:

            te_theta.append(tmp)

        self.plasma_state['electron_density'] = np.power(10., self.profile_function(ne_theta))

        if self.logte:
            self.plasma_state['electron_temperature'] = np.power(10., self.profile_function(te_theta))
        else:
            self.plasma_state['electron_temperature'] = self.profile_function(te_theta)

        x_param = 'conc*tec*jj_frac'
        x_index = self.all_the_indicies['x_index']
        no_data_index = self.all_the_indicies['no_data_index']

        tmp_shift = 0
        tmp_shift_ti = 0
        tmp_shift_tau = 0
        tmp_nodata_shift = 0

        # for tmp_isotope in self.input_dict['isotopes']:
        for tmp_isotope in self.input_dict['physics'].keys():

            if tmp_isotope == 'X':
                try:
                    for countx, tmpx in enumerate(self.plasma['physics'][tmp_isotope].keys()): # ['lines']:
                        self.plasma_state[tmp_isotope][tmpx][x_param] = np.power(10, theta[ x_index + countx ])
                except:

                    print(x_index, countx, tmpx)
                    print(len(theta))

                    raise

            elif any([tmp_isotope == h for h in self.hydrogen_isotopes]):

                if self.conc:
                    tmp_log_conc = theta[ self.all_the_indicies['conc_index'] + tmp_shift ] + \
                                       theta[self.all_the_indicies['te_index'] - 1]
                else:
                    tmp_log_conc = theta[self.all_the_indicies['conc_index'] + tmp_shift]

                tmp_ti = np.power(10., theta[ self.all_the_indicies['ti_index'] + tmp_shift_ti ] )

                self.plasma_state[tmp_isotope]['0'] = {'conc': np.power(10., tmp_log_conc), 'ti': tmp_ti}

                tmp_shift += 1
                tmp_shift_ti += 1

            else:

                # tmp_number_of_ions = len(self.input_dict['physics'][tmp_isotope]['ions'])
                # TODO: Need to add functionality for ion resolution
                for tmp_ion_counter, tmp_ion in enumerate(self.input_dict['physics'][tmp_isotope]['ions']):

                    if self.conc:
                        tmp_log_conc = theta[self.all_the_indicies['conc_index'] + tmp_shift] + \
                                   theta[self.all_the_indicies['te_index'] - 1]
                    else:
                        tmp_log_conc = theta[self.all_the_indicies['conc_index'] + tmp_shift]

                    if self.netau:
                        tmp_log_tau = theta[self.all_the_indicies['tau_index'] + tmp_shift_tau] - \
                                      theta[self.all_the_indicies['te_index'] - 1]
                    else:
                        tmp_log_tau = theta[self.all_the_indicies['tau_index'] + tmp_shift_tau]
                        pass

                    self.plasma_state[tmp_isotope][tmp_ion] = \
                        {'conc': np.power(10., tmp_log_conc),
                         'ti': np.power(10., theta[ self.all_the_indicies['ti_index'] + tmp_shift_ti ] ),
                         'tau': np.power( 10.,  tmp_log_tau )}

                    if self.inference_resolution['ion_resolved_tau']:
                        tmp_shift += 1
                        tmp_shift_tau += 1

                    if self.inference_resolution['ion_resolved_temperatures']:
                        tmp_shift_ti += 1

                if not self.inference_resolution['ion_resolved_tau']:
                    tmp_shift += 1
                    tmp_shift_tau += 1

                if not self.inference_resolution['ion_resolved_temperatures']:
                    tmp_shift_ti += 1


            # update non-physics parameters
            #
            # check if the species has a nodata line
            if tmp_isotope in self.no_data_lines.keys():

                # check if the ion has a nodata line
                for tmp_ion in self.no_data_lines[tmp_isotope].keys():

                    # check that the ion has a subdict
                    if tmp_ion not in self.plasma_state[tmp_isotope].keys():
                        self.plasma_state[tmp_isotope][tmp_ion] = {}

                    # update the 'conc*tec*jj_frac' parameter
                    for tmp_line in self.no_data_lines[tmp_isotope][tmp_ion].keys():

                        try:
                            self.plasma_state[tmp_isotope][tmp_ion][tmp_line] = {}
                            self.plasma_state[tmp_isotope][tmp_ion][tmp_line]['conc*tec*jj_frac'] = \
                                theta[no_data_index + tmp_nodata_shift]
                        except KeyError:
                            print(tmp_isotope, tmp_ion, tmp_line)
                            raise
                        except IndexError:
                            print(tmp_isotope, tmp_ion, tmp_line)
                            print(len(theta), no_data_index, tmp_nodata_shift)
                            raise
                        except:
                            raise


                        tmp_nodata_shift += 1

            pass


        self.calc_total_impurity_electrons()
        self.plasma_state['main_ion_density'] = self.plasma_state['electron_density'] - self.total_impurity_electrons

        pass

    def test(self):

        for tag in self.tags:
            t = theta[self.slices[tag]]
            info = self.info_map[tag]

            for key in info:
                self.plasma_dict[key] = t

    def gets_the_slices_for_each_ion_given_chosen_flags(self, theta):

        theta = copy.copy(theta) # avoids theta changing external varriables

        # check is the chords are calibrated
        if any(self.is_chord_not_calibrated):
            tmp_a_cal_array = theta[self.all_the_indicies['calibration_index']:
                                    self.all_the_indicies['intercept_index']]

            self.plasma_state['a_cal'] = np.power(10, tmp_a_cal_array)

        self.plasma_state['intercept'] = np.power(10, theta[self.all_the_indicies['intercept_index']])

        if self.hydrogen_isotope:
            self.plasma_state['B-field'] = theta[self.all_the_indicies['b_index']]
            self.plasma_state['view_angle'] = theta[self.all_the_indicies['view_angle_index']] * 180

        ne_theta = theta[self.all_the_indicies['ne_index']:
                         self.all_the_indicies['te_index']]

        ne_theta = theta[self.theta_slices['electron_density']]

        te_theta = []

        if self.profile_function.dr:
            te_theta.append([0])

        for tmp in theta[self.all_the_indicies['te_index']:
                         self.all_the_indicies['upper_te_index']]:

            te_theta.append(tmp)

        self.plasma_state['electron_density'] = np.power(10., self.profile_function(ne_theta))

        if self.logte:
            self.plasma_state['electron_temperature'] = np.power(10., self.profile_function(te_theta))
        else:
            self.plasma_state['electron_temperature'] = self.profile_function(te_theta)

        x_param = 'conc*tec*jj_frac'
        x_index = self.all_the_indicies['x_index']
        no_data_index = self.all_the_indicies['no_data_index']

        tmp_shift = 0
        tmp_shift_ti = 0
        tmp_shift_tau = 0
        tmp_nodata_shift = 0

        # for tmp_isotope in self.input_dict['isotopes']:
        for tmp_isotope in self.input_dict['physics'].keys():

            if tmp_isotope == 'X':
                try:
                    for countx, tmpx in enumerate(self.plasma['physics'][tmp_isotope].keys()): # ['lines']:
                        self.plasma_state[tmp_isotope][tmpx][x_param] = np.power(10, theta[ x_index + countx ])
                except:

                    print(x_index, countx, tmpx)
                    print(len(theta))

                    raise

            elif any([tmp_isotope == h for h in self.hydrogen_isotopes]):

                if self.conc:
                    tmp_log_conc = theta[ self.all_the_indicies['conc_index'] + tmp_shift ] + \
                                       theta[self.all_the_indicies['te_index'] - 1]
                else:
                    tmp_log_conc = theta[self.all_the_indicies['conc_index'] + tmp_shift]

                tmp_ti = np.power(10., theta[ self.all_the_indicies['ti_index'] + tmp_shift_ti ] )

                self.plasma_state[tmp_isotope]['0'] = {'conc': np.power(10., tmp_log_conc), 'ti': tmp_ti}

                tmp_shift += 1
                tmp_shift_ti += 1

            else:

                # tmp_number_of_ions = len(self.input_dict['physics'][tmp_isotope]['ions'])
                # TODO: Need to add functionality for ion resolution
                for tmp_ion_counter, tmp_ion in enumerate(self.input_dict['physics'][tmp_isotope]['ions']):

                    if self.conc:
                        tmp_log_conc = theta[self.all_the_indicies['conc_index'] + tmp_shift] + \
                                   theta[self.all_the_indicies['te_index'] - 1]
                    else:
                        tmp_log_conc = theta[self.all_the_indicies['conc_index'] + tmp_shift]

                    if self.netau:
                        tmp_log_tau = theta[self.all_the_indicies['tau_index'] + tmp_shift_tau] - \
                                      theta[self.all_the_indicies['te_index'] - 1]
                    else:
                        tmp_log_tau = theta[self.all_the_indicies['tau_index'] + tmp_shift_tau]
                        pass

                    self.plasma_state[tmp_isotope][tmp_ion] = \
                        {'conc': np.power(10., tmp_log_conc),
                         'ti': np.power(10., theta[ self.all_the_indicies['ti_index'] + tmp_shift_ti ] ),
                         'tau': np.power( 10.,  tmp_log_tau )}

                    if self.inference_resolution['ion_resolved_tau']:
                        tmp_shift += 1
                        tmp_shift_tau += 1

                    if self.inference_resolution['ion_resolved_temperatures']:
                        tmp_shift_ti += 1

                if not self.inference_resolution['ion_resolved_tau']:
                    tmp_shift += 1
                    tmp_shift_tau += 1

                if not self.inference_resolution['ion_resolved_temperatures']:
                    tmp_shift_ti += 1


            # update non-physics parameters
            #
            # check if the species has a nodata line
            if tmp_isotope in self.no_data_lines.keys():

                # check if the ion has a nodata line
                for tmp_ion in self.no_data_lines[tmp_isotope].keys():

                    # check that the ion has a subdict
                    if tmp_ion not in self.plasma_state[tmp_isotope].keys():
                        self.plasma_state[tmp_isotope][tmp_ion] = {}

                    # update the 'conc*tec*jj_frac' parameter
                    for tmp_line in self.no_data_lines[tmp_isotope][tmp_ion].keys():

                        try:
                            self.plasma_state[tmp_isotope][tmp_ion][tmp_line] = {}
                            self.plasma_state[tmp_isotope][tmp_ion][tmp_line]['conc*tec*jj_frac'] = \
                                theta[no_data_index + tmp_nodata_shift]
                        except KeyError:
                            print(tmp_isotope, tmp_ion, tmp_line)
                            raise
                        except IndexError:
                            print(tmp_isotope, tmp_ion, tmp_line)
                            print(len(theta), no_data_index, tmp_nodata_shift)
                            raise
                        except:
                            raise


                        tmp_nodata_shift += 1

            pass


        self.calc_total_impurity_electrons()
        self.plasma_state['main_ion_density'] = self.plasma_state['electron_density'] - self.total_impurity_electrons

        pass

    def calc_total_impurity_electrons(self, set=True):

        total_impurity_electrons = 0

        ne = self.plasma_state['electron_density']
        te = self.plasma_state['electron_temperature']

        for tmp_element in list(self.input_dict['physics'].keys()):

            if any([tmp_element == t for t in ('D', 'H', 'X')]):
                pass
            else:
                try:
                    tau = self.plasma_state[tmp_element]['tau']
                    conc = self.plasma_state[tmp_element]['conc']
                except KeyError:
                    tmp_key = min( list( self.plasma_state[tmp_element].keys() ) )
                    tau = self.plasma_state[tmp_element][tmp_key]['tau']
                    conc = self.plasma_state[tmp_element][tmp_key]['conc']
                    pass
                except:
                    raise

                tau = np.zeros(len(ne)) + tau

                tmp_in = np.array([tau, ne, te]).T

                average_charge = self.impurity_average_charge[tmp_element](tmp_in)

                total_impurity_electrons += average_charge * conc

                pass

        if set:
            self.total_impurity_electrons = np.nan_to_num(total_impurity_electrons)
        else:
            return np.nan_to_num( total_impurity_electrons )

    def build_impurity_average_charge(self):

        tmp_impurity_average_charge = {}

        for tmp_element in list(self.input_dict['physics'].keys()):

            if any([tmp_element == t for t in ('D', 'H', 'X')]):
                pass
            else:
                try:
                    tmp_file = self.input_dict['physics'][tmp_element]['effective_charge_406']
                except KeyError:
                    print(self.input_dict['physics'][tmp_element].keys())
                    raise
                except:
                    raise

                tmp_impurity_average_charge[tmp_element] = build_tec406(tmp_file)

        self.impurity_average_charge = tmp_impurity_average_charge

    # TODO: Need to add functionality for no data lines and mystery lines
    def chain_bounds(self):

        '''
        self.all_the_indicies = [intercept_index,
                                 ne_index, te_index,
                                 conc_index, tau_index, ti_index,
                                 x_index, no_data_index]

        e.g. theta = [1e0
                      -2 10 1e13 8 4
                      1.0 0.1
                      1e-1 1e-1
                      10 20]

        :return:
        '''

        # print(self.all_the_indicies)

        theta_bounds = []
        theta_widths = []

        self.default_start = []

        # Bounds
        cal_bounds = [0, 17]
        cal_widths = 1

        intercept_bounds = [0, 17]
        intercept_widths = 2

        dr_bounds = [-10, 10]
        fwhm_bounds = [0.05, 5.0]

        dr_widths = 0.5
        fwhm_widths = 1

        ne_bounds = [12, 16]
        # te_bounds = [11, 18] # if log(Pe)

        if self.logte:
            te_bounds = [-1, 3]
        else:
            te_bounds = [0.5, 50]

        ne_widths = 2
        te_widths = 5

        b_field_bounds = [0, 3] # 5]
        viewangle_bounds = [0, 2]

        b_field_widths = 0.1
        viewangle_widths = 10

        if self.conc:
            conc_bounds = [-5, 0]
        else:
            conc_bounds = [0, 15]

        if self.netau:
            tau_bounds = [8.7, 17] # sampled as log10(ne*tau)
        else:
            tau_bounds = [-7, 0]

        ti_bounds = [-1, 2]

        conc_widths = 0.5
        tau_widths = 2
        ti_widths = 1

        ems_bounds = [0, 20]

        ems_widths = 1

        atomic_bounds = [conc_bounds, ti_bounds, tau_bounds]
        atomic_widths = [conc_widths, ti_widths, tau_widths]

        # Structuring theta_bounds and widths

        num_not_calibrated = len( np.where(self.is_chord_not_calibrated == True)[0] )

        if num_not_calibrated > 0:

            for counter in np.arange( num_not_calibrated ):

                theta_bounds.append(cal_bounds)
                theta_widths.append(cal_widths)

                self.default_start.append( 10 )

            self.default_start.append(1)
        else:
            self.default_start.append(11)

        theta_bounds.append(intercept_bounds)
        theta_widths.append(intercept_widths)

        if self.profile_function.dr:
            theta_bounds.append(dr_bounds)
            theta_widths.append(dr_widths)

            self.default_start.append(-0.1)

            for counter in np.arange(self.profile_function.number_of_varriables-2):
                theta_bounds.append(fwhm_bounds)
                theta_widths.append(fwhm_widths)

                self.default_start.append(1)

            theta_bounds.append(ne_bounds)
            theta_widths.append(ne_widths)

            self.default_start.append(13.7)

            for counter in np.arange(self.profile_function.number_of_varriables-2):
                theta_bounds.append(fwhm_bounds)
                theta_widths.append(fwhm_widths)

                self.default_start.append(1)

            theta_bounds.append(te_bounds)
            theta_widths.append(te_widths)

            self.default_start.append(10)
            # self.default_start.append(15) # if log(Pe)
        else:
            for counter in np.arange(self.profile_function.number_of_varriables-1):
                theta_bounds.append(fwhm_bounds)
                theta_widths.append(fwhm_widths)

                self.default_start.append(1)

            theta_bounds.append(ne_bounds)
            theta_widths.append(ne_widths)

            self.default_start.append(13.7)

            for counter in np.arange(self.profile_function.number_of_varriables-1):
                theta_bounds.append(fwhm_bounds)
                theta_widths.append(fwhm_widths)

                self.default_start.append(1)

            theta_bounds.append(te_bounds)
            theta_widths.append(te_widths)

            self.default_start.append(10)
            # self.default_start.append(15) # if log(Pe)

        if self.hydrogen_isotope:

            theta_bounds.append(b_field_bounds)
            theta_bounds.append(viewangle_bounds)

            theta_widths.append(b_field_widths)
            theta_widths.append(viewangle_widths)

            for tmp in [1, 1]:

                self.default_start.append(tmp)

            # conc bounds
            if self.inference_resolution['ion_resolved_tau']:
                number_of_things_tc = self.number_of_ions
            else:
                number_of_things_tc = self.number_of_isotopes

            counter = 0
            while counter < (number_of_things_tc):
                theta_bounds.append(atomic_bounds[0])
                theta_widths.append(atomic_widths[0])

                self.default_start.append(-1)

                counter += 1

            # ti bounds
            if self.inference_resolution['ion_resolved_temperatures']:
                number_of_things_ti = self.number_of_ions
            else:
                number_of_things_ti = self.number_of_isotopes

            counter = 0
            while counter < (number_of_things_ti):
                theta_bounds.append(ti_bounds) # atomic_bounds[-1])
                theta_widths.append(ti_widths) # atomic_widths[-1])

                self.default_start.append(1)

                counter += 1

            # tau bounds
            if self.inference_resolution['ion_resolved_tau']:
                number_of_things_tau = self.number_of_ions
            else:
                number_of_things_tau = self.number_of_isotopes

            counter = 0
            while counter < (number_of_things_tau-1):
                theta_bounds.append(tau_bounds) # atomic_bounds[1])
                theta_widths.append(tau_widths) # atomic_widths[1])

                self.default_start.append(11.5)

                counter += 1

        else:
            if self.netau:
                default_tau = 11.5
            else:
                default_tau = -1

            atomic_default_start = [-1, 1, default_tau]

            key_checks = ['ion_resolved_tau', 'ion_resolved_temperatures', 'ion_resolved_tau']

            for counter, tmp_atomic_bounds in enumerate(atomic_bounds):

                if self.inference_resolution[key_checks[counter]]:
                    number_of_things = self.number_of_ions
                else:
                    number_of_things = self.number_of_isotopes

                counter1 = 0
                while counter1 < number_of_things:

                    theta_bounds.append(tmp_atomic_bounds)
                    theta_widths.append(atomic_widths[counter])

                    self.default_start.append(atomic_default_start[counter])

                    counter1 += 1

                if self.inference_resolution['ion_resolved_tau']: pass
                if self.inference_resolution['ion_resolved_temperatures']: pass

        num_of_ems = self.number_of_xlines + self.number_of_no_data_lines

        if num_of_ems > 0:
            counter = 0

            while counter < num_of_ems:

                self.default_start.append(11)

                theta_bounds.append(ems_bounds)
                theta_widths.append(ems_widths)

                counter += 1

        self.theta_bounds = np.array(theta_bounds)
        self.theta_widths = np.array(theta_widths)

    def count_species(self):

        nun_species = len( self.input_dict['physics'].keys() )
        num_ions = 0
        num_xlines = 0
        num_no_data_lines = 0

        for tmp_species in self.input_dict['physics'].keys():

            if tmp_species == 'X':
                num_xlines = len(self.input_dict['physics'][tmp_species].keys())
                nun_species -= 1
            else:
                num_ions += len(self.input_dict['physics'][tmp_species]['ions'])

        self.number_of_isotopes, self.number_of_ions, \
        self.number_of_xlines, self.number_of_no_data_lines = \
            nun_species, num_ions, num_xlines, num_no_data_lines

    def where_the_no_data_lines(self):

        no_data_lines = {}

        for counter0, species in enumerate(self.plasma['physics'].keys()):

            if species != 'X':

                for counter1, ion in enumerate(self.plasma['physics'][species]['ions']):

                    for counter2, line in enumerate(self.plasma['physics'][species][ion].keys()):

                        if line != 'no_data_lines':

                            if not 'tec' in self.plasma['physics'][species][ion][line]:

                                # check if there is a species subdict in the first places
                                # then if there are species there is the one wanted there
                                if not any([t == species for t in no_data_lines.keys()]):
                                    no_data_lines[species] = {}
                                    no_data_lines[species][ion] = {}
                                    no_data_lines[species][ion][line] = 0
                                # so now we have accounted for not have the species
                                #
                                # what if the the species doesn't have any ion subdicts
                                # what if the the species doesn't have the wanted ion
                                elif not any([t == ion for t in no_data_lines[species].keys()]):
                                    no_data_lines[species][ion] = {}
                                    no_data_lines[species][ion][line] = 0
                                # so now we have accounted for not having the ions of a species
                                #
                                # what if the ion does not have any lines - then we shall add it
                                else:
                                    no_data_lines[species][ion][line] = 0

                            # print(species, ion, line)

        return no_data_lines

    def is_theta_within_bounds(self, theta):

        out = []

        for counter, bound in enumerate(self.theta_bounds):

            out.append( within(theta[counter], bound) )

        return out

import collections
from baysar.lineshapes import MeshLine


def power10(var):

    return np.power(10, var)


class arb_obj(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, *args):
        return args


class arb_obj_single_input(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, theta):
        return theta


class NewPlasmaLine():

    """
    PlasmaLine is a object that holds the simulated 1D plasma profiles and the function for updating the
    plamsa profiles. The plasma paramaters and profiles are also organised into a dictionary which are
    accessible to the line objects in the initialised spectrometer objects. This in essence is the parameter
    handler which handles the interface between the sampler and the forward model.
    """

    def __init__(self, input_dict, profile_function=None, cal_functions=None, background_functions=None):

        self.input_dict = input_dict
        self.num_chords = len(self.input_dict['chords'].keys())

        self.set_up_theta_functions(profile_function, cal_functions, background_functions)

        self.hydrogen_isotopes = ['H', 'D', 'T']
        self.hydrogen_isotope = any(['H' == t or 'D' == t or 'T' == t
                                     for t in list(self.input_dict['physics'].keys())])

        self.build_tags_slices_and_bounds()
        self.build_impurity_average_charge()

    def __call__(self, theta):

        return self.new_update_plasma(theta)

    # Todo: Add in automated boundaries
    def build_tags_slices_and_bounds(self):

        self.tags = []
        bounds = []
        slice_lengths = []

        for chord, cal_func, back_func in zip(np.arange(self.num_chords), self.cal_functions, self.background_functions):

            self.tags.append('cal'+str(chord))
            self.tags.append('background'+str(chord))

            slice_lengths.append(cal_func.number_of_variables)
            slice_lengths.append(back_func.number_of_variables)

            bounds.append(cal_func.bounds)
            bounds.append(back_func.bounds)

        self.tags.append('electron_density')
        self.tags.append('electron_temperature')

        slice_lengths.append(self.profile_function.number_of_variables_ne)
        slice_lengths.append(self.profile_function.number_of_variables_te)

        [bounds.append(self.profile_function.bounds_ne) for num in np.arange(self.profile_function.number_of_variables_ne)]
        [bounds.append(self.profile_function.bounds_te) for num in np.arange(self.profile_function.number_of_variables_te)]

        self.species = list(self.input_dict['physics'])
        hydrogen_shape_tags = ['b-field', 'viewangle']
        hydrogen_shape_bounds = [[0, 10], [0, 2]]

        if self.hydrogen_isotope:
            for tag, b in zip(hydrogen_shape_tags, hydrogen_shape_bounds):
                self.tags.append(tag)
                slice_lengths.append(1)
                bounds.append(b)

        impurity_tags = ['_dens', '_Ti', '_tau']
        impurity_bounds = [[9, 15], [-1, 3], [-6, 3]]
        for tag in impurity_tags:
            for s in species:
                if ((tag == '_tau') and (s in self.hydrogen_isotopes)) or s=='X':
                    pass
                else:
                    for ion in self.input_dict['physics'][s]['ions']:
                        sion = s+'_'+ion+tag
                        tmp_b = impurity_bounds[np.where([t in sion for t in impurity_tags])[0][0]]

                        self.tags.append(sion)
                        slice_lengths.append(1)
                        bounds.append(tmp_b)

                        # print(self.tags[-1], bounds[-1], bounds)

        if 'X' in self.species:
            for line in self.input_dict['physics']['X'].keys():
                self.tags.append('X_' + line.replace(' ', '_'))
                slice_lengths.append(1)
                bounds.append([0, 20])

        # building dictionary to have flags for which parameters are resoloved
        self.is_resolved = collections.OrderedDict()
        ti_resolved = self.input_dict['inference_resolution']['ion_resolved_temperatures']
        tau_resolved = self.input_dict['inference_resolution']['ion_resolved_tau']
        # checking if Ti and tau is resolved
        res_not_apply = [not any([tag.endswith(t) for t in impurity_tags]) for tag in self.tags]
        ti_resolution_check = [tag.endswith('_Ti') and ti_resolved for tag in self.tags]
        dens_tau_resolution_check = [(tag.endswith('_tau') or tag.endswith('_dens')) and tau_resolved for tag in self.tags]
        # actually making dictionary to have flags for which parameters are resoloved
        for tag, is_ti_r, is_tau_r, rna in \
                zip(self.tags, ti_resolution_check, dens_tau_resolution_check, res_not_apply):
            self.is_resolved[tag] = is_ti_r or is_tau_r or rna

            # # adding the bounds for the impurity_tags
            # if any([t in tag for t in impurity_tags]):
            #     bounds.append(impurity_bounds[ np.where([t in tag for t in impurity_tags]) ])

        # building the slices to map BaySAR input to model parameters
        slices = []
        bounds_to_remove_indicies = []
        for p, L, b, n in zip(self.tags, slice_lengths, bounds, np.arange(len(bounds))):
            if len(slices) is 0:
                slices.append( (p, slice(0, L)) )
            elif self.is_resolved[p]:
                last = slices[-1][1].stop
                slices.append((p, slice(last, last+L)))
            # if the parameter is not resolved there is a check for previous tags that it will share a slice with
            else:
                current_imp_tag = [imp_tag for imp_tag in impurity_tags if p.endswith(imp_tag)]

                slc = [s for tag, s in slices if (tag.startswith(p[:1]) and tag.endswith(current_imp_tag[0]))]

                if len(slc) != 0:
                    slices.append((p, slc[0]))
                    bounds_to_remove_indicies.append(n)
                else:
                    last = slices[-1][1].stop
                    slices.append((p, slice(last, last+L)))

        self.n_params = slices[-1][1].stop
        self.slices = collections.OrderedDict(slices)
        self.theta_bounds = [np.array(b) for b, n in zip(bounds, np.arange(len(bounds))) if not any([n==check for check in bounds_to_remove_indicies])]
        self.bounds = bounds

        assert self.n_params==len(self.theta_bounds), 'self,n_params!=len(self.theta_bounds)'

        self.assign_theta_functions()

    def update_plasma_theta_tags(self, theta):

        plasma_theta = []

        for p in self.slices.keys():
            plasma_theta.append((p, theta[self.slices[p]]))

        self.plasma_theta = collections.OrderedDict(plasma_theta)

    def new_update_plasma(self, theta):

        assert len(theta)==self.n_params, 'len(theta)!=self.n_params'

        self.update_plasma_theta_tags(theta)

        plasma_state = []

        for p, f_check in zip(self.slices.keys(), self.function_check):
            values = theta[self.slices[p]]

            if f_check:
                plasma_state.append((p, self.theta_functions[p](values)))
            else:
                plasma_state.append((p, values))

        self.plasma_state_tags = collections.OrderedDict(plasma_state)
        self.plasma_state_tags['main_ion_density'] = self.plasma_state_tags['electron_density'] - self.calc_total_impurity_electrons(False)

    def set_up_theta_functions(self, profile_function=None, cal_functions=None, background_functions=None):

        if profile_function is None:
            x = np.linspace(1, 2, 3)
            profile_function = MeshLine(x=x, zero_bounds=-2, bounds=[0, 3], log=True)
            self.profile_function = arb_obj(electron_density=profile_function,
                                            electron_temperature=profile_function,
                                            number_of_variables_ne=3,
                                            number_of_variables_te=3,
                                            bounds_ne=[11, 16], bounds_te=[-1, 2])
        else:
            self.profile_function = profile_function

        tmp_func = arb_obj_single_input(number_of_variables=1, bounds=[5, 20])

        if cal_functions is None:
            self.cal_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.cal_functions = cal_functions

        if background_functions is None:
            self.background_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.background_functions = background_functions

    def assign_theta_functions(self):

        theta_functions = [self.cal_functions, self.background_functions,
                           self.profile_function.electron_density,
                           self.profile_function.electron_temperature,
                           [power10 for tag in self.tags if any([tag.startswith(s+'_') for s in self.species])]]
        theta_functions = [[f] if type(f)!=list else f for f in theta_functions]
        theta_functions = [f1 for f0 in theta_functions for f1 in f0]

        theta_functions_tuples = []

        function_tags = ['cal', 'back', 'electron_density', 'electron_temperature']
        function_tags_full = [tag for tag in self.tags if any([check in tag for check in function_tags]) or any([tag.startswith(s+'_') for s in self.species])]

        self.function_check = [any([check in tag for check in function_tags]) or any([tag.startswith(s+'_') for s in self.species]) for tag in self.tags]
        # self.function_check = [any([check in tag for check in function_tags]) for tag in plasma.tags]

        for tag, func in zip(function_tags_full, theta_functions):
            theta_functions_tuples.append((tag, func))

        self.theta_functions = collections.OrderedDict(theta_functions_tuples)

    # Todo: need to correct
    def calc_total_impurity_electrons(self, set=True):

        total_impurity_electrons = 0

        ne = self.plasma_state_tags['electron_density']
        te = self.plasma_state_tags['electron_temperature']

        for tmp_element in list(self.input_dict['physics'].keys()):

            if any([tmp_element == t for t in ('D', 'H', 'X')]):
                pass
            else:
                tau_tag = [tag for tag in plasma.tags if tmp_element in tag and tag.endswith('_tau')]
                dens_tag = [tag for tag in plasma.tags if tmp_element in tag and tag.endswith('_dens')]

                for t_tag, d_tag in zip(tau_tag, dens_tag):

                    tau = self.plasma_state_tags[t_tag]
                    conc = self.plasma_state_tags[d_tag]

                    tau = np.zeros(len(ne)) + tau

                    tmp_in = np.array([tau, ne, te]).T

                    average_charge = self.impurity_average_charge[tmp_element](tmp_in)

                    total_impurity_electrons += average_charge * conc

        if set:
            self.total_impurity_electrons = np.nan_to_num(total_impurity_electrons)
        else:
            return np.nan_to_num(total_impurity_electrons)

    def build_impurity_average_charge(self):

        tmp_impurity_average_charge = {}

        for tmp_element in list(self.input_dict['physics'].keys()):

            if any([tmp_element == t for t in ('D', 'H', 'X')]):
                pass
            else:
                tmp_file = self.input_dict['physics'][tmp_element]['effective_charge_406']
                tmp_impurity_average_charge[tmp_element] = build_tec406(tmp_file)

        self.impurity_average_charge = tmp_impurity_average_charge

    # TODO: Need to add functionality for no data lines and mystery lines
    def is_theta_within_bounds(self, theta):

        out = []

        for counter, bound in enumerate(self.theta_bounds):
            out.append(within(theta[counter], bound))

        return out


if __name__ == '__main__':

    from baysar.input_functions import make_input_dict

    num_chords = 1
    wavelength_axes = [np.linspace(4000, 4100, 512)]
    experimental_emission = [1e12*np.random.rand() for w in wavelength_axes[0]]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ['D', 'N']
    ions = [ ['0'], ['1', '2', '3'] ]
    noise_region = [[4040, 4050]]
    mystery_lines = [[[4070], [4001, 4002]], [1, [0.4, 0.6]]]

    input_dict = make_input_dict(num_chords=num_chords,
                                 wavelength_axes=wavelength_axes, experimental_emission=experimental_emission,
                                 instrument_function=instrument_function, emission_constant=emission_constant,
                                 noise_region=noise_region, species=species, ions=ions,
                                 mystery_lines=mystery_lines, refine=[0.01],
                                 ion_resolved_temperatures=False, ion_resolved_tau=True)

    plasma = NewPlasmaLine(input_dict)

    rand_theta = np.random.rand(plasma.n_params)
    plasma(rand_theta)

    # # for k in plasma.plasma_state_tags.keys():
    # for k in plasma.slices.keys():
    #     print(k, plasma.plasma_state_tags[k], type(plasma.plasma_state_tags[k]), plasma.slices[k],
    #           plasma.theta_bounds[plasma.slices[k]], type(plasma.theta_bounds[plasma.slices[k]]))
    #
    # print(plasma.bounds)
    # print(plasma.is_theta_within_bounds(rand_theta))


    pass