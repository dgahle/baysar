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

    def __init__(self, input_dict, profile_fucntion=None, conc=True, logte=False, netau=False):

        self.netau = netau
        self.conc = conc
        self.logte = logte

        self.input_dict = input_dict

        self.plasma = input_dict

        self.inference_resolution = self.input_dict['inference_resolution']

        if profile_fucntion is None:
            # TODO: x range for arange needs to be a top level variable
            los_x = np.arange(-15., 15., 0.2)
            self.profile_fucntion = Eich(x=los_x, reduced=True)
            self.profile_fucntion.number_of_varriables = 3
            self.profile_fucntion.dr = True

            self.profile_fucntion_num_varriables = self.profile_fucntion.number_of_varriables

            # self.profile_fucntion = Gaussian( x )
            # self.profile_fucntion = Gaussian( x=np.arange(-30., 30., 3) )

        else:
            self.profile_fucntion = profile_fucntion # ...
            self.profile_fucntion_num_varriables = self.profile_fucntion.number_of_varriables

        self.count_species()

        self.no_data_lines = self.where_the_no_data_lines()
        self.build_impurity_average_charge()

        self.plasma_state = {}

        self.hydrogen_isotopes = ['H', 'D', 'T']
        self.hydrogen_isotope = any(['H' == t or 'D' == t or 'T' == t
                                     for t in list(self.input_dict['physics'].keys())])

        self.build_plasma()
        # self.update_plasma(np.zeros(int(1e3)))

        self.plasma_state['los'] = self.profile_fucntion.x

        # self.theta_bounds = self.chain_bounds()
        self.chain_bounds()

    def __call__(self, theta):

        return self.update_plasma(theta)

    def build_plasma(self):

        self.all_the_indicies = {}

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

        else:
            intercept_index = 0

        self.all_the_indicies['intercept_index'] = intercept_index

        # defining indicies for electron density and temperature profiles
        self.all_the_indicies['ne_index'] = intercept_index + 1
        self.all_the_indicies['te_index'] = self.all_the_indicies['ne_index'] + \
                                            self.profile_fucntion_num_varriables


        # defining indicies for viewing-angle, b-field and concentration
        if self.hydrogen_isotope:
            if self.profile_fucntion.dr:
                k_dr = 1
            else:
                k_dr = 0

            self.all_the_indicies['b_index'] = self.all_the_indicies['te_index'] + \
                                               self.profile_fucntion_num_varriables - k_dr

            self.all_the_indicies['view_angle_index'] = self.all_the_indicies['b_index'] + 1
            self.all_the_indicies['conc_index'] = self.all_the_indicies['view_angle_index'] + 1

            self.all_the_indicies['upper_te_index'] = self.all_the_indicies['b_index']

            self.plasma_state['B-field'] = None # theta[b_index]
            self.plasma_state['view_angle'] = None # theta[view_angle_index]
        else:
            if self.profile_fucntion.dr:
                k_dr = 1
            else:
                k_dr = 0

            self.all_the_indicies['conc_index'] = self.all_the_indicies['te_index'] + \
                                                  self.profile_fucntion_num_varriables - k_dr

            self.all_the_indicies['upper_te_index'] = self.all_the_indicies['conc_index']

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

        te_theta = []

        if self.profile_fucntion.dr:
            te_theta.append([0])

        for tmp in theta[self.all_the_indicies['te_index']:
                         self.all_the_indicies['upper_te_index']]:

            te_theta.append(tmp)

        self.plasma_state['electron_density'] = np.power(10., self.profile_fucntion(ne_theta))

        if self.logte:
            self.plasma_state['electron_temperature'] = np.power(10., self.profile_fucntion(te_theta))
        else:
            self.plasma_state['electron_temperature'] = self.profile_fucntion(te_theta)

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

        if self.profile_fucntion.dr:
            theta_bounds.append(dr_bounds)
            theta_widths.append(dr_widths)

            self.default_start.append(-0.1)

            for counter in np.arange(self.profile_fucntion.number_of_varriables-2):
                theta_bounds.append(fwhm_bounds)
                theta_widths.append(fwhm_widths)

                self.default_start.append(1)

            theta_bounds.append(ne_bounds)
            theta_widths.append(ne_widths)

            self.default_start.append(13.7)

            for counter in np.arange(self.profile_fucntion.number_of_varriables-2):
                theta_bounds.append(fwhm_bounds)
                theta_widths.append(fwhm_widths)

                self.default_start.append(1)

            theta_bounds.append(te_bounds)
            theta_widths.append(te_widths)

            self.default_start.append(10)
            # self.default_start.append(15) # if log(Pe)
        else:
            for counter in np.arange(self.profile_fucntion.number_of_varriables-1):
                theta_bounds.append(fwhm_bounds)
                theta_widths.append(fwhm_widths)

                self.default_start.append(1)

            theta_bounds.append(ne_bounds)
            theta_widths.append(ne_widths)

            self.default_start.append(13.7)

            for counter in np.arange(self.profile_fucntion.number_of_varriables-1):
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

                    for counter2, line in enumerate(self.plasma['physics'][species][ion]['lines']):

                        # print(species, ion, line)

                        if not any([k=='tec' for k in self.plasma['physics'][species][ion][line].keys()]):

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



if __name__ == '__main__':

    input_dict = {}

    num_chords = 1

    input_dict['number_of_chords'] = num_chords

    input_dict['chords'] = {}

    wavelength_axes = [[]]
    experimental_emission = [[]]

    instrument_function = [[]]
    emission_constant = [...]
    noise_region = [[]]

    for counter0, chord in enumerate(np.arange(num_chords)):
        # tmp = 'chord' + str(counter0)
        tmp = counter0

        input_dict['chords'][tmp] = {}
        input_dict['chords'][tmp]['meta'] = {}

        input_dict['chords'][tmp]['meta']['wavelength_axis'] = wavelength_axes[counter0]
        input_dict['chords'][tmp]['meta']['experimental_emission'] = experimental_emission[counter0]

        input_dict['chords'][tmp]['meta']['instrument_function'] = instrument_function[counter0]
        input_dict['chords'][tmp]['meta']['emission_constant'] = emission_constant[counter0]
        input_dict['chords'][tmp]['meta']['noise_region '] = noise_region[counter0]

        pass

    '''
    n_ii_cwls = [3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92]
    n_ii_jjr = [1, 0.92, 0.08, 0.456, 0.211, 0.197, 0.026, 0.022]
    # n_ii_pec_keys = [0, 2, 2, 3, 3, 3, 3, 3]
    n_ii_pec_keys = [3, 6, 6, 7, 7, 7, 7, 7]

    # n_iii_cwls = [3998.63, 4003.58, 4097.33, 4103.34]
    # n_iii_jjr = [0.375, 0.625, 0.665, 0.335]
    # n_iii_pec_keys = [1, 1, 4, 4]    
    '''

    chord0_dict = {}

    species = ['D', 'N', 'X']
    ions = [['0'], ['1', '2']]

    cwl = [[[3968.99, 4100.58]],
           [[3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92],
            [3998.63, 4003.58, 4097.33, 4103.34]]]
    n_pec = [[[3968.99, 4100.58]],
             [[3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92],
              [3998.63, 4003.58, 4097.33, 4103.34]]]
    f_jj = [[[3968.99, 4100.58]],
            [[3995., 4026.09, 4039.35, 4041.32, 4035.09, 4043.54, 4044.79, 4056.92],
             [3998.63, 4003.58, 4097.33, 4103.34]]]

    atomic_charge = [1, 7]
    ma = [2, 14]

    for counter0, isotope in enumerate(species):

        # if counter0 == 0:
        #     input_dict['isotopes'] = species

        chord0_dict[isotope] = {}
        chord0_dict[isotope]['atomic_mass'] = ma[counter0]
        chord0_dict[isotope]['atomic_charge'] = atomic_charge[counter0]

        for counter1, ion in enumerate(ions[counter0]):

            if counter1 == 0:
                chord0_dict[isotope]['ions'] = ions[counter0]

            chord0_dict[isotope][ion] = {}

            for counter2, line in enumerate(cwl[counter0][counter1]):

                if counter2 == 0:
                    chord0_dict[isotope][ion]['lines'] = []

                chord0_dict[isotope][ion]['lines'].append(str(line))

                chord0_dict[isotope][ion][str(line)] = {}

                chord0_dict[isotope][ion][str(line)]['wavelength'] = line
                chord0_dict[isotope][ion][str(line)]['pec_key'] = n_pec[counter0][counter1][counter2]
                chord0_dict[isotope][ion][str(line)]['jj_frac'] = f_jj[counter0][counter1][counter2]

                pass

            pass

        pass

    input_dict['chords'][0]['physics'] = chord0_dict

    profile_funciton = Gaussian( x=np.arange(-50., 50., 3) )

    plasma = PlasmaLine(input_dict=input_dict['chords'][0]['physics'],
                        profile_fucntion=profile_funciton, profile_fucntion_num_varriables=3)

    theta = [1e12,
             -5, 30, 1e13,
                  5, 10,
             0.8, 0.2,
             1e-4, 1e-4,
             5, 10]

    plasma(theta)

    print( plasma.plasma_state )



    pass