import numpy as np

import os, sys, io
import copy, warnings

import collections
from baysar.lineshapes import MeshLine
from baysar.spectrometers import within

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from adas import run_adas406, read_adf15

def power10(var):
    return np.power(10, var)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class arb_obj(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, *args):
        return args


class arb_obj_single_input(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, theta):
        if type(theta) not in (float, int):
            theta = theta[0]
        return np.power(10, theta)

atomic_number={'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10}
def get_number_of_ions(element):
    return atomic_number[element]+1




class PlasmaLine():

    """
    PlasmaLine is a object that holds the simulated 1D plasma profiles and the function for updating the
    plamsa profiles. The plasma paramaters and profiles are also organised into a dictionary which are
    accessible to the line objects in the initialised spectrometer objects. This in essence is the parameter
    handler which handles the interface between the sampler and the forward model.
    """

    def __init__(self, input_dict, profile_function=None, cal_functions=None, calwave_functions=None, background_functions=None):

        'input_dict input has been checked if created with make_input_dict'

        self.input_dict = input_dict
        self.num_chords = len(self.input_dict['wavelength_axis'])
        self.species = self.input_dict['species']
        self.impurity_species = [s for s in self.species if not any(['H_' in s or 'D_' in s or 'T_' in s])]
        self.hydrogen_species = [s for s in self.species if s not in self.impurity_species]
        self.hydrogen_isotopes = ['H', 'D', 'T']
        self.contains_hydrogen = (self.species!=self.impurity_species)
        self.adas_plasma_inputs = {'te': np.logspace(-1, 2, 60), # TODO can we get the raw data instead?
                                   'ne': np.logspace(12, 15, 15),
                                   'tau': np.logspace(-8, 3, 12)}
        self.adas_plasma_inputs['big_ne'] = np.array([self.adas_plasma_inputs['ne'] for t in self.adas_plasma_inputs['te']]).T

        self.get_impurities()
        self.get_impurity_ion_bal()
        self.build_impurity_average_charge()
        self.get_impurity_tecs()
        if self.contains_hydrogen:
            self.get_hydrogen_pecs()

        self.get_theta_functions(profile_function, cal_functions, calwave_functions, background_functions)
        self.build_tags_slices_and_bounds()

        self.los = self.profile_function.electron_density.x

        self(np.random.rand(self.n_params))

    def __call__(self, theta):

        return self.update_plasma_state(theta)

    def get_impurities(self):
        self.impurities = []
        for species in self.impurity_species:
            split_index = np.where([p == '_' for p in species])[0][0]
            elem = species[:split_index]
            if elem not in self.impurities:
                self.impurities.append(elem)

    def build_tags_slices_and_bounds(self):

        self.tags = []
        bounds = []
        slice_lengths = []

        for chord, cal_func, calwave_func, back_func in zip(np.arange(self.num_chords), self.cal_functions, self.calwave_functions, self.background_functions):

            self.tags.append('cal'+str(chord))
            self.tags.append('calwave'+str(chord))
            self.tags.append('background'+str(chord))

            slice_lengths.append(cal_func.number_of_variables)
            slice_lengths.append(calwave_func.number_of_variables)
            slice_lengths.append(back_func.number_of_variables)

            bounds.append(cal_func.bounds)
            bounds.append(calwave_func.bounds)
            bounds.append(back_func.bounds)

        self.tags.append('electron_density')
        self.tags.append('electron_temperature')

        slice_lengths.append(self.profile_function.number_of_variables_ne)
        slice_lengths.append(self.profile_function.number_of_variables_te)

        [bounds.append(self.profile_function.bounds_ne) for num in np.arange(self.profile_function.number_of_variables_ne)]
        [bounds.append(self.profile_function.bounds_te) for num in np.arange(self.profile_function.number_of_variables_te)]

        hydrogen_shape_tags = ['b-field', 'viewangle']
        hydrogen_shape_bounds = [[0, 10], [0, 2]]

        if self.contains_hydrogen:
            for tag, b in zip(hydrogen_shape_tags, hydrogen_shape_bounds):
                self.tags.append(tag)
                slice_lengths.append(1)
                bounds.append(b)

        impurity_tags = ['_dens', '_Ti', '_tau']
        impurity_bounds = [[9, 15], [-1, 3], [-6, 3]]
        for tag in impurity_tags:
            for s in self.species:
                is_h_isotope = any([s[0:2]==h+'_' for h in self.hydrogen_isotopes])
                if not (tag == '_tau' and is_h_isotope):
                    sion = s+tag
                    tmp_b = impurity_bounds[np.where([t in sion for t in impurity_tags])[0][0]]

                    self.tags.append(sion)
                    slice_lengths.append(1)
                    bounds.append(tmp_b)

        if 'X_lines' in self.input_dict:
            for line in self.input_dict['X_lines']:
                line = str(line)[1:-1].replace(', ', '_')
                self.tags.append('X_' + line)
                slice_lengths.append(1)
                bounds.append([0, 20])

        # building dictionary to have flags for which parameters are resoloved
        self.is_resolved = collections.OrderedDict()
        ti_resolved = self.input_dict['ion_resolved_temperatures']
        tau_resolved = self.input_dict['ion_resolved_tau']
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
        keep_theta_bound = []
        for p, L, b, n in zip( self.tags, slice_lengths, bounds, np.arange(len(bounds)) ):
            [keep_theta_bound.append(self.is_resolved[p]) for counter in np.arange(L)]
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
                else:
                    last = slices[-1][1].stop
                    slices.append((p, slice(last, last+L)))
                    keep_theta_bound[-1] = True

        self.n_params = slices[-1][1].stop
        self.slices = collections.OrderedDict(slices)
        self.theta_bounds = np.array([np.array(b) for n, b in enumerate(bounds) if keep_theta_bound[n]], dtype=float)
        self.bounds = bounds

        assert self.n_params==len(self.theta_bounds), 'self,n_params!=len(self.theta_bounds)'
        self.assign_theta_functions()

    def update_plasma_theta(self, theta):
        plasma_theta = []
        for p in self.slices.keys():
            plasma_theta.append((p, theta[self.slices[p]]))
        self.plasma_theta = collections.OrderedDict(plasma_theta)

    def update_plasma_state(self, theta):
        assert len(theta)==self.n_params, 'len(theta)!=self.n_params'
        self.update_plasma_theta(theta)

        plasma_state = []
        for p, f_check in zip(self.slices.keys(), self.function_check):
            values = np.array(theta[self.slices[p]]).flatten()
            if f_check:
                values = self.theta_functions[p](values)
            plasma_state.append((p, values))

        self.plasma_state = collections.OrderedDict(plasma_state)
        self.plasma_state['main_ion_density'] = self.plasma_state['electron_density'] - self.calc_total_impurity_electrons(False)

    def get_theta_functions(self, profile_function=None, cal_functions=None, calwave_functions=None, background_functions=None):

        if profile_function is None:
            x = np.linspace(1, 9, 5)
            profile_function = MeshLine(x=x, zero_bounds=-2, bounds=[0, 10], log=True)
            self.profile_function = arb_obj(electron_density=profile_function,
                                            electron_temperature=profile_function,
                                            number_of_variables_ne=len(x),
                                            number_of_variables_te=len(x),
                                            bounds_ne=[11, 16], bounds_te=[-1, 2])
        else:
            self.profile_function = profile_function

        tmp_func = arb_obj_single_input(number_of_variables=1, bounds=[-5, 20])
        if cal_functions is None:
            self.cal_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.cal_functions = cal_functions

        if background_functions is None:
            self.background_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.background_functions = background_functions

        tmp_func = arb_obj_single_input(number_of_variables=1, bounds=[-1e-5, 1e-5])
        if calwave_functions is None:
            self.calwave_functions = [tmp_func for num in np.arange(self.num_chords)]
        else:
            self.calwave_functions = calwave_functions



    def assign_theta_functions(self):
        theta_functions = [self.cal_functions, self.calwave_functions, self.background_functions,
                           [self.profile_function.electron_density, self.profile_function.electron_temperature],
                           [power10 for tag in self.tags if any([tag.startswith(s+'_') for s in self.species]) or 'X_' in tag]]
        theta_functions = np.concatenate(theta_functions).tolist()
        # theta_functions = np.concatenate( [f if type(f)!=list else f for f in theta_functions] ).tolist()

        theta_functions_tuples = []
        function_tags = ['cal', 'back', 'electron_density', 'electron_temperature']
        function_tags_full = [tag for tag in self.tags if any([check in tag for check in function_tags]) or \
                                                          any([tag.startswith(s+'_') for s in self.species]) or \
                                                          'X_' in tag]

        self.function_check = [any([check in tag for check in function_tags]) or
                               any([tag.startswith(s+'_') for s in self.species]) or
                               ('X_' in tag) for tag in self.tags]
        # self.function_check = [any([check in tag for check in function_tags]) for tag in plasma.tags]

        assert len(function_tags_full)==len(theta_functions), 'len(function_tags_full)!=len(theta_functions)'

        for tag, func in zip(function_tags_full, theta_functions):
            theta_functions_tuples.append((tag, func))

        self.theta_functions = collections.OrderedDict(theta_functions_tuples)

    def calc_total_impurity_electrons(self, set_attribute=True):
        total_impurity_electrons = 0

        ne = self.plasma_state['electron_density']
        te = self.plasma_state['electron_temperature']

        for species in self.impurity_species:
            tau_tag = species+'_tau'
            dens_tag = species+'_dens'

            tau = self.plasma_state[tau_tag]
            conc = self.plasma_state[dens_tag]
            tau = np.zeros(len(ne)) + tau

            tmp_in = (tau, ne, te)
            average_charge = np.exp(self.impurity_average_charge[species](tmp_in))

            total_impurity_electrons += average_charge * conc

        if set_attribute:
            self.total_impurity_electrons = np.nan_to_num(total_impurity_electrons)
        else:
            return np.nan_to_num(total_impurity_electrons)

    def build_impurity_average_charge(self):
        interps = []

        for species in self.impurity_species:
            interps.append( (species, self.build_impurity_electron_interpolator(species)) )

        self.impurity_average_charge = dict(interps)

    def build_impurity_electron_interpolator(self, species):

        te = self.adas_plasma_inputs['te']
        ne = self.adas_plasma_inputs['ne']
        tau = self.adas_plasma_inputs['tau']

        elem, ion = self.species_to_elem_and_ion(species)
        ionp1 = ion + 1

        z_eff = ion*self.impurity_ion_bal[elem][:, :, :, ion] + ionp1*self.impurity_ion_bal[elem][:, :, :, ionp1]

        return RegularGridInterpolator((tau, ne, te), np.log(z_eff), bounds_error=False)

    def get_impurity_ion_bal(self):

        ion_bals = []

        te = self.adas_plasma_inputs['te']
        ne = self.adas_plasma_inputs['ne']
        tau = self.adas_plasma_inputs['tau']

        for elem in self.impurities:
            num_ions=get_number_of_ions(elem)
            bal = np.zeros((len(tau), len(ne), len(te), num_ions))
            for t_counter, t in enumerate(tau):
                with HiddenPrints():
                    out, _ = run_adas406(year=96, elem=elem, te=te, dens=ne, tint=t, all=True)
                bal[t_counter, :, :, :] = out['ion'].clip(1e-50)

            ion_bals.append((elem, bal))

        self.impurity_ion_bal = dict(ion_bals)

    def build_impurity_tec(self, file, exc, rec, elem, ion):

        te = self.adas_plasma_inputs['te']
        ne = self.adas_plasma_inputs['ne']
        tau = self.adas_plasma_inputs['tau']
        big_ne = self.adas_plasma_inputs['big_ne']

        with HiddenPrints():
            pecs_exc, _ = read_adf15(file, exc, te, ne, all=True)  # (te, ne), _
            pecs_rec, _ = read_adf15(file, rec, te, ne, all=True)  # (te, ne), _

        tec406 = np.zeros((len(tau), len(ne), len(te)))

        for t_counter, t in enumerate(tau):
            ionbal = self.impurity_ion_bal[elem]
            tec406[t_counter, :, :] = big_ne*(pecs_exc.T*ionbal[t_counter, :, :, ion] +
                                              pecs_rec.T*ionbal[t_counter, :, :, ion+1])

        # log10 is spitting out errors ::( but it still runs ::)
        # What about scipy.interpolate.Rbf ? # TODO - 1e40 REEEEEEEEEEEE
        return RegularGridInterpolator((tau, ne, te), np.log(tec406.clip(1e-40)), bounds_error=False)

    def get_impurity_tecs(self):
        tecs = []
        for species in self.impurity_species:
            for line in self.input_dict[species].keys():
                line_str = str(line).replace(', ', '_')
                for bad_character in ['[', ']', '(', ')']:
                    line_str=line_str.replace(bad_character, '')
                line_tag = species + '_' + line_str
                file = self.input_dict[species][line]['pec']
                exc = self.input_dict[species][line]['exc_block']
                rec = self.input_dict[species][line]['rec_block']
                elem, ion = self.species_to_elem_and_ion(species)
                tec = self.build_impurity_tec(file, exc, rec, elem, ion)
                tecs.append((line_tag, tec))

        self.impurity_tecs = dict(tecs)

    @staticmethod
    def species_to_elem_and_ion(species):
        split_index_array = np.where([t=='_' for t in species])[0]
        split_index = split_index_array[0]
        assert len(split_index_array)==1, species+' does not have the correct format: elem_ionstage'
        return species[:split_index], int(species[split_index+1:])

    def get_hydrogen_pecs(self):
        te = self.adas_plasma_inputs['te']
        ne = self.adas_plasma_inputs['ne']

        pecs = []
        for species in self.hydrogen_species:
            for line in self.input_dict[species].keys():
                line_tag = species+'_'+str(line).replace(', ', '_')
                file = self.input_dict[species][line]['pec']
                exc = self.input_dict[species][line]['exc_block']
                rec = self.input_dict[species][line]['rec_block']

                for block, r_tag in zip([exc, rec], ['exc', 'rec']):
                    rates, _ = read_adf15(file, block, te, ne, all=True)  # (te, ne), _# TODO - 1e50 alarm
                    pecs.append( (line_tag+'_'+r_tag, RectBivariateSpline(ne, te, np.log(rates.T.clip(1e-50))).ev) )
                # return

        self.hydrogen_pecs = dict(pecs)

    def is_theta_within_bounds(self, theta):
        return [(b.min()<r) and (r<b.max()) for b, r in zip(self.theta_bounds, theta)]


if __name__ == '__main__':

    from baysar.input_functions import make_input_dict

    num_chords = 1
    wavelength_axis = [np.linspace(3900, 4300, 512)]
    experimental_emission = [np.array([1e12*np.random.rand() for w in wavelength_axis[0]])]
    instrument_function = [np.array([0, 1, 0])]
    emission_constant = [1e11]
    species = ['D', 'N']
    ions = [ ['0'], ['1', '2', '3'] ]
    noise_region = [[4040, 4050]]
    mystery_lines = [ [[4070], [4001, 4002]],
                       [[1], [0.4, 0.6]]]

    input_dict = make_input_dict(wavelength_axis=wavelength_axis, experimental_emission=experimental_emission,
                                instrument_function=instrument_function, emission_constant=emission_constant,
                                noise_region=noise_region, species=species, ions=ions,
                                mystery_lines=mystery_lines, refine=[0.01],
                                ion_resolved_temperatures=False, ion_resolved_tau=True)

    plasma = PlasmaLine(input_dict)

    # rand_theta = np.random.rand(plasma.n_params)
    # plasma(rand_theta)
    #
    # # for k in plasma.plasma_state_tags.keys():
    # for k in plasma.slices.keys():
    #     print(k, plasma.plasma_state[k], type(plasma.plasma_state[k]), plasma.slices[k],
    #           plasma.theta_bounds[plasma.slices[k]], type(plasma.theta_bounds[plasma.slices[k]]))
    #
    # k = 'main_ion_density'
    # print(k, plasma.plasma_state[k])
    # print(plasma.bounds)
    # print(plasma.is_theta_within_bounds(rand_theta))


    pass
