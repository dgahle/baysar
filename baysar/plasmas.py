import numpy as np

import os, sys, io
import copy

from baysar.lineshapes import Gaussian, Eich
from baysar.linemodels import build_tec406

from baysar.spectrometers import within

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

        if type(theta) in (float, int):
            return theta
        else:
            return theta[0]


class PlasmaLine():

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

        self.los = self.profile_function.electron_density.x

        self.hydrogen_isotopes = ['H', 'D', 'T']
        self.hydrogen_isotope = any(['H' == t or 'D' == t or 'T' == t
                                     for t in list(self.input_dict['physics'].keys())])

        self.build_tags_slices_and_bounds()
        self.build_impurity_average_charge()

        self(np.random.rand(self.n_params))

    def __call__(self, theta):

        return self.update_plasma_state(theta)

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
            for s in self.species:
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
            values = theta[self.slices[p]]

            if f_check:
                plasma_state.append((p, self.theta_functions[p](values)))
            else:
                plasma_state.append((p, values[0]))

        self.plasma_state = collections.OrderedDict(plasma_state)
        self.plasma_state['main_ion_density'] = self.plasma_state['electron_density'] - self.calc_total_impurity_electrons(False)

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

        ne = self.plasma_state['electron_density']
        te = self.plasma_state['electron_temperature']

        for tmp_element in list(self.input_dict['physics'].keys()):

            if any([tmp_element == t for t in ('D', 'H', 'X')]):
                pass
            else:
                tau_tag = [tag for tag in self.tags if tmp_element in tag and tag.endswith('_tau')]
                dens_tag = [tag for tag in self.tags if tmp_element in tag and tag.endswith('_dens')]

                for t_tag, d_tag in zip(tau_tag, dens_tag):

                    tau = self.plasma_state[t_tag]
                    conc = self.plasma_state[d_tag]

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
    experimental_emission = [np.array([1e12*np.random.rand() for w in wavelength_axes[0]])]
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

    plasma = PlasmaLine(input_dict)

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