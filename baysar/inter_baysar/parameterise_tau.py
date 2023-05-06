#
#
# want to produce and interpolator for <Te> and chi = 1/(ne*tau)
#
#
# Imports
from adas import read_adf11, run_adas406
from baysar.plasmas import get_adf11, get_meta
from matplotlib.pyplot import subplots, legend
from numpy import linspace, logspace, zeros, log10, array, concatenate, savez, log, exp
from numpy import load as load_npz
from os.path import exists
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Union, Optional


# Variables
ATOMIC_DATA_CACHE_PATH: Path = Path(__file__).parent / 'cache'


# Functions and classes
class TauFromTeEms:

    def __init__(self, element: str, charge: int, res: int = 30, tau_res: int = 20, load: bool=False):

        self.element = element
        self.charge = charge
        self.is1 = self.charge + 1
        self.adas_year = 96
        self.te = logspace(-1, 2, res)
        self._ne = 1e13
        self.ne = self._ne + zeros(res)
        self.tau_res = tau_res

        self.get_plasma_conditions()
        self.get_power_rates()
        if not load:
            self.get_te_ems()
            self.get_interpolate()
            self.save(ATOMIC_DATA_CACHE_PATH / f'tau_te_model_{self.element}{self.charge}.npz')

    def get_plasma_conditions(self):

        self.tau_upstream = logspace(1, -5, self.tau_res)
        self.tau_downstream = logspace(1, -2, self.tau_res)

        chi_upstream = 1 / (self._ne * self.tau_upstream)
        chi_downstream = 1 / (self._ne * self.tau_downstream)

        self.chi_upstream = 14 + log10(chi_upstream)
        self.chi_downstream = - (14 + log10(chi_downstream))

    def get_power_rates(self):

        self.plt_adf11 = get_adf11(self.element, self.adas_year, 'plt', adf11_dir='/home/adas/adas/adf11/')
        self.prb_adf11 = get_adf11(self.element, self.adas_year, 'prb', adf11_dir='/home/adas/adas/adf11/')
        self.plt_rates = read_adf11(file=self.plt_adf11, adf11type='plt', is1=self.is1, te=self.te, dens=self.ne)
        self.prb_rates = read_adf11(file=self.prb_adf11, adf11type='prb', is1=self.is1, te=self.te, dens=self.ne)


    def get_te_ems(self):

        # calculat <Te>ems at each tau
        # upstream transport
        te_ems = []
        for tint in self.tau_upstream:
            bal, _ = run_adas406(elem=self.element, te=self.te, dens=self.ne, tint=tint)

            lz_exc = bal['ion'][:, self.charge] * self.plt_rates
            lz_rec = bal['ion'][:, self.is1] * self.prb_rates
            lz = lz_exc + lz_rec
            te_ems.append(self.te.dot(lz / lz.sum()))

        # downstream transport
        te_ems_rec = []
        downstram_meta = get_meta(self.element.capitalize(), -1)
        for tint in self.tau_downstream:
            bal, _ = run_adas406(elem=self.element, te=self.te, dens=self.ne, tint=tint, meta=downstram_meta)

            lz_exc = bal['ion'][:, self.charge] * self.plt_rates
            lz_rec = bal['ion'][:, self.is1] * self.prb_rates
            lz = lz_exc + lz_rec
            te_ems_rec.append(self.te.dot(lz / lz.sum()))

        # clean Te so that the relationship is single valued
        te_ems = array(te_ems)
        te_ems_rec = array(te_ems_rec)
        self.te_ems_upstream = te_ems.clip(min=te_ems[0])
        self.te_ems_downstream = te_ems_rec.clip(max=te_ems_rec[0])

    def get_interpolate(self):

        self.te_ems_full = concatenate([self.te_ems_downstream[0:][::-1], self.te_ems_upstream])
        self.chi_full = concatenate([self.chi_downstream[0:][::-1], self.chi_upstream])

        self.chi_from_te_ems = interp1d(x=self.te_ems_full, y=self.chi_full, kind='linear')

        self.model_te_bounds = (0.2, min(self.te_ems_full.max(), 50))

    def tau_from_chi(self, chi, log:bool = False):
        log_tau = (14 - abs(chi)) - log10(self._ne)
        if log:
            return log_tau
        else:
            return 10 ** log_tau

    def tau_from_te_ems(self, te_ems, log:bool = False):
        chi = self.chi_from_te_ems(te_ems)
        return self.tau_from_chi(chi, log=log)

    def plot_model(self):

        fig, ax = subplots()
        # plotting
        ax.plot(self.chi_full, self.te_ems_full, 'x', label='ADAS')
        res = 100
        te_interp = linspace(*self.model_te_bounds, res)
        ax.plot(self.chi_from_te_ems(te_interp), te_interp, '--', label='Interp1D')
        # format figure
        ax.set_ylabel(f"{self.element.capitalize()}^{self.charge}+: <Te>ems / eV")
        ax.set_xlabel("D")

        ax.set_ylim(bottom=0.1, top=50)
        ax.set_xlim(self.chi_full.min(), self.chi_full.max())
        ax.set_yscale('log')
        # ax.set_xscale('log')

        legend()

        savefig_path: Path = Path(__file__).parent / 'output' / f'tau_model_{self.element}{self.charge}'
        fig.savefig(savefig_path)

    def to_dict(self) -> dict:
        output_dict = {}
        output_dict['element'] = self.element
        output_dict['charge'] = self.charge
        output_dict['te_ems_upstream'] = self.te_ems_upstream
        output_dict['te_ems_downstream'] = self.te_ems_downstream

        return output_dict

    def save(self, file_path: Union[Path, str]):
        savez(file_path, **self.to_dict())

    @classmethod
    def load(cls, file_path: Union[Path, str] = None, **kwargs):
        if file_path is not None:
            loaded_data = load_npz(file_path, allow_pickle=True)
            ion_bal = cls(element=loaded_data['element'][()], charge=loaded_data['charge'][()], load=True)
            ion_bal.te_ems_upstream = loaded_data['te_ems_upstream']
            ion_bal.te_ems_downstream = loaded_data['te_ems_downstream']
            ion_bal.get_interpolate()

            return ion_bal
        elif 'element' in kwargs and 'charge' in kwargs:
            element = kwargs['element']
            charge = kwargs['charge']
            filename: Path = ATOMIC_DATA_CACHE_PATH / f"tau_te_model_{element}{charge}.npz"
            if exists(filename):
                return TauFromTeEms.load(filename)
            else:
                return cls(element=kwargs['element'], charge=kwargs['charge'])



from scipy.interpolate import RectBivariateSpline
class Interp3D:

    def __init__(self, data, x, y, z):

        self._data = data
        self._x = x
        self._y = y
        self._z = z

        self.build_2d_interpolators()

    def __call__(self, theta, *args, **kwargs):
        return self.evaluate_3d_interpolator(*theta)

    def build_2d_interpolators(self):

        self._2d_interpolators = []
        for z_idex, z in enumerate(self._z):
            tmp_interpolator = RectBivariateSpline(self._x, self._y, self._data[:, :, z_idex])
            self._2d_interpolators.append(tmp_interpolator)

    def evaluate_3d_interpolator(self, x, y, z:Union[int, float], evaluation_check:bool=False) -> Union[tuple, array]:
        # use search sorted to find the closed two 2d interpolators (this means z can only be a scalar)
        # calculate the wieghts between the two
        lower_z_weight, upper_z_weight = self.get_z_weights(z)
        # evalute the appropriate 2D interplators
        lower_index, upper_index = self.get_z_indicies(z)
        lower_2d_interp_ev = self._2d_interpolators[lower_index].ev(x, y)
        upper_2d_interp_ev = self._2d_interpolators[upper_index].ev(x, y)
        # calculate the weighted average
        interp3d_ev = lower_z_weight * lower_2d_interp_ev + upper_z_weight * upper_2d_interp_ev
        if evaluation_check:
            return lower_2d_interp_ev, interp3d_ev, upper_2d_interp_ev
        else:
            return interp3d_ev
    
    def evaluation_check(self, theta:Union[tuple, list]) -> array:
        return self.evaluate_3d_interpolator(*theta, evaluation_check=True)

    def get_z_indicies(self, z:Union[float, int]):
        upper_index = self._z.searchsorted(z)
        return upper_index - 1, upper_index

    def get_z_weights(self, z:Union[float, int]):
        lower_index, upper_index = self.get_z_indicies(z)
        upper_z = self._z[upper_index]
        lower_z = self._z[lower_index]
        dz =  upper_z - lower_z
        weights = (z - array([lower_z, upper_z])) / dz

        return 1 - abs(weights)

    def to_dict(self):
        return {'x': self._x, 'z': self._z, 'z': self._z}

    def save(self, file_path: Union[Path, str]):
        savez(file_path, **self.to_dict())
        raise RuntimeError("Method not implimented!")

    @classmethod
    def load(self, file_path: Union[Path, str]):
        dict = load_npz(file_path)
        raise RuntimeError("Method not implimented!")


from baysar.line_data import adas_line_data
from baysar.plasmas import get_adf11
adf11_types = ['scd', 'acd', 'plt', 'prb']
def get_adf11s(element:str) -> dict:
    adf11 = {}
    elem_adas_yr = adas_line_data[element]['ionisation_balance_year']

    for adf11_type in adf11_types:
        if adf11_type in adas_line_data[element]:
            adf11[adf11_type] = adas_line_data[element][adf11_type]
        elif elem_adas_yr is not None:
            adf11[adf11_type] = get_adf11(element, yr=elem_adas_yr, type=adf11_type)
        else:
            raise ValueError(f"{element} data is missing references for {adf11_type.capitalize()} data!")

    return adf11

def get_array_range(data:array) -> list:
    return [data.min(), data.max()]

from numpy import moveaxis
class Adas406Interp:

    def __init__(self, element:str, res: int = 30, tau_res:int = 20, adf11s:dict=None, loading:bool=False):

        self.element = element.lower().capitalize()
        self.te_res = res
        self.tau_res = tau_res

        self.adf11s = adf11s
        if self.adf11s is None:
            self.adf11s = get_adf11s(self.element)

        if not loading:
            self.get_plasma_conditions()
            self.run_adas406()
            self.build_interpolators()

    def __call__(self, ne, te, chi, *args, **kwargs):
        
        return self.evaluate(ne, te, chi)

    def evaluate(self, ne, te, chi):

        ion_bal = []
        for ion in self.charge_state_interp3d:
            ion_bal.append( exp(ion([ne, te, chi])) )

        return moveaxis(array(ion_bal), [0], [-1])

    def evaluation_check(self, ne, te, chi) -> list:
        ion_bal = []
        ion_bal_lower = []
        ion_bal_upper = []

        for ion in self.charge_state_interp3d:
            lower, mid, upper = ion.evaluation_check([ne, te, chi])
            ion_bal.append(mid)
            ion_bal_lower.append(lower)
            ion_bal_upper.append(upper)

        ion_bals = [ion_bal_lower, ion_bal, ion_bal_upper]
        return [exp(moveaxis(array(bal), [0], [-1])) for bal in ion_bals]

    def get_plasma_conditions(self):

        self.te = logspace(-1, 2, self.te_res)
        self.ne = logspace(12, 15, self.te_res//3)

        self.tau_upstream = logspace(1, -5, self.tau_res)
        self.tau_downstream = logspace(1, -2, self.tau_res)

        self._ne = 1e13
        chi_upstream = 1 / (self._ne * self.tau_upstream)
        chi_downstream = 1 / (self._ne * self.tau_downstream)

        self.chi_upstream = 14 + log10(chi_upstream)
        self.chi_downstream = - (14 + log10(chi_downstream))
        self.chi_long = concatenate((self.chi_downstream[1:][::-1], self.chi_upstream))

        self.get_bounds()

    def tau_to_chi(self, tau:Union[int, float], log:bool=False):
        return

    def get_bounds(self):
        self.bounds = array([get_array_range(self.ne),
                             get_array_range(self.te),
                             get_array_range(self.chi_long) ])

    def run_adas406(self):

        ionisation_balances = []
        # downstream transport
        downstram_meta = get_meta(self.element.capitalize(), -1)
        for tint in self.tau_downstream[1:][::-1]:
            bal, _ = run_adas406(files=self.adf11s, elem=self.element, te=self.te, dens=self.ne, tint=tint, meta=downstram_meta, all=True)
            ionisation_balances.append(bal['ion'])

        # upstream transport
        for tint in self.tau_upstream:
            bal, _ = run_adas406(files=self.adf11s, elem=self.element, te=self.te, dens=self.ne, tint=tint, all=True)
            ionisation_balances.append(bal['ion'])

        self.ionisation_balances = moveaxis(array(ionisation_balances), [0, 1, 2, 3], [2, 0, 1, 3])

        shape_check_reference = (self.ne.shape[0], self.te.shape[0])
        if self.ionisation_balances.shape[:2] != shape_check_reference:
            raise ValueError(self.ionisation_balances.shape, shape_check_reference)

    def build_interpolators(self):

        interp = []
        interp_3d_inputs = {}
        interp_3d_inputs['x'] = self.ne
        interp_3d_inputs['y'] = self.te
        interp_3d_inputs['z'] = self.chi_long
        for charge in range(self.ionisation_balances.shape[-1]):
            interp_3d_inputs['data'] = log(self.ionisation_balances[:, :, :, charge].clip(1e-10))
            interp.append( Interp3D(**interp_3d_inputs) )

        self.charge_state_interp3d = interp

    def plot_check(self, ne, te, chi):

        # get data to plot
        ion_bal_lower, ion_bal, ion_bal_upper = self.evaluation_check(ne, te, chi)

        # plot thte interpoaltion over the actual data and plot the 3D inter
        fig, ax = subplots()
        # plot data

        for charge in range(ion_bal.shape[-1]):
            colour = f"C{charge}"
            ax.plot(te, ion_bal[:, charge], color=colour)
            ax.plot(te, ion_bal_lower[:, charge], '--', color=colour)
            ax.plot(te, ion_bal_upper[:, charge], '-.', color=colour)

        # format plot
        ax.set_xlim(get_array_range(te))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Fractional Abundance {self.element}")
        ax.set_xlabel("Te / eV")

        #  show/save plot
        fig.show()

    def to_dict(self) -> dict:
        output_dict = {}
        # need to save the atomic data for reference (self.adf11s)
        # need to save ion)_beal (self.ionisation_balances)
        output_dict['element'] = self.element
        output_dict['adf11s'] = self.adf11s
        output_dict['ne'] = self.ne
        output_dict['te'] = self.te
        output_dict['chi_long'] = self.chi_long
        output_dict['ionisation_balances'] = self.ionisation_balances

        return output_dict


    def save(self, file_path:Union[Path, str]):
        savez(file_path, **self.to_dict())

    @classmethod
    def load(cls, file_path:Union[Path, str]):
        loaded_data = load_npz(file_path, allow_pickle=True)
        ion_bal = cls(element=loaded_data['element'][()], adf11s=loaded_data['adf11s'][()], loading=True)
        # ion_bal.get_plasma_conditions()
        ion_bal.ne = loaded_data['ne']
        ion_bal.te = loaded_data['te']
        ion_bal.chi_long = loaded_data['chi_long']
        ion_bal.ionisation_balances = loaded_data['ionisation_balances']
        ion_bal.build_interpolators()

        return ion_bal


if __name__=='__main__':
    ions: dict = dict(
        c=[2],
        n=[1, 2, 3]
    )
    element: str
    charge: int
    for element in ions:
        for charge in ions[element]:
            tau_model = TauFromTeEms.load(element=element, charge=charge)
            tau_model.plot_model()
    #
    # # ion_bal = Adas406Interp(element=element, tau_res=3)
    # # check_inputs = [[2e13, 5e13], [5, 7], 1]
    # # check0 = ion_bal(*check_inputs)
    # file = 'tmp.npz'
    # # ion_bal.save(file)
    # # del ion_bal
    # ion_bal = Adas406Interp.load(file)
    # res = 50
    # ne = 2e13 + zeros(res)
    # te = linspace(1, 20, res)
    # chi = 2
    # check_inputs = (ne, te, chi)
    # ion_bal.plot_check(*check_inputs)
    # # check1 = ion_bal(*check_inputs)
    # # print( check0==check1 )
    #
    # # tau_model = TauFromTeEms(element=element, charge=charge)
    # # tau_model.plot_model()
    #
    # # need to impliment tau_to_chi and then update build_spectral_database.py then see how big a sample size is needed
    #
    #
