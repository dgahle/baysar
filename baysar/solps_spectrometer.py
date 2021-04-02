import numpy as np
import matplotlib.pyplot as plt

# 1) Get solps data from mds
from MDSplus import Connection as MDSConnection
from scipy.interpolate import RectBivariateSpline
# Load plasma from SOLPS model
# mds_number = 150141 # 121829 #
# 2) Get line of sight plasmas


def get_weights(self, r, z, error=1):
    r_log_weights = np.square((self.r_points - r)/error)
    z_log_weights = np.square((self.z_points - z)/error)
    weights = np.exp(-0.5*np.square(r_log_weights + z_log_weights))
    k_norm = weights.sum()

    return weights / k_norm



def _get_los_coordinates(pupil, angle, bounds=None):
    l = np.arange(0, 200, 1e-1)
    r = pupil[0] - l * np.cos(angle)
    z = pupil[1] + l * np.sin(angle)

    los, x_los = np.array([r, z]), l

    if bounds is not None:
        checks = np.where([min(bounds[0]) < r < max(bounds[0]) and min(bounds[1]) < z < max(bounds[0]) for r, z in zip(*los)])
        los = los[:, checks].sum(1)
        x_los = x_los[checks]

    return los, x_los

def get_los_coordinates(self, los=15):
    # check within square qrid
    pupil = self.spectrometer['pupil']
    angles = self.spectrometer['angles']
    los, x_los = _get_los_coordinates(pupil, angles[los])
    # check within irregular grid
    checks=[]
    los_ne = []
    for i in range(los.shape[-1]):
        tmp_weight = get_weights(self,*los[:, i], 0.2)
        tmp_ne = (tmp_weight * self._ne).sum()
        if not np.isnan(tmp_ne):
            checks.append(True)
            los_ne.append(tmp_ne)
        else:
            checks.append(False)

    los = los[:, np.where(checks)].sum(1)
    x_los = x_los[checks]

    checks = np.where([abs(d_l) > 2e11 for d_l in np.diff(los_ne)])

    los_fin = los[:, 1:][:, checks].sum(1)
    x_los = x_los[1:][checks]

    return los_fin, x_los

def _get_los_data(self, los, grid):
    los_x = []
    for i in range(los.shape[-1]):
        tmp_weight = get_weights(self,*los[:, i], 0.2)
        tmp = (tmp_weight * grid).sum()
        if not np.isnan(tmp):
            los_x.append(tmp)

    return np.array(los_x)

def get_los_data(self, los):
    los_ne = []
    los_te = []
    for i in range(los.shape[-1]):
        tmp_weight = get_weights(self,*los[:, i], 0.2)
        tmp_ne = (tmp_weight * self._ne).sum()
        tmp_te = (tmp_weight * self._te).sum()
        if not np.isnan(tmp_ne):
            los_ne.append(tmp_ne)
            los_te.append(tmp_te)

    return np.array(los_ne), np.array(los_te)



def get_los_species(self, los):
    species_dens_old = {}
    species_dens = {}
    neutral_index=0
    for ion_index, species in enumerate(self.species_list):
        if species.endswith("0") and "+" not in species:
            z0 = species[:-1]
            z = species[-1]
            species_dens_old[f"{z0}_{z}"] = _get_los_data(self, los, self._atomic_dens[neutral_index])
            neutral_index+=1
            species_dens[z0] = [species_dens_old[f"{z0}_{z}"]]
        else:
            z0, z = species.split("+")
            species_dens_old[f"{z0}_{z}"] = _get_los_data(self, los, self._fluid_dens[ion_index])
            species_dens[z0].append(species_dens_old[f"{z0}_{z}"])

    for z0 in species_dens:
        species_dens[z0] = np.array(species_dens[z0])

    return species_dens, species_dens_old

from baysar.diagnostic_settings import get_spectrometer_settings
class SOLPSPlasma:
    def __init__(self, mds_number, mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'):
        # Setup connection to server
        self.mds = MDSConnection(mds_server)
        self.mds_number = mds_number
        self.mds.openTree('solps', self.mds_number)

        self.tokamak = self.mds.get("\IDENT::TOP:EXP")
        self.spectrometer = get_spectrometer_settings(self.tokamak) # {'pupil': pupil, 'angles': angles}

        self.get_data()
        self.get_los_plasmas()

        self.mds.closeAllTrees()

    def get_data(self):
        # grid coordinates
        self.r_points = self.mds.get('\SNAPSHOT::TOP.GRID:CR').data() * 1e2
        self.z_points = self.mds.get('\SNAPSHOT::TOP.GRID:CZ').data() * 1e2
        # Te and ne
        self._te = self.mds.get('\SOLPS::TOP.SNAPSHOT.TE').data()
        self._ne = self.mds.get('\SOLPS::TOP.SNAPSHOT.NE').data() * 1e-6
        # species
        self.species_list = self.mds.get('\IDENT::TOP:SPECIES').decode('utf-8').split()
        self.elements = [z0[:-1] for z0 in self.species_list if z0.endswith("0")]
        # species dens
        self._atomic_dens = self.mds.get('\SNAPSHOT::TOP:DAB2').data() * 1e-6
        self._atomic_temp = self.mds.get('\SNAPSHOT::TOP:TAB2').data()
        self._fluid_dens = self.mds.get('\SNAPSHOT::TOP:na').data() * 1e-6

    def _get_los_plasma(self, los_index=5):
        # get los
        los, x_los = get_los_coordinates(self, los_index)
        # get ne and Te
        ne, te = get_los_data(self, los)
        # get species
        species_dens, _ = get_los_species(self, los)

        return x_los, ne, te, species_dens

    def get_los_plasmas(self):
        # Setup lists to full (tp be turned into arrays later)
        self.x_los = []
        self.ne = []
        self.te = []
        self.species_dens = {}
        for z0 in self.elements:
            self.species_dens[z0] = []

        # fill lists
        for counter, angle in enumerate(self.spectrometer['angles']):
            print(counter, angle)
            x_los, ne, te, species_dens = self._get_los_plasma(counter)

            if len(x_los) == 0:
                raise ValueError(f"Line of sight does not intersect plasma! (LOS {counter}, angle {angle})")

            self.x_los.append(x_los)
            self.ne.append(ne)
            self.te.append(te)
            for z0 in self.species_dens:
                self.species_dens[z0].append(species_dens[z0])



from adas import read_adf15
from baysar.lineshapes import gaussian_norm
from baysar.linemodels import stehle_param
class EmissionProfile:
    def __init__(self, line_data, solps):
        self.solps = solps
        for key in line_data:
            if key == "wavelenght":
                setattr(self, "wavelength", np.array([line_data[key]]).flatten())
            if key == "jj_frac":
                setattr(self, key, np.array([line_data[key]]).flatten())
            else:
                setattr(self, key, line_data[key])

        self.get_emission_profiles()
        self.get_lineshape()

    def get_emission_profiles(self):
        self.emission_profiles = []
        self.emission_profiles_integrated = []
        self.ems_te = []
        self.ems_ne = []
        for los_counter in range( len(self.solps.x_los) ):
            los = self.solps.x_los[los_counter]
            te = self.solps.te[los_counter]
            ne = self.solps.ne[los_counter]
            charge = int(self.charge)
            n_exc = self.solps.species_dens[self.element][los_counter][charge]
            n_rec = self.solps.species_dens[self.element][los_counter][charge+1]

            pec_exc, _ = read_adf15(te=te, dens=ne, file=self.pec, block=self.exc_block)
            pec_rec, _ = read_adf15(te=te, dens=ne, file=self.pec, block=self.rec_block)
            ems = ne * (n_exc * pec_exc + n_rec * pec_rec)

            self.emission_profiles.append(ems)
            self.emission_profiles_integrated.append( np.trapz(ems, los) )
            self.ems_te.append(ems.dot(te)/ems.sum())
            self.ems_ne.append(ems.dot(ne)/ems.sum())

    def get_lineshape(self):
        self.lineshapes = []
        for los_counter, (ems0, ems_ne, ems_te) in enumerate( zip(self.emission_profiles_integrated, self.ems_ne, self.ems_te) ):
            # Stark shape for hydrogen
            if self.element in ['H', 'D', 'T']:
                n_upper = self.n_upper
                n_lower = self.n_lower
                tmp_lineshapes = ems0 * stehle_param(n_upper, n_lower, self.wavelength.mean(), self.wavelengths, ems_ne, ems_te)
            else:
                # Doppler shape
                tmp_lineshapes = []
                for w0, jj in zip(self.wavelength, self.jj_frac):
                    doppler_width = w0 * 7.715e-5 * np.sqrt(ems_te / self.atomic_mass) # taken from baysar.linemodels
                    tmp_lineshapes.append( ems0 * gaussian_norm(self.wavelengths, w0, doppler_width, jj) )
                tmp_lineshapes = sum(tmp_lineshapes)

            self.lineshapes.append(tmp_lineshapes)

from numpy.random import normal
from baysar.line_data import adas_line_data
from baysar.plasmas import radiated_power
from baysar.lineshapes import gaussian_norm
class SyntheticSpectrometer:
    def __init__(self, mds_number, wavelengths=None, line_data=adas_line_data):
        self.solps = SOLPSPlasma(mds_number)
        self.line_data = line_data
        if wavelengths is None:
            self.wavelengths = np.linspace(3960.0, 4105.0, 1024)
        else:
            self.wavelengths = wavelength

        self._wavelengths = np.linspace(self.wavelengths.min(), self.wavelengths.max(), int(len(self.wavelengths) * 50))
        self._calibration_constant = 1e9

        self.get_lines()
        self.get_spectra()
        self.get_power()

    def get_lines(self):
        self.lines = []
        for z0 in self.solps.elements:
            print(z0)
            for z in self.line_data[z0]['ions']:
                if z in self.line_data[z0]:
                    for l in self.line_data[z0][z]:
                        if type(l) is not str:
                            tmp_l = np.array([l]).flatten()
                            if any([self.wavelengths.min() < jj < self.wavelengths.max() for jj in tmp_l]):
                                print(z0, z, tmp_l)
                                print(self.line_data[z0][z][l])
                                self.line_data[z0][z][l]['element'] = z0
                                self.line_data[z0][z][l]['charge'] = z
                                self.line_data[z0][z][l]['atomic_mass'] = self.line_data[z0]['atomic_mass']
                                self.line_data[z0][z][l]['wavelengths'] = self._wavelengths
                                self.lines.append(EmissionProfile(self.line_data[z0][z][l], self.solps))

    def get_spectra(self):
        # need to add convolution
        # self._instrument_function = gaussian_smoother(length=len(self._wavelengths), width=2e-2*len(self._wavelengths))
        width = self.solps.spectrometer['instrument_width']*len(self._wavelengths)/len(self.wavelengths)
        pixels = np.arange(len(self.wavelengths))
        self._instrument_function_array = gaussian_norm(pixels, pixels.mean(), self.solps.spectrometer['instrument_width'], 1)
        self._instrument_function = gaussian_smoother(length=len(self._wavelengths), asymmetry=len(self.wavelengths), width=width)

        self.spectra = []
        for i in range( len(self.solps.spectrometer['angles']) ):
            tmp_spectra = sum([l.lineshapes[i] for l in self.lines])
            self.spectra.append( self._instrument_function.dot(tmp_spectra) )

        per_sr = 1 /  (4 * np.pi)
        self.spectra = 100 * self._calibration_constant +  per_sr * np.array(self.spectra)


        # need to add noise
        self._error = np.sqrt(np.square(self._calibration_constant) + self._calibration_constant * self.spectra)
        self._noise = normal(loc=0.0, scale=self._error) # , size=None)

        self.spectra = (self.spectra + self._noise).clip(self._calibration_constant)

    def get_power(self):
        self.radiative_power = []

        for chord in range( len(self.solps.spectrometer['angles']) ):
            tmp_los = self.solps.x_los[chord]
            tmp_power = 0 * self.solps.te[chord].copy()
            te = self.solps.te[chord]
            ne = self.solps.ne[chord]
            for z0 in self.solps.elements:
                z0_dens = self.solps.species_dens[z0][chord]
                for z in range(z0_dens.shape[0] - 1):
                    is1 = z + 1
                    n_exc = z0_dens[z]
                    n_rec = z0_dens[is1]

                    if 'plt' in adas_line_data[z0]:
                        adf11_plt = adas_line_data[z0]['plt']
                    else:
                        adf11_plt = None

                    if 'prb' in adas_line_data[z0]:
                        adf11_prb = adas_line_data[z0]['prb']
                    else:
                        adf11_prb = None

                    tmp_power += sum(radiated_power(n_exc, n_rec, ne, te, is1, adf11_plt=adf11_plt, adf11_prb=adf11_prb, elem=z0.lower(),  yr=96, all=False))
                    # print(z0, adf11_plt, adf11_prb, tmp_power.max())

            self.radiative_power.append(np.trapz(tmp_power, tmp_los))

from copy import deepcopy, copy
def get_output_dict(spectrometer, chord):
    spectra = spectrometer.spectra[chord].copy()
    wave = spectrometer.wavelengths.copy()
    cal_curve = spectrometer._calibration_constant
    intfun_9 = spectrometer._instrument_function_array.copy()

    # 2) get reference plasma
    reference_plasma = {}
    reference_plasma['electron_density'] = deepcopy(spectrometer.solps.ne[chord])
    reference_plasma['electron_temperature'] = deepcopy(spectrometer.solps.te[chord])
    tmp_los = deepcopy(spectrometer.solps.x_los[chord])
    tmp_los *= -1
    tmp_los -= tmp_los.min()
    for key in copy([k for k in reference_plasma.keys()]):
        reference_plasma[key+'_los'] = tmp_los

    reference_plasma['elements'] = {}
    for z0 in spectrometer.solps.elements:
        reference_plasma['elements'][z0+'_dens'] = deepcopy(spectrometer.solps.species_dens[z0][chord])

    reference_plasma['mds_number'] = spectrometer.solps.mds_number
    reference_plasma['tokamak'] = spectrometer.solps.tokamak.decode('utf-8')

    # 3 ) get bolo
    bolo = spectrometer.radiative_power[chord]
    bolo_prior_input = [bolo, 0.2 * bolo]

    _output = {'wave': wave, 'spectra': spectra, 'intfun': intfun_9, 'cal_curve': cal_curve, 'bolo': bolo_prior_input, 'plasma_reference': reference_plasma}
    output = {}
    for key in _output:
        output[key] = deepcopy(_output[key])

    return output


from baysar.lineshapes import gaussian_norm
from scipy.sparse import csc_matrix
def gaussian_smoother(length, width, asymmetry=None):
    if asymmetry is None:
        shape = (length, length)
        A = 1
    else:
        shape = (asymmetry, length)
        A = asymmetry / length

    smoother = np.zeros(shape)
    x = np.arange(shape[-1])
    for i in range(shape[0]):
        tmp_gaussian = gaussian_norm(x, int(i / A), width, A)
        smoother[i, :] = tmp_gaussian / tmp_gaussian.sum()

    smoother = csc_matrix(smoother) # *self.dispersion_ratios)
    smoother.eliminate_zeros()

    return smoother
