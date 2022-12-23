#
#
#
#
#
# Imports
from adas import read_adf15
from argparse import ArgumentParser
from baysar.lineshapes import gaussian_norm
from baysar.input_functions import get_species_data
from numpy import array, concatenate, exp, load, log10, ndarray, power, savez, sqrt, square, zeros
from os.path import exists
from parameterise_tau import TauFromTeEms, Adas406Interp
from pathlib import Path
from scipy.constants import pi
from scipy.io import readsav
from time import time
from tqdm import tqdm


# Variables
INTER_BAYSAR_PATH: Path = Path(__file__).parent
CACHE_PATH: Path = INTER_BAYSAR_PATH / 'cache'
# Argparse
parser=ArgumentParser()

parser.add_argument('--initial_distribution', type=str, default='output/test_initial_distribution.npy')
parser.add_argument('--ion_charge', type=int, default=1)
parser.add_argument('--element', type=str, default='n')
parser.add_argument('--save', type=str, default=None)
parser.add_argument('--plot', action='store_true')

args=parser.parse_args()


# Functions
atomic_number={'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10}


def get_number_of_ions(element):
    return atomic_number[element]+1


def get_meta(element, index=0):
    num=get_number_of_ions(element)
    meta=zeros(num)
    meta[index]=1
    return meta


def f(theta, lines, log_tau, adas406_interp):
    te, log_ne, chi, log_dl_cz = theta

    tint = power(10, log_tau)
    dens = power(10, log_ne)
    # frac, energy = run_adas406(year=96, elem=args.element, meta=meta, tint=tint, te=te, dens=dens)
    # bal = frac['ion'] # (x, charge)
    bal = adas406_interp(*(10 ** log_ne, te, chi))
    bal_slice: slice = slice(args.ion_charge, args.ion_charge + 2)
    f_exc, f_rec = bal[bal_slice]

    dl_ems = []
    for l in lines:
        adf15 = lines[l]['pec']
        exc_block = lines[l]['exc_block']
        rec_block = lines[l]['rec_block']

        exc_pec = read_adf15(adf15=adf15, block=exc_block, te=te, ne=dens).data
        rec_pec = read_adf15(adf15=adf15, block=rec_block, te=te, ne=dens).data

        tmp_dl_ems = power(10, log_dl_cz) * dens ** 2 * (f_exc * exc_pec + f_rec * rec_pec)
        dl_ems.append(tmp_dl_ems)

    return array(dl_ems)


def g(dl_ems, lines, theta):
    _intensity = [de*lines[l]['jj_frac'] for de, l in zip(dl_ems, lines)]
    intensity = concatenate(_intensity)
    cwls, wave, sigma = theta
    res = (cwls[:, None] - wave[None, :]) / sigma
    logp = - 0.5 * square(res)
    spectra = exp(logp) / (sqrt(2 * pi) * sigma)
    spectra = intensity[:, None] * spectra / (4 * pi)
    return spectra.sum(0)


def get_tau_model() -> TauFromTeEms:
    file_tau_model = CACHE_PATH / f"tau_te_model_{args.element}{args.ion_charge}.npz"
    if exists(file_tau_model):
        tau_model = TauFromTeEms.load(file_tau_model)
    else:
        tau_model = TauFromTeEms(element=args.element, charge=args.ion_charge)
        tau_model.save(file_tau_model)

    return tau_model


def get_ion_bal_model() -> Adas406Interp:
    file_adas406_interp = CACHE_PATH / f"adas_406_cache_{args.element}.npz"
    if exists(file_adas406_interp):
        adas406_interp = Adas406Interp.load(file_adas406_interp)
    else:
        adas406_interp = Adas406Interp(element=args.element)
        adas406_interp.save(file_adas406_interp)

    return adas406_interp


def main() -> None:
    # Load initial theta distributions
    initial_distribution: ndarray = load(args.initial_distribution) # (num, 4) Te, log(ne), log(tau), log(dl.cz)
    # Load atomic data then calculate emissivities and spectra    
    # is1 = args.ion_charge + 1
    # meta = get_meta(args.element.capitalize())
    species = f'{args.element.capitalize()}_{args.ion_charge}'
    get_species_data_args = [[args.element.capitalize()], [str(args.ion_charge)]]
    get_species_data_kwargs = {'wavelength_axis': array([[3980, 4060]])}
    pec_database = get_species_data(*get_species_data_args, **get_species_data_kwargs)[species]
    # Get wavelength axis from experimental data
    wave = 10 * readsav('data/dgahle_32244_rov014_interelm.sav')['wavelength']
    _cwls = [array([pec_database[l]['wavelenght']]).flatten() for l in pec_database]
    cwls = concatenate(_cwls)
    sigma = 0.8
    # Build database
    spectra_database = []
    print("building spectra data base")
    start_time = time()
    # Load atomic models (tau_model and adas406)
    tau_model: TauFromTeEms = get_tau_model()
    adas406_interp: Adas406Interp = get_ion_bal_model()
    # build initial distribution
    for theta in tqdm(initial_distribution):
        te, log_ne, log_dl_cz = theta
        # log_tau = tau_model.tau_from_te_ems(te, log=True)
        log_tau = tau_model.chi_from_te_ems(te)
        theta_in = te, log_ne, log_tau, log_dl_cz
        tmp_dl_ems = f(theta_in, pec_database, log_tau, adas406_interp) # LS resolved
        tmp_spectra = g(tmp_dl_ems, pec_database, theta=[cwls, wave, sigma]) # need to make jj resolved
        spectra_database.append((tmp_spectra, tmp_dl_ems, theta))
    # Callbacks
    print()
    build_time = round(time() - start_time, 2)
    print(f"Built spectral database in {build_time}s!")
    # format and save spectra data base
    spectra_database_dict = {}
    keys = ['spectra', 'dl_ems', 'theta']
    print()
    for i, k in enumerate(keys):
        spectra_database_dict[k] = array([sd[i] for sd in spectra_database])
        print(f'{k}.shape is {spectra_database_dict[k].shape}')
    
    spectra_database_dict['wavelengths'] = wave
    # Save
    if args.save is not None:
        save_type = type(args.save)
        if save_type is not str:
            raise TypeError(f'--save must pass a str not a {save_type}')
        else:
            savez(args.save, **spectra_database_dict, allow_pickle=True)
            print(f"Saved '{args.save}.npz'!")
    # Plot
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(spectra_database_dict['spectra'].T)
        plt.show()


if __name__ == "__main__":
    main()
