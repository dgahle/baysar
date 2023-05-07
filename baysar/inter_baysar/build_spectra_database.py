from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
    "--initial_distribution", type=str, default="output/test_initial_distribution.npy"
)
parser.add_argument("--ion_charge", type=int, default=1)
parser.add_argument("--element", type=str, default="n")
parser.add_argument("--save", type=str, default=None)
parser.add_argument("--plot", action="store_true")


args = parser.parse_args()

import numpy as np

# load intial theta distributions
initial_distribution = np.load(
    args.initial_distribution
)  # (num, 4) Te, log(ne), log(tau), log(dl.cz)

# load atomic data then calculate emissivities and spectra
from time import time

from OpenADAS import read_adf15, run_adas406

from baysar.input_functions import get_species_data
from baysar.lineshapes import gaussian_norm
from baysar.plasmas import get_meta

is1 = args.ion_charge + 1
meta = get_meta(args.element.capitalize())
species = f"{args.element.capitalize()}_{args.ion_charge}"
get_species_data_args = [[args.element.capitalize()], [str(args.ion_charge)]]
get_species_data_kwargs = {"wavelength_axis": np.array([[3980, 4060]])}
pec_database = get_species_data(*get_species_data_args, **get_species_data_kwargs)[
    species
]


def f(theta, lines):
    te, log_ne, chi, log_dl_cz = theta

    tint = np.power(10, log_tau)
    dens = np.power(10, log_ne)
    # frac, energy = run_adas406(year=96, elem=args.element, meta=meta, tint=tint, te=te, dens=dens)
    # bal = frac['ion'] # (x, charge)
    bal = adas406_interp(*(10**log_ne, te, chi))
    f_exc, f_rec = bal[args.ion_charge : is1 + 1]

    dl_ems = []
    for l in lines:
        adf15 = lines[l]["pec"]
        exc_block = lines[l]["exc_block"]
        rec_block = lines[l]["rec_block"]

        exc_pec, _ = read_adf15(file=adf15, block=exc_block, te=te, dens=dens)
        rec_pec, _ = read_adf15(file=adf15, block=rec_block, te=te, dens=dens)

        tmp_dl_ems = (
            np.power(10, log_dl_cz) * dens**2 * (f_exc * exc_pec + f_rec * rec_pec)
        )
        dl_ems.append(tmp_dl_ems)

    return np.array(dl_ems)


from scipy.io import readsav

wave = 10 * readsav("data/dgahle_32244_rov014_interelm.sav")["wavelength"]
# wave = np.linspace(0, 1, 512)
_cwls = [np.array([pec_database[l]["wavelenght"]]).flatten() for l in pec_database]
cwls = np.concatenate(_cwls)
sigma = 0.8


def g(dl_ems, lines):
    _intensity = [de * lines[l]["jj_frac"] for de, l in zip(dl_ems, lines)]
    intensity = np.concatenate(_intensity)
    res = (cwls[:, None] - wave[None, :]) / sigma
    logp = -0.5 * np.square(res)
    spectra = np.exp(logp) / (np.sqrt(2 * np.pi) * sigma)
    spectra = intensity[:, None] * spectra / (4 * np.pi)
    return spectra.sum(0)


spectra_database = []
from os.path import exists

from numpy import log10
from parameterise_tau import Adas406Interp, TauFromTeEms

print("building spectra data base")
start_time = time()
# load tau_model
file_tau_model = f"tau_te_model_{args.element}{args.ion_charge}.npz"
if exists(file_tau_model):
    tau_model = TauFromTeEms.load(file_tau_model)
else:
    tau_model = TauFromTeEms(element=args.element, charge=args.ion_charge)
    tau_model.save(file_tau_model)
# load adas406_interp
file_adas406_interp = f"adas_406_cache_{args.element}.npz"
if exists(file_adas406_interp):
    adas406_interp = Adas406Interp.load(file_adas406_interp)
else:
    adas406_interp = Adas406Interp(element=args.element)
    adas406_interp.save(file_adas406_interp)
# build initial distribution
for theta in initial_distribution:
    te, log_ne, log_dl_cz = theta
    # log_tau = tau_model.tau_from_te_ems(te, log=True)
    log_tau = tau_model.chi_from_te_ems(te)
    theta_in = te, log_ne, log_tau, log_dl_cz
    tmp_dl_ems = f(theta_in, pec_database)  # LS resolvd
    tmp_spectra = g(tmp_dl_ems, pec_database)  # need to make jj resolvd
    spectra_database.append((tmp_spectra, tmp_dl_ems, theta))

print()
build_time = np.round(time() - start_time, 2)
print(f"Built spectral database in {build_time}s!")

# format and save spectra data base
spectra_database_dict = {}
keys = ["spectra", "dl_ems", "theta"]

print()
for i, k in enumerate(keys):
    spectra_database_dict[k] = np.array([sd[i] for sd in spectra_database])
    print(f"{k}.shape is {spectra_database_dict[k].shape}")

spectra_database_dict["wavelengths"] = wave
if args.plot:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(spectra_database_dict["spectra"].T)
    plt.show()

if args.save is not None:
    save_type = type(args.save)
    if save_type is not str:
        raise TypeError(f"--save must pass a str not a {save_type}")
    else:
        np.savez(args.save, **spectra_database_dict, allow_pickle=True)
        print(f"Saved '{args.save}.npz'!")
