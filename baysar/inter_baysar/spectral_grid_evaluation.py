from argparse import ArgumentParser

parser = ArgumentParser()

default_spectra_database_file = (
    "/home/dgahle/thesis/inter_baysar/output/test_spectra_database.npz"
)
parser.add_argument(
    "--spectra_database", type=str, default=default_spectra_database_file
)
parser.add_argument("--save", type=str, default="output/test_spectra_fit")


args = parser.parse_args()

from time import time

import numpy as np

start_time = time()

formatted_data = np.load("data/test_formated_spectra.npz")
spectra = formatted_data["spectra"]
sigma = formatted_data["error"]

# training_data = np.random.random((sample_size, pixel))
# spectra_database = np.load('/home/dgahle/thesis/inter_baysar/test_spectra_database.npz')
spectra_database = np.load(args.spectra_database)
training_data = spectra_database["spectra"]

res = (spectra[None, :, :, :] - training_data[:, :, None, None]) / sigma
logp_long = -0.5 * np.square(res)
logp = logp_long.sum(1)

print(f"logp.shape {logp.shape}")
super_size = np.array(res.shape, dtype=float)
# super_size = np.array([training_data.shape[0], pixel, num_chords, time_num], dtype=float)
runtime = np.round(time() - start_time, 3)
n_comparisons = np.round(np.prod(logp.shape) * 1e-6, 3)
print(f"{super_size.tolist()} in {runtime} s ({n_comparisons}M comparisons)")
ram_estimation = (32 / 4) * np.prod(super_size) * 1e-9
# ram_estimation = 32 * np.prod(np.array((err.shape), dtype=float)) * 1e-9
print(f"Memory requirement {np.round(ram_estimation, 3)} GB")

pdf = {"logp": logp, "theta": spectra_database["theta"]}

if args.save is not None:
    np.savez(args.save, **pdf, allow_pickle=True)
    print(f"Saved '{args.save}.npy'!")
