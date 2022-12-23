#
#
#
#
#
# Imports
from argparse import ArgumentParser
from numba import float32, njit, vectorize
from numpy import array, ndarray, load, prod, round, savez, square, sum
from pathlib import Path
from time import time


# Variables
INTER_BAYSAR_PATH: Path = Path(__file__).parent
DEFAULT_INPUT: Path = INTER_BAYSAR_PATH / 'data' / 'test_formated_spectra.npz'
OUTPUT_PATH: Path = INTER_BAYSAR_PATH / 'output'
default_spectra_database_file: Path = OUTPUT_PATH / 'test_spectra_database.npz'

# Args parse
parser = ArgumentParser()
parser.add_argument('--spectra_database', type=str, default=default_spectra_database_file)
parser.add_argument('--save', type=str, default='output/test_spectra_fit')
args = parser.parse_args()


# Functions
# from numba.cuda import jit
# @jit('void(float32[:], float32[:], float32[:])')
# def _calculate_posterior(data, grid, error):
#     res = (data - grid) / error
#     res_squared = res * res
#     logp = - 0.5 * res_squared
#     return logp


@njit
def calculate_posterior(data, grid, error):
    res = (data - grid) / error
    return - 0.5 * square(res).sum(1)
    # return _calculate_posterior(data, grid, error).sum(1)


def main() -> None:
    # Start timer
    start_time: float = time()
    # Load data to fit
    formatted_data: dict[str, ndarray] = load(DEFAULT_INPUT)
    spectra: ndarray = formatted_data['spectra']
    sigma: ndarray = formatted_data['error']
    # Load spectral database
    # training_data = np.random.random((sample_size, pixel))
    # spectra_database = np.load('/home/dgahle/thesis/inter_baysar/test_spectra_database.npz')
    spectra_database: dict[str, ndarray] = load(args.spectra_database)
    training_data: ndarray = spectra_database['spectra']
    # Calculate the Posterior
    logp = calculate_posterior(data=spectra[None, :, :, :], grid=training_data[:, :, None, None], error=sigma)
    # logp, res = out
    # Print data
    print(f'logp.shape {logp.shape}')
    # super_size = array(res.shape, dtype=float)
    # super_size = np.array([training_data.shape[0], pixel, num_chords, time_num], dtype=float)
    runtime = round(time() - start_time / 5, 3)
    n_comparisons = round(prod(logp.shape) * 1e-6, 3)
    print( f"Run in {runtime} s ({n_comparisons}M comparisons)" )
    # print( f"{super_size.tolist()} in {runtime} s ({n_comparisons}M comparisons)" )
    # ram_estimation = (32/4) * prod(super_size) * 1e-9
    # # ram_estimation = 32 * np.prod(np.array((err.shape), dtype=float)) * 1e-9
    # print(f"Memory requirement {round(ram_estimation, 3)} GB")
    # Save output
    pdf = {'logp': logp, 'theta': spectra_database['theta']}
    if args.save is not None:
        savez(args.save, **pdf, allow_pickle=True)
        print(f"Saved '{args.save}.npy'!")


if __name__ == "__main__":
    main()
