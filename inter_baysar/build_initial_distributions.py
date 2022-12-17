#
#
#
#
#
# Imports
from argparse import ArgumentParser
import numpy as np


# Variables
te_bounds = [1, 20]
log_ne_bounds = [13.5, 15]
log_tau_bounds = [-5, 2]
log_dl_cz_bounds = [-2, 0]


parser = ArgumentParser()

parser.add_argument('--sample_size', type=int, default=100000)
parser.add_argument('--save', type=str, default=None)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--gaussian', action='store_true')

args=parser.parse_args()


# Function
def main() -> None:
    include_tau = False
    if include_tau:
        bounds = np.array([te_bounds, log_ne_bounds, log_tau_bounds, log_dl_cz_bounds]) # param, lims
    else:
        bounds = np.array([te_bounds, log_ne_bounds, log_dl_cz_bounds])  # param, lims

    if args.gaussian:
        mean = bounds.mean(1)
        sigma = (bounds[:, -1] - bounds[:, 0])
        # sigma = np.sqrt(sigma)
        n_params = len(mean)
        cov = np.zeros((n_params, n_params))
        for n, s in enumerate(sigma):
            cov[n, n] = s

        if include_tau:
            tau_te_corr = - 0.85
            tau_te_cov = tau_te_corr * np.sqrt(cov[0, 0]) * np.sqrt(cov[2, 2])
            cov[2, 0] = tau_te_cov
            cov[0, 2] = tau_te_cov

            corr = np.zeros((n_params, n_params))
            from itertools import product
            for i, j in product(range(n_params), range(n_params)):
                corr[i, j] = cov[i, j] / (np.sqrt(cov[i, i]) * np.sqrt(cov[j, j]))


            print(mean)
            # print(bounds)
            # print(sigma)
            print(np.round(cov, 2))
            print(corr)
            print()

        from scipy.stats import norm
        from numpy.random import multivariate_normal
        sample_size = args.sample_size
        initial_dist = multivariate_normal(mean, cov, sample_size)

        for n in range(n_params):
            initial_dist[:, n] = initial_dist[:, n].clip(*bounds[n, :])

    else:
        from numpy import array, moveaxis
        from numpy.random import uniform

        n_params = bounds.shape[0]
        initial_dist = []
        for b in bounds:
            uniform_distribution = uniform(b.min(), b.max(), args.sample_size)
            initial_dist.append(uniform_distribution)

        initial_dist = moveaxis(array(initial_dist), [0, 1], [1, 0])

        check = (initial_dist.shape == (args.sample_size, n_params))
        if not check:
            print(initial_dist.shape, (args.sample_size, n_params))
            raise ValueError()


    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure()

        plt.plot(initial_dist[:, 0], initial_dist[:, 2], 'x', alpha=0.1)

        # plt.ylim(*bounds[2])
        # plt.xlim(*bounds[0])

        plt.show()

    if args.save is not None:
        save_type = type(args.save)
        if save_type is not str:
            raise TypeError(f'--save must pass a str not a {save_type}')
        else:
            np.save(args.save, initial_dist, allow_pickle=True, fix_imports=True)
            print(f"Saved '{args.save}.npy'!")


if __name__ == "__main__":
    main()
