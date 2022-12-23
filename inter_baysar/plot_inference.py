#
#
#
#
#
# Imports
from scipy.io import readsav
# import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from scipy.signal import fftconvolve
from baysar.lineshapes import gaussian_norm
from numpy import array, average, cumsum, diff, linspace, load, ndarray, ones, power, sort, square, sqrt


# Variables
PDF_PATH: str = 'output/test_spectra_fit.npz'


# Functions
def get_interval(data, probs, interval=0.50, log=True):

    if log:
        probs = power(10, probs) # [:, 0, :])
        probs = probs / probs.sum(0)[None, :]

    cdf_index: ndarray = probs.argsort(axis=1)
    cdf: ndarray = cumsum(sort(probs), axis=1)
    interval_index: ndarray = square(cdf-(1-interval)).argmin(1)
    # interval_bounds = []
    # for t in range(probs.shape[1]):
    #     tmp_indicies = probs[:, t].argsort()
    #     cdf = cumsum(probs[tmp_indicies,t])
    #     lhs_indicies = square(cdf-(1-interval)).argmin(0)
    #     tmp_interval_thetas = data[tmp_indicies[lhs_indicies:]]
    #     tmp_interval_bounds = array([tmp_interval_thetas.min(0), tmp_interval_thetas.max(0)])
    #     interval_bounds.append(tmp_interval_bounds)

    return array(interval_bounds)


def main() -> None:
    # Load posterior
    pdf_dict = load(PDF_PATH, allow_pickle=True)
    # Format and unpack
    logp = pdf_dict['logp']
    logp_norm = logp[:, 0, :] - logp.max(0)[0, None, :]
    thetas = pdf_dict['theta']
    # Get the modes
    mode_indicies = logp.argmax(0)[0]
    mode_thetas = array([thetas[i] for i in mode_indicies])
    mode_thetas_indicies = []
    for t in range(logp_norm.shape[-1]):
        mode_thetas_indicies.append(logp_norm[:,t].argmax())
    # Format modes
    mode_thetas_indicies = array(mode_thetas_indicies)
    mode_thetas_alt = thetas[mode_thetas_indicies]
    # Unpack by parameter
    te = mode_thetas_alt[:, 0]
    ne = power(10, mode_thetas_alt[:, 1])
    # tau = power(10, mode_thetas_alt[:, 2])
    dl_cz = power(10, mode_thetas_alt[:, -1])
    # dl_cz = power(10, mode_thetas_alt[:, 3])
    # Get High Density Interval
    interval_bounds = get_interval(thetas, logp[:, 0, :], interval=0.30)
    # Variables for plotting
    time = readsav('data/dgahle_32244_rov014_interelm.sav')['time']
    time_res = diff(time).mean()
    x = linspace(-150, 150, 301)
    smoother = gaussian_norm(x, x.mean(), 0.1/time_res, 1)
    # Plot
    fig, ax = plt.subplots(2, 2, sharex=True)
    ax[0, 0].plot(time, fftconvolve(mode_thetas[:, 0], smoother, mode='same'))
    ax[1, 0].plot(time, fftconvolve(power(10, mode_thetas[:, 1]), smoother, mode='same'))
    # ax[0, 1].plot(time, fftconvolve(power(10, mode_thetas[:, 2]), smoother, mode='same'))
    ax[1, 1].plot(time, fftconvolve(power(10, mode_thetas[:, -1]), smoother, mode='same'))
    te_grid = thetas[:, 0, None]*ones(len(time))[None, :]
    time_grid = array([time for t in range(thetas.shape[0])])
    # Format
    ax[0, 0].set_ylabel('Te / eV')

    axs = [ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]]
    ps = range(thetas.shape[1])
    ps = [0, 1, -1]
    for p in ps:
        p_array = thetas[:, p]
        p_grid = array([p_array for t in time]).T

        weights = power(10, logp[:, 0])
        p_mean = average(p_grid, weights=weights, axis=0)
        variance = average((p_grid-p_mean)**2, weights=weights, axis=0)
        std = sqrt(variance)

        lower_bounds = interval_bounds[:, 0, p]
        upper_bounds = interval_bounds[:, 1, p]

        if not p == 0:
            p_mean = power(10, p_mean)
            lower_bounds = power(10, lower_bounds)
            upper_bounds = power(10, upper_bounds)

        k_p3 = (1e2 / 1)
        if p == 3: # or p == -1:
            p_mean = k_p3 * p_mean
            lower_bounds = k_p3 * lower_bounds
            upper_bounds = k_p3 * upper_bounds

        axs[p].plot(time, p_mean, color='C1', alpha=0.3)
        # axs[p].plot(time, fftconvolve(p_mean, smoother, mode='same'), color='C2')
        # axs[p].plot(time, fftconvolve(lower_bounds, smoother, mode='same'), '--', color='C2')
        # axs[p].plot(time, fftconvolve(upper_bounds, smoother, mode='same'), '--', color='C2')
        # axs[p].plot(time, std + fftconvolve(p_mean, smoother, mode='same'), '--', color='C2')
        # axs[p].plot(time, - std + fftconvolve(p_mean, smoother, mode='same'), '--', color='C2')



    ax[1, 0].set_ylabel('ne / cm-3')
    ax[0, 1].set_ylabel('tau / s')
    ax[0, 1].set_yscale('log')
    ax[1, 1].set_ylabel('cz / cm %')
    # ax[1, 1].set_yscale('log')
    for a in ax.flatten():
        a.set_xlim(0, time.max())

    for a in ax[1]:
        a.set_xlabel('time / s')

    fig.tight_layout()
    fig.savefig('output/inference_plot')

    plt.show()


if __name__ == "__main__":
    main()
