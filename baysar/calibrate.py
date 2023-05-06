import numpy as np


def lorentzian(x, x0, width, area):
    square_res = ((x - x0) / width) ** 2
    inv_peak = np.pi * width * (1 + square_res)
    return area / inv_peak


from scipy.optimize import fmin_l_bfgs_b
from scipy.signal import fftconvolve

from baysar.lineshapes import gaussian_norm
from tulasa.general import close_plots, plot


def fit_peak(x, y, instrument_function=None):
    # get error
    y_err = np.sqrt(y.min() * (y + 1))

    # set up peak fit
    approx_area = np.trapz(y, x)
    x0 = [x.mean(), 0.5 * (x.max() - x.min()), approx_area]
    bounds = [
        (x.min(), x.max()),
        (1, x.max() - x.min()),
        (0.2 * approx_area, 5 * approx_area),
    ]

    # build cost function
    def func(x0):
        fm = gaussian_norm(x, *x0)
        if instrument_function is not None:
            fm = fftconvolve(fm, instrument_function, mode="same")
        return 0.5 * np.square((y - fm) / y_err).sum()

    # run fit
    x_opt, f, d = fmin_l_bfgs_b(
        func, x0, approx_grad=True, bounds=bounds, maxfun=150000, maxiter=150000
    )
    # print(x_opt, f)
    # plot([y, gaussian_norm(x, *x_opt)], multi='fake')
    return x_opt


def clip_ends(x, y):
    """
    Isolates the peak of the data and clips it out of the array.

    :param x:
    :param y:
    :return:
    """
    yargmax = y.argmax()
    lhs_gradients = np.diff(y[:yargmax][::-1])
    rhs_gradients = np.diff(y[yargmax:])

    lhs_checks = lhs_gradients[::-1] > 0
    if any(lhs_checks):
        lhs_index = np.where(lhs_checks)[0][-1] + 1
    else:
        lhs_index = 0

    rhs_checks = rhs_gradients > 0
    if any(rhs_checks):
        rhs_index = yargmax + np.where(rhs_checks)[0][0] + 1
    else:
        rhs_index = len(y)

    yslice = slice(lhs_index, rhs_index)

    return x[yslice], y[yslice]


from baysar.tools import clip_data


def get_pixel_centre(
    spectra, wavelengths, reference, window=None, instrument_function=None
):
    if window is None:
        window = np.array([-1.5, 1.5])
    # get spectral window
    x0, y0 = clip_data(wavelengths, spectra, reference + window)
    # remove funny ends
    x1, y1 = clip_ends(x0, y0)
    # fit peak (cwl, width, height)
    cwl, _, _ = fit_peak(x1, y1, instrument_function)
    # # lhs_shift=5
    # # return pixel centre
    # lhs=np.where(wavelengths==x1[0])[0][0]
    # pixel_centre=lhs+lhs_shift
    #
    # return pixel_centre
    return cwl


from scipy.interpolate import interp1d


def calibrate_spectra(
    spectra, wavelengths, references, window=None, instrument_function=None
):
    """
    Calibrates the spectra from the passed references using a 1D interpolator. Interpolates the spectra
    to match the reference values.

    :param spectra:
    :param wavelengths:
    :param references:
    :return:
    """

    # get pixel locations for reference peaks
    reference_pixels = []
    for ref in references:
        ref_pixel = get_pixel_centre(
            spectra, wavelengths, ref, window, instrument_function
        )
        reference_pixels.append(ref_pixel)

    # print(f'reference_pixels {reference_pixels}')
    wave1 = interp1d(reference_pixels, references, fill_value="extrapolate")(
        wavelengths
    )
    wave2 = np.linspace(wave1.min(), wave1.max(), len(wave1))
    spectra2 = interp1d(wave1, spectra, fill_value="extrapolate")(wave2)

    return (spectra2, wave2)


if __name__ == "__main__":
    filename = "/home/dgahle/baysar_psi2020/experiment/cal_test_data.npz"
    data = np.load(filename)

    spectra = data.get("spectra")
    wave = data.get("wave")

    references = [3955.85, 3995.00, 4041.32, 4097.33, 4121.93]
    references = [
        3955.85,
        3968.99,
        3973.26,
        3995.00,
        4041.32,
        4097.33,
        4100.61,
        4103.43,
        4121.93,
    ]
    references = [
        3955.85,
        3968.99,
        3995.00,
        4035.09,
        4041.32,
        4097.33,
        4100.61,
        4103.43,
        4121.93,
        4133.7,
    ]
    # references=[3955.85, 3968.99, 3995.00, 4035.09, 4041.32, 4097.33, 4100.61, 4103.43, 4121.93, 4133.7]
    window = np.array([-1.0, 1.0])
    spectra1, wave1 = calibrate_spectra(
        spectra, wave, references=references, window=window
    )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(wave, spectra)
    plt.plot(wave1, spectra1)  # , '--')
    # for cwl in out:
    #     plt.plot([cwl, cwl], [spectra.min(), spectra.max()], 'k--')
    for cwl in references:
        plt.plot([cwl, cwl], [spectra.min(), spectra.max()], "r--")
    plt.ylim(1e11, 1e14)
    plt.yscale("log")
    plt.show()

    plt.figure()
    plt.plot(np.diff(wave))
    plt.plot(np.diff(wave1))
    plt.show()

    pass
