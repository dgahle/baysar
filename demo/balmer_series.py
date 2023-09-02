# Imports
from numpy import linspace, ndarray
from numpy.random import normal

from baysar.input_functions import make_input_dict
from baysar.lineshapes import gaussian
from baysar.posterior import BaysarPosterior


# Variables


# Functions
def main() -> None:
    num_chords: int = 0
    num_pixels: int = 1024
    wavelength_axis: ndarray = linspace(3700, 4500, num_pixels)
    experimental_emission: ndarray = normal(0, 1e12, num_pixels)
    instrument_function: ndarray = gaussian(wavelength_axis, wavelength_axis.mean(), 6., 1)
    emission_constant: float = 1e11
    noise_region: list[float, float] = [4150., 4250.]
    species: list[str] = ['H']
    ions: list[list[str]] = [['0']]
    mystery_lines = None

    input_dict: dict = make_input_dict(
        num_chords=num_chords,
        wavelength_axis=[wavelength_axis],
        experimental_emission=[experimental_emission],
        instrument_function=[instrument_function],
        emission_constant=[emission_constant],
        noise_region=[noise_region],
        species=species,
        ions=ions,
        mystery_lines=mystery_lines,
        refine=[0.01],
        ion_resolved_temperatures=False,
        ion_resolved_tau=False,
    )

    posterior: BaysarPosterior = BaysarPosterior(input_dict=input_dict)

    # # Plot random start
    # from tulasa.plotting_functions import plot_fit
    # sample = [posterior.last_proposal]
    # plot_fit(posterior, sample, size=100, alpha=0.1, ylim=(1e11, 1e+16), error_norm=False, plasma_ref=None,
    #          filename=None, nitrogen=None, deuterium=None)

    # out, big_out = posterior.optimise(
    #     pop_size=12, num_eras=1, generations=3, threads=12, initial_population=None, random_sample_size=1000,
    #     random_order=3, perturbation=None, filename=None, plot=False, plasma_reference=None, return_out=True,
    #     maxiter=100
    # )

    # Todo: NaNs in the bounds come from the SpectrometerChord estimation
    # Todo: NaNs being created in the forward model (also need to check the types going through)
    # Todo: Why are the lineshape models so much larger than the thesis tagged version?

    from scipy.optimize import fmin_l_bfgs_b
    start: ndarray = posterior.random_start()

    def callback(theta: list[float]) -> None:
        print(
            f"logp = {posterior(theta):.1f} - theta = {theta}"
        )

    callback(start)

    x, f, d = fmin_l_bfgs_b(
        posterior.cost,
        start,
        fprime=None,
        approx_grad=True,
        bounds=posterior.plasma.theta_bounds.tolist(),
        m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1,
        maxfun=15000, maxiter=10, disp=None, callback=callback, maxls=20
    )



    pass


if __name__ == "__main__":
    main()
