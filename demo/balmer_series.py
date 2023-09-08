# Imports
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from json import load
from pathlib import Path

from inference.mcmc import GibbsChain
from numpy import array, diff, log10, ndarray, power
from numpy.random import normal
from pandas import DataFrame
from scipy.optimize import fmin_l_bfgs_b

from baysar.input_functions import make_input_dict
from baysar.posterior import BaysarPosterior
from tulasa.general import plot

# Variables


# Functions
def get_gibbs_chain(posterior, start, bounds=None) -> GibbsChain:
    chain: GibbsChain = GibbsChain(posterior=posterior, start=start)
    if bounds is not None:
        for n_param, bound in enumerate(bounds):
            chain.set_boundaries(n_param, bound)

    return chain


def optimise_posterior(posterior: BaysarPosterior, minutes=1.) -> GibbsChain:
    start: ndarray = posterior.random_start()
    chain: GibbsChain = get_gibbs_chain(posterior, start, bounds=posterior.plasma.theta_bounds)
    chain.run_for(minutes=minutes)

    success_criteria: float = -0.5 * sum(
        [len(_wavelengths) for _wavelengths in posterior.input_dict["wavelength_axis"]]
    )
    sucecssful: bool = success_criteria < max(chain.probs)

    tries: int = 0
    while not sucecssful and tries < 3:
        tries += 1
        chain.run_for(minutes=minutes)
        sucecssful: bool = max(chain.probs) < success_criteria

    if success_criteria:
        print("Fit was successful!")
    else:
        print("Fit was unsuccessful!")

    return chain


@dataclass
class SuperBFGS:

    func: Callable
    bounds: list

    def __call__(self, theta, *args, **kwargs):
        # Unpack theta
        epsilon: float = power(10, theta[0])
        func_theta: list[float] = theta[1:]

        x, f, d = fmin_l_bfgs_b(
            self.func,
            theta[1:],
            approx_grad=True,
            bounds=self.bounds[1:],
            m=10, factr=1e7, pgtol=1e-5,
            epsilon=1e-8, iprint=-1, maxfun=15000,
            maxiter=1, disp=None,
            callback=self.callback,
            maxls=20
        )

        return f

    def callback(self, func_theta):
        print(f"logP = {self.func(func_theta):.1f}", end='\n', flush=True)


def len_slice(index: slice) -> int:
    step: int = 1 if index.step is None else index.step
    return (index.stop - index.start) // step


def get_param_index(posterior: BaysarPosterior) -> list[str]:

    param_slices: dict = posterior.plasma.slices
    index: list[str] = []
    for param in param_slices:
        param_slice: slice = param_slices[param]
        num_param_slice: int = len_slice(param_slice)
        sub_index: list[str] = [
            param if num_param_slice == 1 else f"{param}#{n}" for n in range(num_param_slice)
        ]
        index.extend(sub_index)

    return index


def get_theta_fit_df(reference: ndarray, fit: ndarray, posterior: BaysarPosterior) -> DataFrame:

    param_index: list[str] = get_param_index(posterior)
    data: ndarray = array([reference, fit]).T
    data = array([reference, fit, diff(data).flatten()]).T
    df_fit: DataFrame = DataFrame(
        data=data,
        index=param_index,
        columns=[
            "Target",
            "Fit",
            "Diff"
        ]
    )

    return df_fit


def main() -> None:
    # TODO: Need a check for negative emission spectra!

    #################
    ### Zero fit! ###
    #################

    # Build posterior
    baysar_config_path: Path = Path(__file__).parent / "zero_fit_config.json"
    with open(baysar_config_path, "r") as f:
        baysar_config = load(f)

    input_dict: dict = make_input_dict(**baysar_config)
    posterior: BaysarPosterior = BaysarPosterior(input_dict=input_dict)
    # Pseudo-optimise
    chain: GibbsChain = optimise_posterior(posterior, minutes=0.16)

    #########################
    ### Self vs Self fit! ###
    #########################

    # Reference theta
    reference_theta: list[float] = copy(chain.mode())
    theta_slices: dict[str, slice] = posterior.plasma.slices
    reference_theta[theta_slices["electron_density"]] = [log10(8e13), 0, 3]
    reference_theta[theta_slices["electron_temperature"]] = [log10(5.), 3, 1]
    if not posterior.plasma.no_sample_neutrals:
        reference_theta[theta_slices["H_0_dens"]] = [log10(8e13)]
    reference_theta[theta_slices["H_0_tau"]] = [-8.0]
    reference_theta[theta_slices["H_0_velocity"]] = [0.0]
    # Create Synthetic Spectra
    posterior(reference_theta)
    synthetic_spectra: ndarray = posterior.posterior_components[0].forward_model()
    # Add noise
    num_pixels: int = len(synthetic_spectra)
    wavelength_axis: ndarray = posterior.input_dict['wavelength_axis'][0]
    synthetic_spectra += normal(0, 1e12, num_pixels)
    input_dict["experimental_emission"] = [synthetic_spectra]
    plot(synthetic_spectra, x=wavelength_axis)

    # Rebuild posterior
    print('\n\n')
    posterior: BaysarPosterior = BaysarPosterior(input_dict=input_dict)
    # Pseudo-optimise
    start: ndarray = posterior.random_start()
    chain: GibbsChain = get_gibbs_chain(posterior, start, bounds=posterior.plasma.theta_bounds)
    chain.advance(50)
    # Optimise!
    # hyper_cost_bounds: list = [[-10.0, -5.0], *posterior.plasma.theta_bounds]
    # hyper_cost: SuperBFGS = SuperBFGS(posterior.cost, hyper_cost_bounds)
    start: list[float] = chain.mode()
    x, f, d = fmin_l_bfgs_b(
        posterior.cost,
        start,
        approx_grad=True,
        bounds=posterior.plasma.theta_bounds,
        maxiter=100,
        callback=lambda x: print(f"logP = {posterior(x):.2f}", end='\n', flush=True)
    )

    df_fit: DataFrame = get_theta_fit_df(reference_theta, x, posterior)
    print(df_fit)

    chain = get_gibbs_chain(posterior, x, bounds=posterior.plasma.theta_bounds)
    chain.run_for(minutes=15)

    try:
        chain.plot_diagnostics()
    except IndexError as index_error:
        print(f"GibbsChain.plot_diagnostics - {index_error}")
        plot(chain.probs)

    chain.matrix_plot(reference=reference_theta)

    df_fit: DataFrame = get_theta_fit_df(reference_theta, chain.mode(), posterior)
    print(df_fit)

    _ = posterior(chain.mode())
    fit_spectra: ndarray = posterior.posterior_components[0].forward_model()
    plot(
        array([synthetic_spectra, fit_spectra]).T,
        x=array([wavelength_axis, wavelength_axis]).T
    )

    rmse: ndarray = abs((synthetic_spectra - fit_spectra) / synthetic_spectra)
    plot(rmse, x=wavelength_axis, log=True)

    pass


if __name__ == "__main__":
    main()
