# Imports
from copy import copy
from json import load
from pathlib import Path

from inference.mcmc import GibbsChain
from numpy import array, diff, log10, ndarray
from numpy.random import normal
from pandas import DataFrame

from baysar.input_functions import make_input_dict
from baysar.posterior import BaysarPosterior
from tulasa.general import plot

# Variables


# Functions
def optimise_posterior(posterior: BaysarPosterior) -> GibbsChain:
    start: ndarray = posterior.random_start()
    chain: GibbsChain = GibbsChain(posterior=posterior, start=start)
    for n_param, bound in enumerate(posterior.plasma.theta_bounds):
        chain.set_boundaries(n_param, bound)

    chain.run_for(minutes=0.5)

    success_criteria: float = -0.5 * sum(
        [len(_wavelengths) for _wavelengths in posterior.input_dict["wavelength_axis"]]
    )
    sucecssful: bool = success_criteria < max(chain.probs)

    tries: int = 0
    while not sucecssful and tries < 3:
        tries += 1
        chain.run_for(minutes=0.5)
        sucecssful: bool = max(chain.probs) < success_criteria

    if success_criteria:
        print("Zero fit was successful!")
    else:
        print("Zero fit was unsuccessful!")

    return chain


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


def main() -> None:
    # TODO: Where has H_0_dens gone?!
    # TODO: Need a check for negative emission spectra!

    baysar_config_path: Path = Path(__file__).parent / "zero_fit_config.json"
    with open(baysar_config_path, "r") as f:
        baysar_config = load(f)

    input_dict: dict = (
        make_input_dict(**baysar_config)
    )

    posterior: BaysarPosterior = BaysarPosterior(input_dict=input_dict)

    # Zero fit!
    chain: GibbsChain = optimise_posterior(posterior)

    # Self vs Self fit!
    # Reference theta
    reference_theta: list[float] = copy(chain.mode())
    theta_slices: dict[str, slice] = posterior.plasma.slices
    reference_theta[theta_slices["electron_density"]] = [log10(8e13), 0, 3]
    reference_theta[theta_slices["electron_temperature"]] = [log10(5.), 3, 1]
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
    posterior: BaysarPosterior = BaysarPosterior(input_dict=input_dict)
    # Fit!
    chain: GibbsChain = optimise_posterior(posterior)

    chain.run_for(minutes=5)

    try:
        chain.plot_diagnostics()
    except IndexError as index_error:
        print(f"GibbsChain.plot_diagnostics - {index_error}")
        plot(chain.probs)

    chain.matrix_plot(reference=reference_theta)

    param_index: list[str] = get_param_index(posterior)
    data: ndarray = array([reference_theta, chain.mode()]).T
    data = array([reference_theta, chain.mode(), diff(data).flatten()]).T
    df_fit: DataFrame = DataFrame(
        data=data,
        index=param_index,
        columns=[
            "Target",
            "Fit",
            "Diff"
        ]
    )
    print(df_fit)

    _ = posterior(chain.mode())
    fit_spectra: ndarray = posterior.posterior_components[0].forward_model()
    plot(
        array([synthetic_spectra, fit_spectra]).T,
        x=array([wavelength_axis, wavelength_axis]).T
    )


    pass


if __name__ == "__main__":
    main()
