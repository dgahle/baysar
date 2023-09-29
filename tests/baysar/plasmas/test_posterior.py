# Imports
from json import load
from pathlib import Path

from numpy import isnan, ndarray

from baysar.input_functions import make_input_dict
from baysar.posterior import BaysarPosterior

# Variables
repo_dir: Path = Path(__file__).parent.parent.parent.parent
baysar_config_path: Path = repo_dir / "demo" / "zero_fit_config.json"
with open(baysar_config_path, "r") as f:
    baysar_config = load(f)

input_dict: dict = make_input_dict(**baysar_config)
posterior: BaysarPosterior = BaysarPosterior(input_dict=input_dict)
theta: list[float] = posterior.random_start()


# Functions and Classes
class TestBaysarPosterior:

    def test_gradient(self) -> None:
        gradient: ndarray = posterior.gradient(theta)
        assert isnan(gradient).all(), 'Posterior gradient contains NaNs!'


if __name__ == "__main__":
    pass
