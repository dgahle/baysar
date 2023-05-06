Bayesian Spectral Analysis Routine

[![GitHub Action PyTests](https://github.com/dgahle/baysar/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/dgahle/baysar/actions/workflows/pytest.yml)
[![GitHub Action Linting](https://github.com/dgahle/baysar/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/dgahle/baysar/actions/workflows/linting.yml)
[![Python Version](https://img.shields.io/badge/python->=3.7-blue)](https://www.python.org/downloads/release/python-390/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/dgahle/baysar/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![GitHub issues open](https://img.shields.io/github/issues-raw/dgahle/baysar?style=flat)](https://github.com/dgahle/baysar/issues?q=is%3Aissue+is%3Aopen)
[![GitHub issues closed](https://img.shields.io/github/issues-closed-raw/dgahle/baysar?style=flat)](https://github.com/dgahle/baysar/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub PR closed](https://img.shields.io/github/issues-pr-closed/dgahle/baysar)](https://github.com/dgahle/baysar/pulls?q=is%3Apr+is%3Aclosed)

Bayesian Spectral Analysis Routine (BaySAR) is an agnostic spectral fitting routine that takes descriptions of the 
plasma elemental composition, the electron density and temperature profile functions and spectrometer calibration to 
build bespoke spectral fitting posterior functions that when evaluated infers emission weighted plasma quantities as 
well as line of sight plasma profiles.

================= TO UPDATE START =================

This repository provides a template structure a software development project with some python functionality that is 
generally usefully (demonstrated in `scripts/main.py`).
This includes:
- Folder structure
- Python script template (`scripts/main.py`)
- PyTest example (`tests/test_main.py`)
- GitHub Actions example (`.github/workflows/pytest.yml`)

As well as tracking information in `.gitignore` and python dependencies `requirements.txt`.

================= TO UPDATE END =================

__Requesting changes:__ Raise an issue on the GitHub repository page and describes the error or the desired 
functionality, ideally with test data that can be used to make tutorials and unit tests.

## Getting Started (Build, Test, and Run)

This example repo has been written in Python 3.9 and the dependencies are listed in requirements.txt.
The script setup_project.sh will build the conda environment, install dependencies, and create the input/output folders 
and the configuration file.
The actions the of setup_project.sh are:
1. Create virtual python environment:
`conda create -n baysar python=3.9 --yes`
2. Install the dependencies:
`pip install -r requirements.txt`
3. Setting up the repo directories:
`mkdir input, output`
4. Create config.JSON for the user (copy from metadata):
`cp metadata/config.json config.json`
5. Test repo setup:
`PYTHONPATH=. pytest`

Now the repository is locally set up you can run `scripts/main.py` which is a template for scripts and modules.
To run from the console run `PYTHONPATH=. python scripts/main.py`, within the virtual environment 
(`conda activate baysar`).

## How to Contribute

The following steps describe how to add new functionality to the repo:
1. Raise an issue in this report that describes the functionality you want to add, 
2. branch from `dev` with the name convention `feature\<descriptive-branch-name>`
3. Add code to `feature\<descriptive-branch-name>`
4. Create a (draft) pull request with `dev` as the target (link the PR is the corresponding issue)

In the review the scripts will be reviewed against the coding standards below, passing unit tests that verify the 
functionality, and updated documentation and tutorials. 
Assistance can be provided if you are unfamiliar with implementing some of these standards. 

### Coding Standards

As the code should be easy if not trivial for others to reuse blind there are a series of standards that example 
scripts need to meet:
- Descriptive programming style - As in variable, function, and class names should describe what they are/do.
- Type hinting - All declared variables should have the types declared alongside it.
- Well commented - The example should be understandable from the comments with only brief references to the code.
- Doc string - Comment blocks at the beginning of functions and class methods that describes what the function/method 
does and lists the inputs, outputs, and exceptions that could be raised. This is important for the documentation of the 
repository.
- An example of using the function - This will be used for automated testing to ensure functionality is maintained with
changes in the repo and to write the tutorial in the document.

`isort` and `black` are used to aid code format standardisation. 
This can be done from the comand line using:

```
python -m black .
python -m isort .
```

The configuration for `isort` and `black` are in `pyproject.toml`.  

### Testing

Unit tests should be written to verify that the examples meet their objectives.  
GitHub Actions are used to automate the running of the unit tests as a condition for merging into the main and dev 
branch.
