from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="baysar",
    version="0.0.0",
    author="Daljeet Singh Gahle",
    author_email="daljeet.gahle@strath.ac.uk",
    description="A python module for producing posterior objects for bayesian inference of plasam parameters from atonmic line spectra.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ccfe.ac.uk/dgahle/baysar/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
