########################################################################################################################
### This shell script sets up the python environment and creates folders required to run the scripts that are not stored
### in the repo.
########################################################################################################################

### Setting up the python environment
conda create -n baysar python=3.9 --yes
conda activate baysar
pip install -r requirements.txt

### Setting up the repo directories
mkdir input, output
mkdir output/log

## Create config.JSON for the user (copy from metadata)
cp metadata/config.json config.json

### Test repo setup
PYTHONPATH=. pytest