import subprocess
from time import time
from numpy import round

start_time = time()

subprocess.run(['echo hello world'], shell=True)
# build initial distributions
build_inistial_dist_str = 'python build_initial_distributions.py --sample_size 300  --save output/test_initial_distribution'
subprocess.run([build_inistial_dist_str], shell=True)
# need to cache ion balcalculation 1000 takes 320 s (with grid evaluation < 40s)
build_spectra_database_str = 'python build_spectra_database.py --element n --ion_charge 1 --save output/test_spectra_database'
subprocess.run([build_spectra_database_str], shell=True)

subprocess.run(['python format_experimental_spectra.py'], shell=True)

subprocess.run(['python spectral_grid_evaluation.py'], shell=True)

subprocess.run(['python plot_inference.py'], shell=True)


print()
runtime = round(time()-start_time, 2)
print(f'run_full runtime: {runtime}s')
