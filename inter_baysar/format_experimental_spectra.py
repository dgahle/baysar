from argparse import ArgumentParser

parser=ArgumentParser()

parser.add_argument('--exp_spectra', type=str, default='data/dgahle_32244_rov014_interelm.sav')
parser.add_argument('--save', type=str, default='data/test_formated_spectra')
parser.add_argument('--plot', action='store_true')

args=parser.parse_args()

import numpy as np

# load data
if args.exp_spectra.endswith('sav'):
    from scipy.io import readsav
    data = readsav(args.exp_spectra)

    # In [3]: data.keys()
    # Out[3]: dict_keys(['emiss', 'wavelength', 'time'])
else:
    raise TypeError(f'Method not implimented to load {args.exp_spectra}!')

# calibrate

# format (pixel, chord, time)
shape = (data['wavelength'].shape[0], 1, data['time'].shape[0])
spectra_formatted = np.zeros(shape)
spectra_formatted[:, 0, :] = 1e-5 * data['emiss'].T.clip(1) - 1e12

error = 1e11 + spectra_formatted * 1e-3

out = {'spectra': spectra_formatted, 'error': spectra_formatted}

if args.plot:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(spectra_formatted.mean(-1).mean(-1))
    plt.show()
else:
    np.savez(args.save, **out)
