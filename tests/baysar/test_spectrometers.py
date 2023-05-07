import unittest

import numpy as np
from scipy.io import loadmat

from baysar.lineshapes import gaussian_norm
from baysar.spectrometers import center_of_mass, centre_peak

x = np.arange(31)
peak = gaussian_norm(x=x, cwl=np.mean(x) + 1, fwhm=3, intensity=1)

file_dss = "/home/dgahle/baysar_work/DSS/InstrFunCharac_404nm_HOR.mat"
dss = loadmat(file_dss)
chord = 15
intfunl = dss["InstrFunL"][chord][::-1]
intfunr = dss["InstrFunR"][chord][1:]
# this needs centring propperly - this is cheating
intfun = np.concatenate((intfunl, intfunr))
intfun /= sum(intfun)


class TestCantrePeak(unittest.TestCase):
    def test_output(self):
        # check that the output is proper
        self.assertAlmostEqual(center_of_mass(centre_peak(peak))[0], np.mean(x))
        self.assertAlmostEqual(
            center_of_mass(centre_peak(intfun))[0], np.mean(np.arange(len(intfun)))
        )
        # self.assertEqual(all(np.isreal(tmpout)), True, 'gaussian output contains non real values')
        # self.assertEqual(len(tmpout.shape), 1, 'gaussian output is not a 1D')
        # self.assertEqual(type(tmpout), np.ndarray, TypeError('gaussian output is not a ndarray'))

    # def test_values(self):
    # Make sure value errors are raised when necessary
    # self.assertRaises(ValueError, gaussian, x=np.array([0]), cwl=1, fwhm=-1, intensity=1)
    # self.assertRaises(ValueError, gaussian, x=np.array([0]), cwl=1, fwhm=1, intensity=-1)

    # def test_types(self):
    # Make sure type errors are raised when necessary
    # self.assertRaises(TypeError, gaussian, x=[], cwl=1, fwhm=1, intensity=1)
    # self.assertRaises(TypeError, gaussian, x=np.array(['a']), cwl=1, fwhm=1, intensity=1)
    # self.assertRaises(TypeError, gaussian, x=np.array([0]), cwl='a', fwhm=1, intensity=1)
    # self.assertRaises(TypeError, gaussian, x=np.array([0]), cwl=1, fwhm='a', intensity=1)
    # self.assertRaises(TypeError, gaussian, x=np.array([0]), cwl=1, fwhm=1, intensity='a')


if __name__ == "__main__":
    unittest.main()
