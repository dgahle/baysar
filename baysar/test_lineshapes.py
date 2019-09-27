import unittest
import numpy as np
from baysar.lineshapes import gaussian

class TestGaussian(unittest.TestCase):
    # def test_area(self):
    #     # Test areas when radius > 0
    #     self.assertAlmostEqual(circle_area(1), pi)
    #     self.assertAlmostEqual(circle_area(0), 0)
    #     self.assertAlmostEqual(circle_area(2.1), pi*(2.1**2))

    def test_values(self):
        # Make sure value errors are raised when necessary
        self.assertRaises(ValueError, gaussian, x=np.array([0]), cwl=1, fwhm=-1, intensity=1)

    def test_types(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, gaussian, x=[], cwl=1, fwhm=1, intensity=1)
        self.assertRaises(TypeError, gaussian, x=np.array(['a']), cwl=1, fwhm=1, intensity=1)
        self.assertRaises(TypeError, gaussian, x=np.array([0]), cwl='a', fwhm=1, intensity=1)
        self.assertRaises(TypeError, gaussian, x=np.array([0]), cwl=1, fwhm='a', intensity=1)
