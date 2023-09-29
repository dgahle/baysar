import unittest

import numpy as np

from baysar.lineshapes import (
    gaussian,
    gaussian_check_input,
    gaussian_norm,
    reduce_wavelength,
)

# class TestReduceWavelength(unittest.TestCase):
#     # output tests
#     # input tests
#     def test_types(self):
#         # Make sure type errors are raised when necessary
#         self.assertRaises(TypeError, reduce_wavelength, wavelengths, cwl, half_range, return_indicies)


tmpout = gaussian(x=np.array([0]), cwl=1, fwhm=1, intensity=1)


class TestGaussian(unittest.TestCase):
    def test_output(self):
        # check that the output is proper
        self.assertEqual(
            all(np.isreal(tmpout)), True, "gaussian output contains non real values"
        )
        self.assertEqual(len(tmpout.shape), 1, "gaussian output is not a 1D")
        self.assertEqual(
            type(tmpout), np.ndarray, TypeError("gaussian output is not a ndarray")
        )

    def test_values(self):
        # Make sure value errors are raised when necessary
        self.assertRaises(
            ValueError,
            gaussian_check_input,
            x=np.array([0]),
            cwl=1,
            fwhm=-1,
            intensity=1,
        )
        self.assertRaises(
            ValueError,
            gaussian_check_input,
            x=np.array([0]),
            cwl=1,
            fwhm=1,
            intensity=-1,
        )

    def test_types(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(
            TypeError, gaussian_check_input, x=[], cwl=1, fwhm=1, intensity=1
        )
        self.assertRaises(
            TypeError,
            gaussian_check_input,
            x=np.array(["a"]),
            cwl=1,
            fwhm=1,
            intensity=1,
        )
        self.assertRaises(
            TypeError,
            gaussian_check_input,
            x=np.array([0]),
            cwl="a",
            fwhm=1,
            intensity=1,
        )
        self.assertRaises(
            TypeError,
            gaussian_check_input,
            x=np.array([0]),
            cwl=1,
            fwhm="a",
            intensity=1,
        )
        self.assertRaises(
            TypeError,
            gaussian_check_input,
            x=np.array([0]),
            cwl=1,
            fwhm=1,
            intensity="a",
        )


tmpoutx = np.linspace(0, 2, 50)
tmpout = gaussian_norm(x=tmpoutx, cwl=1, fwhm=0.2, intensity=1)


class TestGaussianNorm(unittest.TestCase):
    def test_output(self):
        # check that the output is proper
        self.assertEqual(
            all(np.isreal(tmpout)),
            True,
            "gaussian_norm output contains non real values",
        )
        self.assertEqual(len(tmpout.shape), 1, "gaussian_norm output is not a 1D")
        self.assertEqual(
            type(tmpout), np.ndarray, TypeError("gaussian_norm output is not a ndarray")
        )
        self.assertAlmostEqual(
            np.trapz(tmpout, tmpoutx),
            1,
            ValueError("Normalisation is not properly functioning"),
        )


if __name__ == "__main__":
    unittest.main()
