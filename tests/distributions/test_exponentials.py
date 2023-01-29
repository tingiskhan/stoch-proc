from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import pytest as pt

from stochproc.distributions import NegativeExponential, DoubleExponential


class TestDoubleExponentials(object):
    def test_negative_distribution(self):
        
        lamda = 1.0
        with pt.raises(ValueError):            
            ne = NegativeExponential(rate=-lamda, validate_args=True)

        atol = 1e-4
        negative_exp = NegativeExponential(rate=lamda, validate_args=True)

        z = np.linspace(-10.0, -atol, dtype=np.float32)
        exp_cdf = np.exp(-lamda * (-z))

        actual_cdf = negative_exp.cdf(torch.from_numpy(z))
        np.testing.assert_allclose(actual=actual_cdf, desired=exp_cdf, atol=atol)

    def test_double_exponential_distribution(self):
        p = 0.4
        rho_plus = 5.0
        rho_minus = 6.0

        de = DoubleExponential(p=p, rho_minus=rho_minus, rho_plus=rho_plus)
        atol = 1e-4

        # Test if the mean is correctly computed. See Hainaut and Moraux 2016, Veronese et al. 2022
        exp_mean = p * 1 / rho_plus - (1 - p) * 1 / rho_minus
        actual_mean = de.mean
        np.testing.assert_allclose(actual=actual_mean, desired=exp_mean, atol=atol)

        # Test if the variance is correctly computed. See Veronese et al. 2022
        exp_variance = p * 2 / rho_plus ** 2 + (1 - p) * 2 / rho_minus ** 2 - exp_mean ** 2
        actual_variance = de.variance
        np.testing.assert_allclose(actual=actual_variance, desired=exp_variance)

        # Test if the Double Exponential is correctly computed. TODO: Replace with log prob instead
        z = np.linspace(-5.0, 5.0, dtype=np.float32)        
        exp_cdf = np.where(z < 0.0, (1.0 - p) * np.exp(rho_minus * z), (1.0 - p) + p * (1 - np.exp(-rho_plus * z)))

        actual_cdf = de.cdf(torch.from_numpy(z))
        np.testing.assert_allclose(actual=actual_cdf, desired=exp_cdf, atol=atol)

        shape = torch.Size([100, 20])
        samples = de.sample(shape)

        assert samples.shape == shape