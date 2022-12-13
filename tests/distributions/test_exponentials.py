from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import pytest as pt

from stochproc.distributions import NegativeExponential, DoubleExponential


class TestDoubleExponentials(object):
    def test_negative_distribution(self):
        
        with pt.raises(ValueError):
            lambda_plus = +100
            ne = NegativeExponential(rate=lambda_plus, validate_args=True)

        lambda_minus = -lambda_plus
        atol = 1 / 10000
        negative_exp = NegativeExponential(rate=lambda_minus, validate_args=True)
        
        # Test if the mean is correctly computed up to a certain atol
        np.testing.assert_allclose(actual=negative_exp.mean, desired=1 / lambda_minus, atol=atol,
                                   err_msg='Expected mean: {}; Mean returned:{}'.format(1 / lambda_minus,
                                                                                        negative_exp.mean))
        # Test if the variance is correctly computed up to a certain atol
        np.testing.assert_allclose(actual=negative_exp.variance, desired=1 / lambda_minus ** 2, atol=atol,
                                   err_msg='Expected variance: {}; Variance returned:{}'.format(1 / lambda_minus ** 2,
                                                                                                negative_exp.variance))
        # Test if the CDF is correctly computed for some values z
        cdf = lambda z: 1 - np.exp(-lambda_minus * z)
        z = -np.array([5., 1., .5, 1 / 10, 1 / 100, 1 / 10000])
        exp_cdf = cdf(z)
        actual_cdf = negative_exp.cdf(torch.Tensor(z))
        np.testing.assert_allclose(actual=actual_cdf, desired=exp_cdf, atol=atol,
                                   err_msg='Negative Exponential CDF computed for {}; it returns: {}, expected: {}'.format(
                                       z, actual_cdf, exp_cdf))

    def test_double_exponential_distribution(self):
        p = .4
        rho_plus = 5.
        rho_minus = -6.
        
        with pt.raises(ValueError):
            de = DoubleExponential(p=p, rho_minus=rho_plus, rho_plus=rho_plus, validate_args=True)
                
        with pt.raises(ValueError):
            de = DoubleExponential(p=p, rho_minus=rho_minus, rho_plus=rho_minus, validate_args=True)

        with pt.raises(ValueError):
            p1 = 1.5
            de = DoubleExponential(p=p1, rho_minus=rho_minus, rho_plus=rho_minus, validate_args=True)

        de = DoubleExponential(p=p, rho_minus=rho_minus, rho_plus=rho_plus)
        atol = 1 / 10000
        # Test if the mean is correctly computed. See Hainaut and Moraux 2016, Veronese et al. 2022
        exp_mean = p * 1 / rho_plus + (1 - p) * 1 / rho_minus  # expected mean
        actual_mean = de.mean
        np.testing.assert_allclose(actual=actual_mean, desired=exp_mean, atol=atol,
                                   err_msg='Double Exponential expected mean: {}; Mean returned by computation:{}'.format(
                                       exp_mean,
                                       actual_mean))
        # Test if the variance is correctly computed. See Veronese et al. 2022
        exp_variance = p * 2 / rho_plus ** 2 + (1 - p) * 2 / rho_minus ** 2 - exp_mean ** 2  # expected variance
        actual_variance = de.variance
        np.testing.assert_allclose(actual=actual_variance, desired=exp_variance,
                                   err_msg='Double Exponential expected variance: {}; variance returned by computation:{}'.format(
                                       exp_variance,
                                       actual_variance))
        # Test if the Double Exponential is correctly computed
        cdf = lambda z: (1 - p) * (1 - np.exp(- rho_minus * z)) if z < 0 else (1 - p) + p * (1 - np.exp(- rho_plus * z))
        z = np.array([5., 1., .5, 1 / 10, 1 / 100, 1 / 10000, -5., -1., -.5, -1 / 10, -1 / 100, -1 / 10000])
        exp_cdf = np.array([cdf(z_) for z_ in z])
        actual_cdf = de.cdf(torch.Tensor(z))
        np.testing.assert_allclose(actual=actual_cdf, desired=exp_cdf, atol=atol,
                                   err_msg='Double Exponential CDF computed for {}; it returns: {}, expected: {}'.format(
                                       z, actual_cdf, exp_cdf))
