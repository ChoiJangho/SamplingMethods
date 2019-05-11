from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class UnivariateDistribution(ABC):
    @abstractmethod
    def get_density(self, zs):
        pass

    @abstractmethod
    def sample(self, n_sample):
        pass


class TwoGaussiansMixture(UnivariateDistribution):
    def __init__(self, mu1, sigma1, mu2, sigma2, alpha):
        self._mu1 = mu1
        self._sigma1 = sigma1
        self._mu2 = mu2
        self._sigma2 = sigma2
        self._a = alpha

    def sample(self, n_sample):
        sample_modes = np.random.uniform(size=n_sample)
        samples = np.zeros(n_sample)
        samples[sample_modes <= self._a] = np.random.normal(
            self._mu1, self._sigma1, n_sample)[sample_modes <= self._a]
        samples[sample_modes > self._a] = np.random.normal(
            self._mu2, self._sigma2, n_sample)[sample_modes > self._a]

        return samples

    def get_density(self, zs):
        return self._a * self._get_single_gaussian_density(
            zs, self._mu1, self._sigma1) \
            + (1. - self._a) * self._get_single_gaussian_density(
            zs, self._mu2, self._sigma2)

    @staticmethod
    def _get_single_gaussian_density(zs, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) \
            * np.exp(- (zs - mu) ** 2 / (2 * sigma ** 2))


class SamplingForUnivariateDistribution:
    def __init__(self, target_distribution:UnivariateDistribution):
        self._target = target_distribution

    def sample_direct(self, n_sample):
        return self._target.sample(n_sample)

    def get_density(self, zs):
        return self._target.get_density(zs)


target_distribution = TwoGaussiansMixture(
    mu1=-1, sigma1=0.5, mu2=2, sigma2=1.0, alpha=0.2)
sampling_handler = SamplingForUnivariateDistribution(
    target_distribution=target_distribution)

n_sample = 1000
# Direct Sampling
samples_direct = sampling_handler.sample_direct(n_sample)


# Plots
# direct sample histogram
_, bins, _ = plt.hist(samples_direct, 30, density=True)
density_bins = sampling_handler.get_density(bins)
plt.plot(bins, density_bins, linewidth=2, color='r')
plt.show()