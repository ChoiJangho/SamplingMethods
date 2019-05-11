""" Main Reference:
Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.
"""
from abc import ABC, abstractmethod
import numpy as np


def get_single_gaussian_density(zs, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) \
           * np.exp(- (zs - mu) ** 2 / (2 * sigma ** 2))


class UnivariateDistribution(ABC):
    @abstractmethod
    def get_density(self, zs):
        pass

    @abstractmethod
    def get_unnormalized_density(self, zs):
        pass

    @abstractmethod
    def sample(self, n_sample):
        pass


class ProposalDistribution(ABC):
    @abstractmethod
    def get_density(self, z_current, z_next):
        pass

    @abstractmethod
    def get_unnormalized_density(self, z_current, z_next):
        pass

    @abstractmethod
    def sample_next(self, z_current):
        pass


class ProposalGaussian(ProposalDistribution):
    def __init__(self, sigma):
        self._sigma = sigma
        self._unnormalize_factor = np.random.uniform()

    def get_density(self, z_current, z_next):
        return get_single_gaussian_density(z_next, z_current, self._sigma)

    def get_unnormalized_density(self, z_current, z_next):
        return self._unnormalize_factor * self.get_density(z_current, z_next)

    def sample_next(self, z_current):
        return np.random.normal(z_current, self._sigma)

class Gaussian(UnivariateDistribution):
    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        self._unnormalize_factor = np.random.uniform()

    def sample(self, n_sample):
        samples = np.random.normal(self._mu, self._sigma, n_sample)
        return samples

    def get_density(self, zs):
        return get_single_gaussian_density(zs, self._mu, self._sigma)

    def get_unnormalized_density(self, zs):
        return self._unnormalize_factor * self.get_density(zs)


class TwoGaussiansMixture(UnivariateDistribution):
    def __init__(self, mu1, sigma1, mu2, sigma2, alpha):
        self._mu1 = mu1
        self._sigma1 = sigma1
        self._mu2 = mu2
        self._sigma2 = sigma2
        self._a = alpha

        self._unnormalize_factor = np.random.uniform()

    def sample(self, n_sample):
        sample_modes = np.random.uniform(size=n_sample)
        samples = np.zeros(n_sample)
        samples[sample_modes <= self._a] = np.random.normal(
            self._mu1, self._sigma1, n_sample)[sample_modes <= self._a]
        samples[sample_modes > self._a] = np.random.normal(
            self._mu2, self._sigma2, n_sample)[sample_modes > self._a]

        return samples

    def get_density(self, zs):
        return self._a * get_single_gaussian_density(
            zs, self._mu1, self._sigma1) \
            + (1. - self._a) * get_single_gaussian_density(
            zs, self._mu2, self._sigma2)

    def get_unnormalized_density(self, zs):
        return self._unnormalize_factor * self.get_density(zs)


class UnivariateSampling:
    LARGE_M = 100
    def __init__(self, target_distribution:UnivariateDistribution,
        proposal_distribution:ProposalDistribution=None):
        self._target = target_distribution
        self._proposal = proposal_distribution

    def sample_direct(self, n_sample):
        return self._target.sample(n_sample)

    @property
    def target(self):
        return self._target

    @property
    def proposal(self):
        return self._proposal

    def sample_by_metropolis_hastings(self, n_sample,
            z_initial, sequence_step=LARGE_M):
        """
        Run MH (for n times) to retain every Mth sample.
        :param n_sample:
        :return: np array of size n_sample, sampled from MH algorithm.
        """
        zs = np.zeros(n_sample)
        for i in range(n_sample):
            sequence = self._run_metropolis_hastings(z_initial, sequence_step)
            zs[i] = sequence[-1]
        return zs

    def _run_metropolis_hastings(self, z_initial, step_to_end):
        def acceptance_thershold(z_t, z_candidate):
            ratio = self._target.get_unnormalized_density(z_candidate) \
                * self._proposal.get_unnormalized_density(z_candidate, z_t) \
                / (self._target.get_unnormalized_density(z_t) *
                   self._proposal.get_unnormalized_density(z_t, z_candidate))
            return min(1.,  ratio)

        sequence = np.zeros(step_to_end+1)
        sequence[0] = z_initial
        for t in range(step_to_end):
            z = sequence[t]
            z_candidate = self._proposal.sample_next(z)
            u = np.random.uniform()
            sequence[t + 1] = z_candidate \
                    if u < acceptance_thershold(z, z_candidate) else z
        return sequence

