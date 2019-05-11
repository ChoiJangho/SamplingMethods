""" Main Reference:
Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    class MetropolisHastingsAnimation:
        def __init__(self, fig, z_initial, step_to_end,
                target, proposal, record):
            self._target = target
            self._proposal = proposal

            self.fig = fig
            self._gs = fig.add_gridspec(1, 3)
            self._ax_main = fig.add_subplot(self._gs[0, 0:2])
            self._ax_status = fig.add_subplot(self._gs[0, 2])
            self._ax_status.set_axis_off()
            self._samples = np.arange(-5, 5, 0.1)
            densities = self._target.get_density(self._samples)
            self._plot_target, = self._ax_main.plot(self._samples, densities,
                linewidth=2, color='k', label='Target Distribution')
            self._plot_proposal, = self._ax_main.plot([], [],
                linewidth=2, color='r', label='Proposal Distribution')
            self._plot_z_indicator_point, = self._ax_main.plot([], [], 'o',
                color='k', label='sequence')
            plt.setp(self._plot_z_indicator_point, markersize=4)
            self._ax_main.legend()
            self._initialized = False
            self._new_candidate_appeared = False
            self._sequence = np.zeros(step_to_end + 1)
            self._z_initial = z_initial
            self._step_to_end = step_to_end
            self._step = 0

            self.ani = animation.FuncAnimation(
                self.fig, self.update, interval=500, repeat=False)
            if record:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=15, metadata=dict(artist='Me'),
                    bitrate=1800)
                self.ani.save('mh-demonstration.mp4', writer=writer)
            plt.show()

        def update(self, frame_num):
            def acceptance_thershold(z_t, z_candidate):
                ratio = self._target.get_unnormalized_density(z_candidate) \
                        * self._proposal.get_unnormalized_density(z_candidate,
                    z_t) \
                        / (self._target.get_unnormalized_density(z_t) *
                           self._proposal.get_unnormalized_density(z_t,
                               z_candidate))
                return min(1., ratio)
            if not self._initialized:
                self._sequence[0] = self._z_initial
                self._plot_proposal.set_data(
                    self._samples.tolist(),
                    self._proposal.get_density(
                        self._samples, self._sequence[0]).tolist()
                )

                self._z_indicator_line = self._ax_main.vlines(
                    self._sequence[0],
                    0.,
                    self._proposal.get_density(
                        self._sequence[0], self._sequence[0]),
                    linestyles='dotted')
                self._plot_z_indicator_point.set_data(
                    self._sequence[0], 0
                )
                self._initialized = True
                return self._plot_z_indicator_point, \
                    self._plot_proposal, self._z_indicator_line,

            if self._step == self._step_to_end:
                self.ani.event_source.stop()
            z = self._sequence[self._step]
            self._plot_proposal.set_data(
                self._samples.tolist(),
                self._proposal.get_density(self._samples, z).tolist())

            y_max = self._proposal.get_density(z, z)
            vline_data = np.zeros((1, 2, 2))
            vline_data[0, 0, 1] = 0
            vline_data[0, 1, 1] = y_max
            vline_data[0, 0, 0] = z
            vline_data[0, 1, 0] = z
            self._z_indicator_line.set_paths(vline_data)
            z_candidate = self._proposal.sample_next(z)
            u = np.random.uniform()
            self._sequence[self._step + 1] = z_candidate \
                if u < acceptance_thershold(z, z_candidate) else z
            self._plot_z_indicator_point.set_data(
                self._sequence[0:self._step + 1].tolist(),
                np.zeros(self._step + 1).tolist()
            )

            self._step += 1
            return self._plot_z_indicator_point, \
                self._plot_proposal, self._z_indicator_line,

    def demonstrate_metropolis_hastings(self,
            z_initial, step_to_end, record=False):

        fig = plt.figure("Metropolis-Hastings", figsize=(16, 9))

        animation_mh = self.MetropolisHastingsAnimation(
            fig, z_initial, step_to_end, self._target, self._proposal, record)


