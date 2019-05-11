import matplotlib.pyplot as plt
from main import TwoGaussiansMixture, ProposalGaussian, UnivariateSampling


target_distribution = TwoGaussiansMixture(
    mu1=-1, sigma1=0.5, mu2=2, sigma2=1.0, alpha=0.2)
proposal_distribution = ProposalGaussian(sigma=0.75)

sampling_handler = UnivariateSampling(
    target_distribution=target_distribution,
    proposal_distribution=proposal_distribution)
n_sample = 500

fig = plt.figure("Metropolis-Hastings")
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], frameon=True)

samples_direct = sampling_handler.sample_direct(n_sample)
_, bins, _ = ax.hist(samples_direct, 30, density=True, label='Direct Sample', color='#aaaaaa')
samples_mh = sampling_handler.sample_by_metropolis_hastings(n_sample, 0.)
_, bins, _ = ax.hist(samples_mh, 30, density=True, label='Metropolis-Hastings', color='#5577ff')
density_bins = sampling_handler.target.get_density(bins)
ax.plot(bins, density_bins, linewidth=2, color='r', label='Target Distribution')
ax.legend()
plt.show()