import matplotlib.pyplot as plt
from univariate import TwoGaussiansMixture, UnivariateSampling


target_distribution = TwoGaussiansMixture(
    mu1=-1, sigma1=0.5, mu2=2, sigma2=1.0, alpha=0.2)
sampling_handler = UnivariateSampling(target_distribution=target_distribution)

n_sample = 1000
# Direct Sampling
samples_direct = sampling_handler.sample_direct(n_sample)


# Plots
fig = plt.figure("Direct Sampling")
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], frameon=True)
# direct sample histogram
_, bins, _ = ax.hist(samples_direct, 30, density=True)
density_bins = sampling_handler.target.get_density(bins)
ax.plot(bins, density_bins, linewidth=2, color='r')
plt.show()