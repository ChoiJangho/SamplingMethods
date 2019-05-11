import matplotlib.pyplot as plt
from univariate import TwoGaussiansMixture, ProposalGaussian, UnivariateSampling


target_distribution = TwoGaussiansMixture(
    mu1=-1, sigma1=0.5, mu2=2, sigma2=1.0, alpha=0.2)
proposal_distribution = ProposalGaussian(sigma=0.75)

sampling_handler = UnivariateSampling(
    target_distribution=target_distribution,
    proposal_distribution=proposal_distribution)

sampling_handler.demonstrate_metropolis_hastings(z_initial=0., step_to_end=100)