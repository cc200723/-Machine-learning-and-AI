import torch
import torch.nn.functional as F

def gaussian_elbo(x1, x2, z, sigma, mu, logvar):

    #
    # Task 2: Compute the evidence lower bound for the Gaussian VAE.
    #         Use the closed-form expression for the KL divergence.
    #

    # Compute the reconstruction loss as the mean squared error
    reconstruction = F.mse_loss(x1, x2, reduction='sum') / (2 * sigma ** 2)

    # Compute the KL divergence using the closed-form expression
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction, divergence
