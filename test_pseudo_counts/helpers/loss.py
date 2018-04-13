import torch

def loss_function(log_probs, x, mu, logvar, kl_weight=1, n_samples=10):
    KLD = 0.5 * torch.mean(
        torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1))

    if len(log_probs.shape) == 4:
        x = x.view(-1, 1, x.shape[1]).clone()
        x = x.repeat(1, n_samples, 1)
        LL = torch.sum((1 - x) * log_probs[:, :, :, 0] + x * log_probs[:, :, :, 1]) / (x.shape[0] * n_samples)
    else:
        LL = torch.sum((1 - x) * log_probs[:, :, 0] + x * log_probs[:, :, 1]) / (x.shape[0])

    elbo = LL - kl_weight * KLD
    loss = -1 * elbo
    return loss, LL, KLD
