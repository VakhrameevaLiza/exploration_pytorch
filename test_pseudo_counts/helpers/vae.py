import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from helpers.convert_to_var_foo import convert_to_var


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class VAE(nn.Module):
    def __init__(self, input_shape=(-1, 10),
                 latent_dim=2, n_samples=10,
                 fc_size=1024, num_layers=2):
        super(VAE, self).__init__()
        self.n_samples = n_samples
        self.input_shape = input_shape

        encoder_layers = []
        encoder_layers.append(nn.Linear(input_shape[-1], fc_size))
        encoder_layers.append(nn.ReLU())
        for i in range(1, num_layers):
            encoder_layers.append(nn.Linear(fc_size // 2 ** (i - 1), fc_size // 2 ** i))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()

        self.fc_mu = nn.Linear(fc_size // 2 ** (num_layers - 1), latent_dim)
        self.fc_logvar = nn.Linear(fc_size // 2 ** (num_layers - 1), latent_dim)

        if torch.cuda.is_available():
            self.fc_mu = self.fc_mu.cuda()
            self.fc_logvar = self.fc_logvar.cuda()

        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, fc_size // 2 ** (num_layers - 1)))
        decoder_layers.append(nn.ReLU())

        for i in range(num_layers - 1):
            decoder_layers.append(nn.Linear(fc_size // 2 ** (num_layers - 1 - i), fc_size // 2 ** (num_layers - i - 2)))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(fc_size, 2 * input_shape[1]))
        self.decoder = nn.Sequential(*decoder_layers)
        if torch.cuda.is_available():
            self.decoder = self.decoder.cuda()

    def encode(self, x):
        enc_out = self.encoder(x)
        return self.fc_mu(enc_out), self.fc_logvar(enc_out)

    def reparameterize(self, mu, logvar):
        if self.training:
            n_samples = self.n_samples
            batch_size = mu.shape[0]
            latent_dim = mu.shape[1]
            eps = np.random.randn(batch_size * latent_dim * n_samples) \
                .reshape((batch_size, n_samples, latent_dim))
            eps = convert_to_var(eps)
            std = logvar.mul(0.5).exp_()
            mu = mu.view(-1, 1, latent_dim)
            std = std.view(-1, 1, latent_dim)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        dec_out = self.decoder(z)
        if len(dec_out.shape) == 3:
            dec_out = dec_out.view(-1, self.n_samples, self.input_shape[1], 2)
            probs = F.softmax(dec_out, dim=-1)
            log_probs = F.log_softmax(dec_out, dim=-1)
            recon_x = probs[:, :, :, 1]
        else:
            dec_out = self.decoder(z)
            dec_out = dec_out.view(-1, self.input_shape[1], 2)
            probs = F.softmax(dec_out, dim=-1)
            log_probs = F.log_softmax(dec_out, dim=-1)
            recon_x = probs[:, :, 1]
        return recon_x, log_probs

    def forward(self, x):
        mu, logvar = self.encode(x.view(self.input_shape))
        z = self.reparameterize(mu, logvar)
        recon_x, log_probs = self.decode(z)
        return recon_x, log_probs, mu, logvar