#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 16, 7, 7)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Flatten(),
        )

    def forward(self, X):
        return self.model(X)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            UnFlatten(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, Z):
        return self.model(Z)


class ConvVAE(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.encoder = Encoder()
        self.enc_mu = nn.Linear(hidden_size, latent_size)  # mu
        self.enc_logvar = nn.Linear(hidden_size, latent_size)  # log_var
        self.fc = nn.Linear(latent_size, hidden_size)
        self.decoder = Decoder()

    def _reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)  # eps ~ Normal(0, 1)
        Z = mu + eps * sigma
        return Z

    def forward(self, X):
        X = self.encoder(X)
        mu = self.enc_mu(X)
        log_var = self.enc_logvar(X)
        Z = self._reparameterize(mu, log_var)
        Z = self.fc(Z)
        X = self.decoder(Z)
        return X, mu, log_var