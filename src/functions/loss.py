#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import functional as F


def VAELoss(X_recon, X, mu, log_var):
    BCE = F.binary_cross_entropy(X_recon, X, reduction="sum")
    D_KL = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
    return BCE + D_KL