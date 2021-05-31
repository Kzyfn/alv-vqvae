import numpy as np
import torch
import torch.nn.functional as F


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def calc_lf0_rmse(natural, generated, lf0_idx, vuv_idx):
    idx = (natural[:, vuv_idx] * (generated[:, vuv_idx] >= 0.5)).astype(bool)
    return (
        rmse(natural[idx, lf0_idx], generated[idx, lf0_idx]) * 1200 / np.log(2)
    )  # unit: [cent]


def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(
        recon_x.view(-1), x.reshape(-1,), reduction="sum"
    )  # F.binary_cross_entropy(recon_x.view(-1), x.view(-1, ), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(KLD)l
    return MSE + KLD


def vqvae_loss(recon_x, x, z, z_unquantized, beta=1):

    MSE = F.mse_loss(
        recon_x.view(-1), x.reshape(-1,), reduction="sum"
    )  # F.binary_cross_entropy(recon_x.view(-1), x.view(-1, ), reduction='sum')

    vq_loss = F.mse_loss(
        z.view(-1), z_unquantized.detach().view(-1,), reduction="sum"
    ) + beta * F.mse_loss(z.detach().view(-1), z_unquantized.view(-1,), reduction="sum")
    # print(KLD)
    return MSE + vq_loss