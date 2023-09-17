"""
potential_funcs.py
Potential functions taken (but modified) from paper "Variational Inference with Normalizing Flows"
"""

import numpy as np
import torch


def w1(z):
    return torch.sin(2 * torch.pi * z[:, 0] / 4)


def w2(z):
    return 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.5) ** 2)


def sigma(x):
    return 1 / (1 + torch.exp(-x))


def w3(z):
    return 3 * sigma((z[:, 0] - 1) / 0.3)


def u_with_argument(z, angle, distance):
    rotational_matrix = torch.Tensor([
        [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
        [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]
    ])
    z = (rotational_matrix @ z.T).T
    term1 = 0.5 * ((torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2) - distance) / 0.4) ** 2
    term2 = safe_log(
        torch.exp(-0.5 * ((z[:, 0] - distance) / 0.6) ** 2) +
        torch.exp(-0.5 * ((z[:, 0] + distance) / 0.6) ** 2)
    )
    return term1 - term2


# =======================================================
# Helper functions
# =======================================================


def taper(z):
    return torch.sigmoid(
        (5 - torch.linalg.vector_norm(z, dim=1, ord=np.inf)) * 10
    )


def add_taper(potential):
    def new_potential(z):
        # so density is exp(-potential(z) + log(taper(z))) = exp(-potential(z)) * taper(z)
        return potential(z) - torch.log(taper(z))

    return new_potential


def safe_log(x):
    return torch.log(x + 1e-5)


# =======================================================
# Potential functions
# =======================================================


def U1(z):
    part_1 = (1 / 2) * ((torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2) - 2) / 0.4) ** 2
    part_2 = safe_log(
        torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) +
        torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
    )
    return part_1 - part_2


@add_taper
def U2(z):
    return (1 / 2) * ((z[:, 1] - w1(z)) / 0.4) ** 2


@add_taper
def U3(z):
    return - safe_log(
        torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2) +
        torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    )


@add_taper
def U4(z):
    return - safe_log(
        torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.34) ** 2) +
        torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    )


def U8(z):
    # this is a potential function I created (with 8 modes, hence the name u8)
    return u_with_argument(z, torch.pi / 4, 3) / 4 + \
           u_with_argument(z, torch.pi / 2 + torch.pi / 4, 3) / 4 + \
           u_with_argument(z, 0, 3) / 4 + \
           u_with_argument(z, torch.pi / 2, 3) / 4


# =======================================================
# Utility function
# =======================================================


def pick_potential_func(index):
    """
    :param index: which potential function to pick
    :return: (potential func, recommended number of components to use in GMM modeling)
    """
    if index == 1:
        U = U1
        K = 10
    elif index == 2:
        U = U2
        K = 10
    elif index == 3:
        U = U3
        K = 20
    elif index == 4:
        U = U4
        K = 30
    elif index == 8:
        U = U8
        K = 30
    else:
        raise ValueError(f"U{index} is not a valid potential function, yet")
    return U, K
