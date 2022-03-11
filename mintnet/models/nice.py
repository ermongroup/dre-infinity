# NICE model from https://github.com/karpathy/pytorch-normalizing-flows

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


def sigmoid_transform(samples, lambd=1e-6): #lambd=0.05 for cifar10
    if samples.shape[1] == 3:
        lambd = 0.05
    samples = torch.sigmoid(samples)
    samples = (samples - lambd) / (1 - 2 * lambd)
    return samples


class MLP(nn.Module):
    """ a simple 6-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(
            torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(
            torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=MLP, nh=200, scale=True,
                 shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)

    def forward(self, x):
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        # z0, z1 = z[:, ::2], z[:, 1::2] #original
        z0, z1 = torch.split(z, z.shape[1] // 2, 1) # edited
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        # x = torch.cat([x0, x1], dim=1) # original
        x = torch.cat([torch.zeros_like(x0), torch.zeros_like(x1)], dim=1) # edited
        x[:, ::2], x[:, 1::2] = x0, x1 # edited added
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device)
        # zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            # zs.append(x)
        # return zs, log_det
        return x, log_det

    def backward(self, z):
        z = z.view(z.size(0), -1)
        m, _ = z.shape
        log_det = torch.zeros(m, device=z.device)
        # xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            # xs.append(z)
        # return xs, log_det
        return z, log_det


class NICE(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.image_size = config.data.image_size
        self.n_channels = config.data.channels
        img_dim = (self.image_size * self.image_size * self.n_channels)
        self.n_layers = config.model.n_layers
        self.h_dim = config.model.h_dim
        self.prior = Normal(0, 1)
        flows = [AffineHalfFlow(dim=img_dim, parity=i%2,
                 nh=self.h_dim, scale=False) for i in range(self.n_layers)]
        flows.append(AffineConstantFlow(dim=img_dim, shift=False))
        self.flow = NormalizingFlow(flows)

    # def forward(self, x):
    #     zs, log_det = self.flow.forward(x)
    #     # prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
    #     # return zs, prior_logprob, log_det
    #     return zs, log_det

    def forward(self, x, reverse=False):
        """
        changed original forward function to make it compatible with RealNVP
        :param x:
        :param reverse:
        :return:
        """
        if not reverse:
            # go in forward direction
            xs, log_det = self.flow.forward(x)
        else:
            # go in reverse direction
            xs, log_det = self.flow.backward(x)
        return xs, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sampling(self, z, rescale=True):
        xs, _ = self.flow.backward(z)
        xs = xs.view(z.size(0), self.n_channels, self.image_size, self.image_size)
        # NOTE: sigmoid transform happening here!
        xs = sigmoid_transform(xs)
        if rescale:
            xs = (xs * 2.) - 1.
        return xs