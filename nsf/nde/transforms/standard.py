"""Implementations of some standard transforms."""

import torch
from torch import nn
from nsf.nde import transforms
import nsf.nsf_utils as nsf_utils


class IdentityTransform(transforms.Transform):
    """Transform that leaves input unchanged."""

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = torch.zeros(batch_size)
        return inputs, logabsdet

    def inverse(self, inputs, context=None):
        return self(inputs, context)


class AffineScalarTransform(transforms.Transform):
    """Computes X = X * scale + shift, where scale and shift are scalars, and scale is non-zero."""

    def __init__(self, shift=None, scale=None):
        super().__init__()

        if shift is None and scale is None:
            raise ValueError('At least one of scale and shift must be provided.')
        if scale == 0.:
            raise ValueError('Scale cannot be zero.')

        self.register_buffer('_shift', torch.tensor(shift if (shift is not None) else 0.))
        self.register_buffer('_scale', torch.tensor(scale if (scale is not None) else 1.))

    @property
    def _log_scale(self):
        return torch.log(torch.abs(self._scale))

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        num_dims = torch.prod(torch.tensor(inputs.shape[1:]), dtype=torch.float)
        # outputs = inputs * self._scale + self._shift
        outputs = inputs * self._scale.to(inputs.device) + self._shift.to(inputs.device)
        logabsdet = torch.full([batch_size], self._log_scale * num_dims)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]
        num_dims = torch.prod(torch.tensor(inputs.shape[1:]), dtype=torch.float)
        # outputs = (inputs - self._shift) / self._scale
        outputs = (inputs - self._shift.to(inputs.device)) / self._scale.to(inputs.device)
        logabsdet = torch.full([batch_size], -self._log_scale * num_dims)
        return outputs, logabsdet


class AffineTransform(transforms.Transform):
    """Computes X = X * scale + shift, where scale and shift are vectors, and scale is non-zero.
    NOTE: the left/right multiply for this is a little sketch, but since we're
    explicitly working with diagonal matrices here, it doesn't matter.
    """

    def __init__(self, shape, shift=None, scale=None):
        super().__init__()

        # self._shift = nn.Parameter(torch.rand(*shape))
        # self._scale = nn.Parameter(torch.rand(*shape))
        self._shift = nn.Parameter(torch.zeros(*shape))
        self._scale = nn.Parameter(torch.ones(*shape))

    @property
    def _log_scale(self):
        return torch.log(torch.abs(self._scale))

    def forward(self, inputs, context=None):
        scale = torch.eye(len(self._scale)).to(inputs.device) * self._scale
        outputs = inputs @ scale + self._shift
        logabsdet = torch.logdet(torch.abs(scale))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        scale = torch.eye(len(self._scale)).to(inputs.device) * self._scale
        outputs = (inputs - self._shift) @ torch.inverse(scale)
        logabsdet = -torch.logdet(torch.abs(scale))
        # TODO: oof
        return outputs, logabsdet


class AffineTransformCopula(transforms.Transform):
    """Computes X = X * scale + shift, where scale and shift are vectors, and scale is non-zero."""

    def __init__(self, shape, shift=None, scale=None):
        super().__init__()

        self._shift = nn.Parameter(torch.randn(*shape))
        self._scale = nn.Parameter(torch.randn(*shape, *shape))

    @property
    def _log_scale(self):
        return torch.log(torch.abs(self._scale))

    def forward(self, inputs, context=None):
        # get cholesky?
        # cov = self._scale @ self._scale.T
        # ell = torch.linalg.cholesky(cov)
        # ell = enforce_lower_diag_and_nonneg_diag(self._scale)
        ell = self._scale
        outputs = (ell @ inputs.T).T + self._shift
        # logabsdet = torch.logdet(torch.abs(ell))
        logabsdet = nsf_utils.logabsdet(ell)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        # get cholesky?
        # cov = self._scale @ self._scale.T
        # ell = torch.linalg.cholesky(cov)
        # ell = enforce_lower_diag_and_nonneg_diag(self._scale)
        ell = self._scale
        inv_scale = torch.inverse(ell)

        outputs = (inv_scale @ (inputs - self._shift).T).T
        # logabsdet = torch.logdet(torch.abs(inv_scale))
        logabsdet = nsf_utils.logabsdet(inv_scale)
        return outputs, logabsdet

    # def forward(self, inputs, context=None):
    #     # get cholesky?
    #     cov = self._scale @ self._scale.T
    #     ell = torch.linalg.cholesky(cov)
    #     inv_scale = torch.inverse(ell)
    #     outputs = (inv_scale @ (inputs - self._shift).T).T
    #     logabsdet = torch.logdet(torch.abs(inv_scale))
    #     return outputs, logabsdet
    #
    # def inverse(self, inputs, context=None):
    #     # get cholesky?
    #     cov = self._scale @ self._scale.T
    #     ell = torch.linalg.cholesky(cov)
    #
    #     outputs = (ell @ inputs.T).T + self._shift
    #     # outputs = (self._scale @ inputs.T).T + self._shift
    #     # logabsdet = torch.logdet(torch.abs(self._scale))
    #     logabsdet = torch.logdet(torch.abs(ell))
    #     return outputs, logabsdet


def enforce_lower_diag_and_nonneg_diag(A, shift=0.0):
    mask = torch.ones_like(A).to(A.device)
    ldiag_mask = torch.tril(mask)
    diag_mask = torch.eye(A.size(0)).to(A.device)
    strict_ldiag_mask = ldiag_mask - diag_mask

    # should I use exp or softplus here?
    return strict_ldiag_mask * A + diag_mask * torch.exp(A - shift)


class AffineTransformv2(transforms.Transform):
    """Computes X = X * scale + shift, where scale and shift are vectors, and scale is non-zero."""

    def __init__(self, shape, shift=None, scale=None):
        super().__init__()

        self._scale = nn.Parameter(scale)
        self._inv_scale = nn.Parameter(torch.inverse(scale))
        self._shift = 0.

    @property
    def _log_scale(self):
        return torch.log(torch.abs(self._scale))

    def forward(self, inputs, context=None):
        outputs = (self._inv_scale @ inputs.T).T + self._shift
        logabsdet = torch.logdet(torch.abs(self._inv_scale))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = (self._scale @ (inputs - self._shift).T).T
        logabsdet = torch.logdet(torch.abs(self._scale))
        return outputs, logabsdet


class AffineTransformv3(transforms.Transform):
    """Computes X = X * scale + shift, where scale and shift are vectors, and scale is non-zero."""

    def __init__(self, shape, shift=None, scale=None):
        super().__init__()

        # seems like it has to be the commented out configuration for copula?
        self._scale = torch.eye(len(scale)).to(scale.device) * scale
        # self._scale = torch.eye(len(scale)) * scale
        self._inv_scale = torch.inverse(self._scale)
        self._shift = 0.

    @property
    def _log_scale(self):
        return torch.log(torch.abs(self._scale))

    def forward(self, inputs, context=None):
        outputs = (self._scale.to(inputs.device) @ inputs.T).T
        logabsdet = torch.logdet(torch.abs(self._scale))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = (self._inv_scale.to(inputs.device) @ inputs.T).T
        logabsdet = torch.logdet(torch.abs(self._inv_scale))
        return outputs, logabsdet