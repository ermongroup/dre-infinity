import torch
import torch.nn as nn
from . import utils
from sde_lib import VPSDE

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@utils.register_model(name='mlp')
class MLP(nn.Module):
  """
  Simple MLP
  """

  def __init__(self, dim, hidden_dim, output_dim, layers):
    self.dim = dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.n_layers = layers

    seq = [nn.Linear(self.dim, self.hidden_dim), nn.ReLU()]
    for _ in range(self.n_layers):
      seq += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
    seq += [nn.Linear(self.hidden_dim, self.output_dim)]
    self.net = nn.Sequential(*seq)

  def forward(self, x):
    return self.net(x)


@utils.register_model(name='toy_scorenet')
class TimeScoreNetwork(nn.Module):
    """
    Simple MLP-based score network (for toy gaussian problems)
    # """
    def __init__(self, config):
        super().__init__()
        self.in_dim = config.data.dim
        self.h_dim = config.model.z_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim+1, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ELU(),
            nn.Linear(self.h_dim, 1),
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        h = self.net(xt)
        return h


@utils.register_model(name='toy_joint_scorenet')
class JointScoreNetwork(nn.Module):
  """
  Simple MLP-based score network (for toy gaussian problems)
  """

  def __init__(self, config):
    super().__init__()
    self.in_dim = config.data.dim
    self.h_dim = config.model.z_dim
    self.config = config
    self.sde = VPSDE()
    self.time = nn.Sequential(
      nn.Linear(self.h_dim, self.h_dim),
      nn.ELU(),
      # added above
      nn.Linear(self.h_dim, self.h_dim),
      nn.ELU(),
      nn.Linear(self.h_dim, 1)
    )
    self.score = nn.Sequential(
      nn.Linear(self.h_dim, self.h_dim),
      nn.ELU(),
      # added above
      nn.Linear(self.h_dim, self.h_dim),
      nn.ELU(),
      nn.Linear(self.h_dim, self.in_dim)
    )

    self.net = nn.Sequential(
      nn.Linear(self.in_dim + 1, self.h_dim),
      nn.ELU(),
      nn.Linear(self.h_dim, self.h_dim * 2),
    )

  def forward(self, x, t):
    xt = torch.cat([x, t], dim=-1)
    h = self.net(xt)
    h_x, h_t = torch.chunk(h, 2, dim=1)
    out_t = self.time(h_t)
    out_x = self.score(h_x)

    # scale by standard deviation of added noise
    _, std = self.sde.marginal_prob(x, t)
    out_x /= std
    # out_x /= torch.sqrt(1-torch.clamp(t, 0., 1-1e-5))
    return [out_x, out_t]


@utils.register_model(name='toy_time_scorenet')
class TimeScoreNetwork(nn.Module):
  """
  Simple MLP-based score network (for toy gaussian problems)
  """

  def __init__(self, config):
    super().__init__()
    self.in_dim = config.data.dim
    self.h_dim = config.model.z_dim
    self.config = config
    self.net = nn.Sequential(
      nn.Linear(self.in_dim + 1, self.h_dim),
      nn.ELU(),
      nn.Linear(self.h_dim, self.h_dim),
      nn.ELU(),
      nn.Linear(self.h_dim, self.h_dim),
      nn.ELU(),
      nn.Linear(self.h_dim, 1),
    )

  def forward(self, x, t):
    xt = torch.cat([x, t], dim=-1)
    h = self.net(xt)
    return h


# @utils.register_model(name='toy_param_mvn_scorenet')
# class MVNParamScoreNetwork(nn.Module):
#   """
#   learning the parameterized score network for multivariate gaussians (high dimensional gaussians experiment for mutual information estimation)
#   """
#
#   def __init__(self, config):
#     super().__init__()
#     self.config = config
#     self.dim = config.data.dim
#     self.h_dim = config.model.z_dim
#     # self.theta = nn.Parameter(torch.randn(1, self.dim).to(device).normal_(0, 0.05))
#     self.theta = nn.Parameter(torch.randn(1, self.dim).to(device) * 2)
#
#   def forward(self, x, t):
#     # out_t = (x - self.theta * t) @ self.theta.T
#     out_t = (x - self.theta.squeeze() * t) @ self.theta.T
#
#     if self.config.model.type == 'time':
#       return out_t
#     else:
#       out_x = -(x - self.theta * t)
#       out = [out_x, out_t]
#
#       return out


@utils.register_model(name='toy_param_mvn_mi')
class MVNParamScoreNetwork(nn.Module):
  """
  learning the parameterized score network for multivariate gaussians (high dimensional gaussians experiment for mutual information estimation)
  """

  def __init__(self, config, cov=None):
    super().__init__()
    self.config = config
    self.dim = config.data.dim
    self.h_dim = config.model.z_dim
    self.theta = nn.Parameter(
      torch.randn(self.dim, self.dim).to(device).normal_(0, 0.05))

  def forward(self, x, t):
    id_mat = torch.eye(self.dim).to(x.device).view(1, 1, self.dim, self.dim)

    # resize things for batching
    t = t.view(-1, 1, 1, 1)
    theta_mat = self.theta.view(1, 1, self.dim, self.dim)

    # items needed
    new_cov = id_mat + (t ** 2 * theta_mat)
    new_cov_inv = torch.inverse(new_cov)

    term1 = (t[:, :, 0, 0] * torch.einsum('bii->b', (
          theta_mat @ new_cov_inv).squeeze()).unsqueeze(1))
    term2 = (t * (x.view(-1, 1, 1, self.dim) @ new_cov_inv @ theta_mat \
                  @ new_cov_inv @ x.view(
      -1, 1, 1, self.dim).permute(0, 1, 3, 2)))[:, :, 0, 0]
    out_t = -term1 + term2

    if self.config.model.type == 'time':
      return out_t
    else:
      out_x = -(x.view(-1, 1, 1, self.dim) @ new_cov_inv).squeeze()
      return out_x, out_t


@utils.register_model(name='toy_param_scorenet')
class ParamScoreNetwork(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.theta = nn.Parameter(torch.randn(1).normal_(0, 0.01))
    self.config = config
    self.sigma_q = config.data.sigmas[0]

  # this is the parameterization with (0.1 + theta*var)
  # def forward(self, xt):
  #   x, t = torch.chunk(xt, 2, dim=-1)
  #   denom = (0.1 + self.theta * t ** 2)
  #   out_x = -x / denom
  #   # out_t = -(self.theta * t)/(2*math.pi * denom)
  #   # out_t = out_t + (x**2 * self.theta * t)/(2 * (denom)**2)
  #
  #   out_t = -(self.theta * t ** 2 - x ** 2 + 0.1) * (self.theta * t)
  #   out_t = out_t / (denom ** 2)
  #
  #   if self.config.model.type == 'time':
  #     return out_t
  #   else:
  #     out_x = -x / denom
  #     return [out_x, out_t]

  def forward(self, x, t):
    coef1 = 1. - self.sigma_q**2
    denom = (1 - coef1 * t**2)

    out_t = (-coef1 * (self.theta - x)**2 - (coef1**2)*(t**2) + coef1) * t
    out_t = out_t/(denom**2)

    if self.config.model.type == 'time':
      return out_t
    else:
      out_x = -(x - self.theta)/denom
      return [out_x, out_t]