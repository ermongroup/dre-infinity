import torch
import numpy as np
from scipy import integrate
from models import utils as mutils
import torch.autograd as autograd

from torchdiffeq import odeint
from functools import partial


def get_toy_density_ratio_fn(rtol=1e-6, atol=1e-6, method='RK45', eps=1e-5):
  """Create a function to compute the density ratios of a given point.
  """

  def ratio_fn(score_model, x, score_type):
    with torch.no_grad():
      def ode_func(t, y, x, score_model):
        score_model.eval()
        t = (torch.ones(x.size(0)) * t).to(x.device).view(-1, 1)
        x = x.to(x.device)

        if score_type == 'joint':
          rx = score_model(x, t)[-1]
        else:
          rx = score_model(x, t)
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # now just a function of t
      p_get_rx = partial(ode_func, x=x, score_model=score_model)
      # TODO: flipped (1, eps) for toy datasets
      solution = integrate.solve_ivp(p_get_rx, (eps, 1.),
                                     np.zeros((x.shape[0],)),
                                     method=method, rtol=rtol, atol=atol)
      nfe = solution.nfev
      density_ratio = solution.y[:, -1]
      print("ratio computation took {} function evaluations.".format(nfe))

      return density_ratio, nfe

  return ratio_fn


def get_density_ratio_fn(sde, inverse_scaler, rtol=1e-6,
                         atol=1e-6, method='RK45', eps=1e-5):
  """Create a function to compute the density ratios of a given point.
  NOTE: this is the one that's being used for the DDPM noise schedule!
  this function is not actually used in the code.
  """

  def ratio_fn(score_model, x):
    with torch.no_grad():
      def ode_func(t, y, x, score_model):
        score_fn = mutils.get_time_score_fn(sde, score_model, train=False, continuous=True)

        t = (torch.ones(x.size(0)) * t).to(x.device)
        t = t.detach()
        x = x.to(x.device)
        rx = score_fn(x, t)  # get timewise-scores only
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # now just a function of t
      p_get_rx = partial(ode_func, x=x, score_model=score_model)
      # TODO: flipped (eps, 1) for DDPM noise
      solution = integrate.solve_ivp(p_get_rx, (1., eps),
                                     np.zeros((x.shape[0],)) + eps,
                                     method=method, rtol=rtol, atol=atol)
      nfe = solution.nfev
      density_ratio = solution.y[:, -1]
      print("ratio computation took {} function evaluations.".format(nfe))

      # compute "approximate" bpds. corresponds to DIRECT method in TRE paper
      # (https://arxiv.org/pdf/2006.12204.pdf page 8)
      shape = x.shape
      N = np.prod(shape[1:])

      log_qp = density_ratio
      log_p = sde.prior_logp(x).cpu().detach().numpy()
      assert log_qp.shape == log_p.shape

      # for actual bpd evaluation
      log_q = (log_qp + log_p)

      print(log_qp[0:10])
      print(log_p[0:10])
      print('log_qp: {}'.format(log_qp.mean()))
      print('log_p: {}'.format(log_p.mean()))

      # compute bpd
      bpd = -log_q / np.log(2)
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      # offset = 8.
      bpd = bpd + offset  # (batch_size, )
      return bpd, density_ratio, nfe

  return ratio_fn


def get_density_ratio_fn_flow(sde, inverse_scaler, rtol=1e-6,
                         atol=1e-6, method='RK45', eps=1e-5):
  """Create a function to compute the density ratios of a given point.
  NOTE: this is the one that's being used for the DDPM noise schedule!
  """

  def ratio_fn(score_model, x, flow_log_det, log_det_logit):
    with torch.no_grad():
      def ode_func(t, y, x, score_model):
        score_fn = mutils.get_time_score_fn(sde, score_model, train=False, continuous=True)

        t = (torch.ones(x.size(0)) * t).to(x.device)
        t = t.detach()
        x = x.to(x.device)
        rx = score_fn(x, t)  # get timewise-scores only
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # now just a function of t
      p_get_rx = partial(ode_func, x=x, score_model=score_model)
      # TODO: flipped (eps, 1) for DDPM noise
      solution = integrate.solve_ivp(p_get_rx, (1., eps),
                                     np.zeros((x.shape[0],)) + eps,
                                     method=method, rtol=rtol, atol=atol)
      nfe = solution.nfev
      density_ratio = solution.y[:, -1]
      print("ratio computation took {} function evaluations.".format(nfe))

      # compute "approximate" bpds. corresponds to DIRECT method in TRE paper
      # (https://arxiv.org/pdf/2006.12204.pdf page 8)
      shape = x.shape
      N = np.prod(shape[1:])

      log_qp = density_ratio
      log_p = sde.prior_logp(x).cpu().detach().numpy()
      assert log_qp.shape == log_p.shape

      # for actual bpd evaluation
      log_q = (log_qp + log_p)

      print(log_qp[0:10])
      print(log_p[0:10])
      print('log_qp: {}'.format(log_qp.mean()))
      print('log_p: {}'.format(log_p.mean()))

      # compute bpd
      # bpd = -log_q / np.log(2)
      # bpd = bpd / N
      # # A hack to convert log-likelihoods to bits/dim
      # offset = 8. + inverse_scaler(-1.)
      # bpd = bpd + offset  # (batch_size, )

      # compute bpd in image space: add jacobian from flow network and logit transformation
      # NOTE: flow_log_det will be 0. if we invert the flow to map z back to image space
      bpd = (-(log_q.sum() + flow_log_det.sum()) - log_det_logit) / (np.log(2) * np.prod(shape))
      offset = 8.
      bpd = bpd + offset  # (1,)
      return bpd, density_ratio, nfe

  return ratio_fn


def get_z_interp_density_ratio_fn_flow(sde, inverse_scaler, mlp=False, rtol=1e-6,
                         atol=1e-6, method='RK45', eps=1e-5):
  """Create a function to compute the density ratios of a given point.
  NOTE: this is the one that's being used for the DDPM noise schedule!
  TODO: we are using this function to evaluate q(x) = MNIST, p(x) = flow trained on MNIST
  """
  # print('I am in the correct DRE function!')
  def ratio_fn(score_model, flow, x):
    with torch.no_grad():
      def ode_func(t, y, x, score_model):
        score_fn = mutils.get_time_score_fn(sde, score_model, train=False, continuous=True)

        n = x.size(0)
        t = (torch.ones(n) * t).to(x.device)
        t = t.detach()
        x = x.to(x.device)
        rx = score_fn(x, t)  # get timewise-scores only
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # now just a function of t
      batch = x.view(x.size(0), -1) if mlp else x
      p_get_rx = partial(ode_func, x=batch, score_model=score_model)
      # TODO: flipped (eps, 1) for DDPM noise
      solution = integrate.solve_ivp(p_get_rx, (1., eps),
                                     np.zeros((x.shape[0],)) + eps,
                                     method=method, rtol=rtol, atol=atol)
      nfe = solution.nfev
      density_ratio = solution.y[:, -1]
      print("ratio computation took {} function evaluations.".format(nfe))

      # compute "approximate" bpds. corresponds to DIRECT method in TRE paper
      # (https://arxiv.org/pdf/2006.12204.pdf page 8)
      shape = x.shape
      N = np.prod(shape[1:])

      log_qp = density_ratio
      # TODO
      log_p = sde.prior_logp(flow, x).cpu().detach().numpy()
      assert log_qp.shape == log_p.shape

      # for actual bpd evaluation
      log_q = (log_qp + log_p)

      print(log_qp[0:10])
      print(log_p[0:10])
      print('log_qp: {}'.format(log_qp.mean()))
      print('log_p: {}'.format(log_p.mean()))

      ####
      # this is equivalent to the thing below, but preserves the array
      # bpd = -(log_q) / np.log(2)
      # bpd = bpd / N
      # offset = 7.
      # bpd = bpd + offset
      ####

      # log_det_logit is 0 here, so we removed it
      # this gives you a scalar value
      bpd = (-log_q.sum()) / (np.log(2) * np.prod(shape))
      offset = 7.  # bc we've rescaled to [-1, 1]

      bpd = bpd + offset  # (1,)
      return bpd, density_ratio, nfe

  return ratio_fn


def get_pathwise_density_ratio_fn(sde, inverse_scaler, rtol=1e-5, atol=1e-5,
                            method='RK45', eps=1e-5):
  """Create a function to compute the density ratios of a given point. this
  requires a model to have been trained via the joint objective
  """
  def ratio_fn(score_model, x):
    with torch.no_grad():
      # TODO: this is a single possible trajectory for y(t)!
      def y_func(t, x, z):
        return x + t[:, None, None, None] * (z - x)

      def f_y(t, x, z):
        return (z - x)

      # let's compute the first integral in the r(x) expression
      def ode_func(t, y, x, z, score_model):
        """NOTE: y is a dummy variable here. yt refers to y(t)"""
        score_fn = mutils.get_score_fn(sde, score_model, train=False,
                                       continuous=True)

        t = (torch.ones(x.size(0)) * t).to(x.device)
        # TODO: make sure you have the order correct if you try this with ddpm
        T = (torch.ones(x.size(0))).to(x.device)  # T = 1
        x = x.to(x.device)
        z = z.to(x.device)
        yT = z

        xy = yT + t[:, None, None, None] * (x - yT)
        score_x = score_fn(xy, T)[0]
        rx = torch.sum(score_x * (x - yT), dim=[1, 2, 3])
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # sample a z to compute your y(t)
      z = sde.prior_sampling(x.shape)
      p_get_rx = partial(ode_func, x=x, z=z, score_model=score_model)
      # TODO: check direction if not using VPSDE
      solution = integrate.solve_ivp(p_get_rx, (1, eps),
                                     np.zeros((x.shape[0],)), method=method,
                                     rtol=rtol, atol=atol)
      nfe = solution.nfev
      term1 = solution.y[:, -1]

      # now we need a second ode function for integrating in the second term
      def ode_func2(t, y, x, z, score_model):
        t = (torch.ones(x.size(0)) * t).to(x.device)
        x = x.to(x.device)
        z = z.to(x.device)
        yt = y_func(t, x, z)

        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)
        score_x, score_t = score_fn(yt, t)
        rx = score_t + torch.sum(f_y(t, x, z) * score_x, dim=[1, 2, 3])
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # TODO: check direction of integration if not using VPSDE
      p2_get_rx = partial(ode_func2, x=x, z=z, score_model=score_model)
      solution = integrate.solve_ivp(p2_get_rx, (1, eps),
                                     np.zeros((x.shape[0],)), method=method,
                                     rtol=rtol, atol=atol)
      term2 = solution.y[:, -1]
      nfe2 = solution.nfev

      print('took a total of  {} function evaluations'.format(nfe + nfe2))
      density_ratio = term1 + term2

      # compute "approximate" bpds. corresponds to DIRECT method in TRE paper
      # (https://arxiv.org/pdf/2006.12204.pdf page 8)
      shape = x.shape
      N = np.prod(shape[1:])

      log_qp = density_ratio
      log_p = sde.prior_logp(x).cpu().detach().numpy()
      assert log_qp.shape == log_p.shape
      log_q = (log_qp + log_p).mean()

      print(log_qp[0:10])
      print(log_p[0:10])
      print('log_qp: {}'.format(log_qp.mean()))
      print('log_p: {}'.format(log_p.mean()))

      # compute bpd
      bpd = -log_q / np.log(2)
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset
      return bpd, density_ratio, nfe

  return ratio_fn


def get_z_interp_pathwise_density_ratio_fn(sde, inverse_scaler, rtol=1e-5, atol=1e-5,
                            method='RK45', eps=1e-5):
  """Create a function to compute the density ratios of a given point. this
  requires a model to have been trained via the joint objective
  """
  def ratio_fn(score_model, flow, x):
    with torch.no_grad():
      # TODO: this is a single possible trajectory for y(t)!
      def y_func(t, x, z):
        return x + t[:, None, None, None] * (z - x)

      def f_y(t, x, z):
        return (z - x)

      # let's compute the first integral in the r(x) expression
      def ode_func(t, y, x, z, score_model):
        """NOTE: y is a dummy variable here. yt refers to y(t)"""
        score_fn = mutils.get_score_fn(sde, score_model, train=False,
                                       continuous=True)

        t = (torch.ones(x.size(0)) * t).to(x.device)
        # TODO: make sure you have the order correct if you try this with ddpm
        T = (torch.ones(x.size(0))).to(x.device)  # T = 1
        x = x.to(x.device)
        z = z.to(x.device)
        yT = z

        xy = yT + t[:, None, None, None] * (x - yT)
        score_x = score_fn(xy, T)[0]
        rx = torch.sum(score_x * (x - yT), dim=[1, 2, 3])
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # sample a z to compute your y(t)
      # TODO: do we want to sample here? we could also fix everything to 0
      # z = sde.prior_sampling(x.shape)
      z = torch.zeros_like(x)

      p_get_rx = partial(ode_func, x=x, z=z, score_model=score_model)
      # TODO: check direction if not using VPSDE
      solution = integrate.solve_ivp(p_get_rx, (1, eps),
                                     np.zeros((x.shape[0],)), method=method,
                                     rtol=rtol, atol=atol)
      nfe = solution.nfev
      term1 = solution.y[:, -1]

      # now we need a second ode function for integrating in the second term
      def ode_func2(t, y, x, z, score_model):
        t = (torch.ones(x.size(0)) * t).to(x.device)
        x = x.to(x.device)
        z = z.to(x.device)
        yt = y_func(t, x, z)

        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)
        score_x, score_t = score_fn(yt, t)
        rx = score_t + torch.sum(f_y(t, x, z) * score_x, dim=[1, 2, 3])
        rx = np.reshape(rx.detach().cpu().numpy(), -1)

        return rx

      # TODO: check direction of integration if not using VPSDE
      p2_get_rx = partial(ode_func2, x=x, z=z, score_model=score_model)
      solution = integrate.solve_ivp(p2_get_rx, (1, eps),
                                     np.zeros((x.shape[0],)), method=method,
                                     rtol=rtol, atol=atol)
      term2 = solution.y[:, -1]
      nfe2 = solution.nfev

      print('took a total of  {} function evaluations'.format(nfe + nfe2))
      density_ratio = term1 + term2

      # compute "approximate" bpds. corresponds to DIRECT method in TRE paper
      # (https://arxiv.org/pdf/2006.12204.pdf page 8)
      shape = x.shape
      N = np.prod(shape[1:])

      log_qp = density_ratio
      log_p = sde.prior_logp(flow, x).cpu().detach().numpy()
      assert log_qp.shape == log_p.shape
      log_q = (log_qp + log_p).mean()

      print(log_qp[0:10])
      print(log_p[0:10])
      print('log_qp: {}'.format(log_qp.mean()))
      print('log_p: {}'.format(log_p.mean()))

      # compute bpd
      bpd = -log_q / np.log(2)
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset
      return bpd, density_ratio, nfe

  return ratio_fn