"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np
import torch.autograd as autograd


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config, name=None):
  """Create the score model."""
  model_name = config.model.name
  if name:
    print('using supplied model name {}'.format(name))
    model_name = name
  # TODO: HACK (can remove if you want, but linear embedding works best here)
  if model_name == 'nscnunet_t':
    assert config.model.embedding_type == 'linear'
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)

  return score_model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)
  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.Z_VPSDE):
    def score_fn(x, t):
      assert continuous
      # labels = t * 999
      labels = t * 1  # TODO: scaling the t's seems to hurt performance atm
      score = model_fn(x, labels)
      std = sde.marginal_prob(torch.zeros_like(x), t)[1]

      # for joint training
      if isinstance(score, list) or isinstance(score, tuple):
        score_x, score_t = score
        if len(x) < 4:
          score_x = score_x / std[:, None]
        else:
          score_x = score_x / std[:, None, None, None]
        return [score_x, score_t.squeeze()]
      else:
        if len(x) < 4:
          return score / std[:, None]
        else:
          return score / std[:, None, None, None]

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  elif isinstance(sde, sde_lib.Z_RQNSF_VPSDE) or isinstance(sde, sde_lib.Z_RQNSF_TFORM_VPSDE):
    def score_fn(x, t):
      assert continuous
      std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      score = model_fn(x, t)

      # for joint training
      if isinstance(score, list) or isinstance(score, tuple):
        score_x, score_t = score
        if len(x) < 4:
          score_x = score_x / std[:, None]
        else:
          score_x = score_x / std[:, None, None, None]
        return [score_x, score_t.squeeze()]
      else:
        if len(x) < 4:
          return score / std[:, None]
        else:
          return score / std[:, None, None, None]
  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def get_time_score_fn(sde, model, train=False, continuous=False):
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.Z_VPSDE) or isinstance(sde, sde_lib.Z_RQNSF_VPSDE) or isinstance(sde, sde_lib.Z_RQNSF_TFORM_VPSDE):
    def score_fn(x, t):
      assert continuous
      # labels = t * 999
      labels = t * 1  # TODO: scaling the t's seems to hurt performance atm
      score = model_fn(x, labels)

      if isinstance(score, list) or isinstance(score, tuple):
        score_x, score_t = score
      else:
        score_t = score
      return score_t.squeeze()


  elif isinstance(sde, sde_lib.Z_RQNSF_VPSDE) or isinstance(sde, sde_lib.Z_RQNSF_TFORM_VPSDE):
    def score_fn(x, t):
      assert continuous
      score = model_fn(x, t)  # just feed in the t's for now?
      if isinstance(score, list) or isinstance(score, tuple):
        score_x, score_t = score
      else:
        score_t = score
      return score_t.squeeze()
  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))