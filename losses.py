"""All functions related to loss computation and optimization.
"""
import math
import torch
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from models import utils as mutils
from datasets import logit_transform
import matplotlib.pyplot as plt


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    print('using amsgrad for Adam? {}'.format(config.optim.amsgrad))
    print('using learning rate: {}'.format(config.optim.lr))
    optimizer = optim.Adam(params, lr=config.optim.lr,
                           betas=(config.optim.beta1, 0.999),
                           eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay,
                           amsgrad=config.optim.amsgrad)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`. Incorporates learning rate scheduling"""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    # TODO: this was present before, where warmup=5000. but it reverts everything
    # back to 0.001 afterwards???
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

    # incorporate schedule by adjusting learning rate
    if step > 0 and step % config.training.snapshot_freq == 0:
      for g in optimizer.param_groups:
        g['lr'] = 0.5 * lr * (
              1. + np.cos((step / config.training.n_iters) * np.pi))
      print('updating learning rate: now {}'.format(g['lr']))
    # as expected
    # print(optimizer.param_groups[0]['lr'])

  return optimize_fn


def v2_optimization_manager(config):
  """Returns an optimize_fn based on `config`. Incorporates learning rate scheduling"""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if step < warmup and warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

    # incorporate schedule by adjusting learning rate
    if step > 0 and step % config.training.snapshot_freq == 0:
      for g in optimizer.param_groups:
        print('current learning rate is: {}'.format(g['lr']))
        g['lr'] = 0.5 * lr * (
              1. + np.cos((step / config.training.n_iters) * np.pi))
      print('updating learning rate: now {}'.format(g['lr']))

  return optimize_fn


def get_joint_sde_loss_fn(sde, train, alpha=0, reduce_mean=True, continuous=True,
                          likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    alpha: coefficient for balancing data score/time score losses
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(score_model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    # reweight dsm_loss and time_loss components
    if alpha > 0:
      # assert alpha < 1
      dsm_alpha = alpha
      time_alpha = (1. - alpha)
    else:
      dsm_alpha = 1.
      time_alpha = 1.

    # get model and data
    score_fn = mutils.get_score_fn(sde, score_model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    t = t.detach()
    px = torch.randn_like(batch)  # noise
    mean, std = sde.marginal_prob(batch, t)
    xt = mean + std[:, None, None, None] * px

    # reweight score terms
    if not likelihood_weighting:
      weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2
    else:
      weighting_fn = lambda t: sde.sde(torch.zeros_like(t), t)[1] ** 2

    def grad_weighting_fn(t):
      with torch.enable_grad():
        t.requires_grad_()
        return autograd.grad(weighting_fn(t).sum(), t)[0]

    # boundary conditions
    t0 = torch.zeros(len(batch), device=batch.device) + eps
    t1 = torch.ones(len(px), device=batch.device)

    # the appropriate weighting functions
    lambda_t = weighting_fn(t)
    lambda_t0 = weighting_fn(t0)
    lambda_t1 = weighting_fn(t1)
    lambda_dt = grad_weighting_fn(t)

    # get data score
    score_x, score_t = score_fn(xt, t)
    losses = torch.square(score_x + px / std[:, None, None, None])
    dsm_loss = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * lambda_t
    rw_dsm_loss = dsm_loss * dsm_alpha

    # need to differentiate score with respect to t
    with torch.enable_grad():
      t.requires_grad_()
      xt_score_dt = autograd.grad(score_fn(xt, t)[-1].sum(), t, create_graph=True)[0]
    xt_score_dt = xt_score_dt.view(batch.shape[0], -1)

    # this is still part of the loss
    loss1 = (2 * torch.sum(xt_score_dt, dim=-1)) * lambda_t
    loss2 = (2 * score_t.squeeze()) * lambda_dt
    loss3 = (score_t.squeeze() ** 2) * lambda_t

    # now the boundary conditions!
    edge1_t = score_fn(batch, t0)[-1]  # data is T = 0
    edge2_t = score_fn(px, t1)[-1]  # noise is T = 1

    # we only want the scores wrt t
    edge1 = (2 * edge1_t.squeeze()) * lambda_t0
    edge2 = (2 * edge2_t.squeeze()) * lambda_t1
    time_loss = loss1 + loss2 + loss3 + edge1 - edge2
    rw_time_loss = time_alpha * time_loss

    # aggregate all the time loss components
    loss = rw_dsm_loss + rw_time_loss

    # let's just construct the loss_dict here and return the final loss
    loss_dict = dict(loss=loss.mean(),
                     dsm_loss=dsm_loss.mean(),
                     total_time_loss=time_loss.mean(),
                     dt_time_loss=loss1.mean(),
                     time_loss=loss2.mean(),
                     time_sq_loss=loss3.mean(),
                     edge0=edge1.mean(),
                     edge1=edge2.mean(),
                     rw_dsm_loss=rw_dsm_loss.mean(),
                     rw_time_loss=rw_time_loss.mean()
                     )
    return loss.mean(), loss_dict

  return loss_fn


def get_time_sde_loss_fn(sde, train, reduce_mean=True, continuous=True,
                         likelihood_weighting=True, eps=1e-5,
                         iw=False, history=None, interpolate=True):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    energy: whether or not to use energy formulation of score network.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(score_model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_time_score_fn(sde, score_model, train=train,
                                        continuous=continuous)

    # get data
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    with torch.no_grad():
      t = torch.sort(t)[0]  # HACK
    px = torch.randn_like(batch)  # noise
    mean, std = sde.marginal_prob(batch, t)
    xt = mean + std[:, None, None, None] * px
    t = t.detach()

    # reweight score_t terms
    if iw:
      assert not likelihood_weighting
    if not likelihood_weighting:
      weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2
    else:
      weighting_fn = lambda t: sde.sde(torch.zeros_like(t), t)[1] ** 2

    def grad_weighting_fn(t):
      with torch.enable_grad():
        t.requires_grad_()
        return autograd.grad(weighting_fn(t).sum(), t)[0]

    # boundary conditions
    t0 = torch.zeros(len(xt), device=batch.device) + eps
    t1 = torch.ones(len(batch), device=batch.device)

    # need to differentiate score with respect to t
    score_t = score_fn(xt, t)
    with torch.enable_grad():
      t.requires_grad_()
      xt_score_dt = autograd.grad(score_fn(xt, t).sum(), t, create_graph=True)[0]
    xt_score_dt = xt_score_dt.view(batch.shape[0], -1)

    # the appropriate weighting functions
    lambda_t = weighting_fn(t)
    lambda_t0 = weighting_fn(t0)
    lambda_t1 = weighting_fn(t1)
    lambda_dt = grad_weighting_fn(t)

    # this is still part of the loss
    loss1 = (2 * torch.sum(xt_score_dt, dim=-1)) * lambda_t
    loss2 = (2 * score_t.squeeze()) * lambda_dt
    loss3 = (score_t.squeeze() ** 2) * lambda_t

    # now the boundary conditions!
    edge1_t = score_fn(batch, t0)  # data is T = 0
    edge2_t = score_fn(px, t1)  # noise is T = 1

    # we only want the scores wrt t
    edge1 = (2 * edge1_t.squeeze()) * lambda_t0
    edge2 = (2 * edge2_t.squeeze()) * lambda_t1

    # loss = loss1 + loss2 + loss3 + edge1 - edge2
    unweighted_loss = loss1 + loss2 + loss3 + edge1 - edge2
    time_loss_no_edges = loss1 + loss2 + loss3

    if iw:
      if train:  # don't reweight for eval
        t_numpy = t.cpu().detach().numpy()
        if interpolate:
          weights = history.weights(t_numpy)
        else:
          # no dependence on t if we are not interpolating
          weights = history.weights()
        # scale to be within [0, 1], otherwise too slow
        weights /= weights.max()  # this shouldn't do anything for the interp

        weights = torch.from_numpy(weights).float().to(unweighted_loss.device)
        # TODO: HACK, this is when the batch size doesn't evenly divide the dataset during training
        if len(weights) != len(batch):
          weights = weights[0:len(batch)]
        weights = weights.view(unweighted_loss.size())

        # all positive weights
        assert (weights > 0).all()
        loss = unweighted_loss * weights

        # update history
        # TODO: annoying, should clean this up
        if not interpolate:
          history.update_with_all_losses(
            ts=t_numpy,
            losses=time_loss_no_edges.squeeze().cpu().detach().numpy())
        else:
          history.update_with_all_losses(
            ts=t_numpy,
            losses=time_loss_no_edges.squeeze().cpu().detach().numpy(),
            weights=weights.cpu().detach().numpy())
      else:
        # for eval, no reweighting bc batch size discrepancy
        loss = unweighted_loss
        weights = torch.ones_like(loss)
    else:
      # if no iw, also just return unweighted loss
      loss = unweighted_loss
      weights = torch.ones_like(loss)

    # variance
    variance = ((loss - torch.mean(loss, dim=0, keepdim=True)) ** 2).mean()

    # collect everything in a loss_dict
    # let's just construct the loss_dict here and return the final loss
    loss_dict = dict(loss=loss.mean(),
                     unweighted_loss=unweighted_loss.mean().item(),
                     variance=variance.item(),
                     total_time_loss=loss.mean().item(),
                     dt_time_loss=loss1.mean().item(),
                     time_loss=loss2.mean().item(),
                     time_sq_loss=loss3.mean().item(),
                     edge0=edge1.mean().item(),
                     edge1=-edge2.mean().item(),
                     weights=weights
                     )

    # collect everything in a loss_dict
    # let's just construct the loss_dict here and return the final loss
    return loss.mean(), loss_dict

  return loss_fn



def get_time_sde_loss_fn_flow_z_interpolate(flow, flow_name, sde, train, mlp=False, reduce_mean=True,
                              continuous=True, likelihood_weighting=True, iw=False,
                                            history=None, eps=1e-5, interpolate=False):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    flow: pre-trained flow model for potentially doing x-space inversion
    train: `True` for training loss and `False` for evaluation loss.
    energy: whether or not to use energy formulation of score network.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(score_model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_time_score_fn(sde, score_model, train=train,
                                        continuous=continuous)

    n = batch.size(0)
    # when data enters this loop, you first want it to be [-1, 1] (checked)
    with torch.no_grad():
      flow.eval()
      z_batch = (batch + 1.) / 2.
      if flow_name in ['mintnet', 'nice', 'realnvp']:
        # undo rescaling, apply logit transform, pass through flow
        z_batch = logit_transform(z_batch)
        z_batch, _ = flow(z_batch, reverse=False)
        z_batch = z_batch.view(batch.size())
      else:
        z_batch *= 256.
        # annoying, but now we need to branch to RQ-NSF flow vs [noise, copula]
        if 'noise' in flow_name or 'copula' in flow_name:
          # apply data transform here (1/256, logit transform, mean-centering)
          z_batch = flow.module.transform_to_noise(z_batch, transform=True, train=train)
        else:
          # for the RQ-NSF flow, the data is dequantized and between [0, 256]
          # and the flow's preprocessing module takes care of normalization
          z_batch = flow.module.transform_to_noise(z_batch)
        z_batch = z_batch.view(batch.size())

    # get data
    # still need to sort at the start so that the initial weights make sense
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    with torch.no_grad():
      t = torch.sort(t)[0]  # HACK

    #   # TODO: trying this out!
    #   u0 = torch.rand(1, device=batch.device) * (sde.T - eps) + eps
    #   t = (u0 + torch.arange(1, n + 1).to(batch.device) / n) % 1

    px = torch.randn_like(batch)  # noise
    mean, std = sde.marginal_prob(z_batch, t)  # feed in z into SDE
    zt = mean + std[:, None, None, None] * px

    with torch.no_grad():
      if flow_name in ['mintnet', 'nice', 'realnvp']:
        # map z -> x via flow, then rescale to [-1, 1]
        xt = flow.module.sampling(zt, rescale=True)
        # now take our noise and map it through flow
        px = flow.module.sampling(px, rescale=True)
      else:
        if 'noise' in flow_name or 'copula' in flow_name:
          xt = flow.module.sample(zt.view(n, -1), context=None, rescale=True, transform=True, train=train)
          px = flow.module.sample(px.view(n, -1), context=None, rescale=True, transform=True, train=train)
        else:
          xt = flow.module.sample(zt.view(n, -1), context=None, rescale=True)
          px = flow.module.sample(px.view(n, -1), context=None, rescale=True)

    # reshape bc mlp
    if mlp:
      px = px.view(n, -1)
      batch = batch.view(n, -1)
      xt = xt.view(n, -1)  # mlp

    # reweight score_t terms
    if not likelihood_weighting:
      weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2
      # weighting_fn = lambda t: torch.ones_like(t)
    else:
      weighting_fn = lambda t: sde.sde(torch.zeros_like(t), t)[1] ** 2

    def grad_weighting_fn(t):
      with torch.enable_grad():
        t.requires_grad_()
        return autograd.grad(weighting_fn(t).sum(), t)[0]

    # boundary conditions
    t0 = torch.zeros(len(xt), device=batch.device) + eps
    t1 = torch.ones(len(batch), device=batch.device)

    # need to differentiate score with respect to t
    score_t = score_fn(xt, t)
    with torch.enable_grad():
      t.requires_grad_()
      xt_score_dt = autograd.grad(score_fn(xt, t).sum(), t, create_graph=True)[0]
    xt_score_dt = xt_score_dt.view(batch.shape[0], -1)

    # the appropriate weighting functions
    lambda_t = weighting_fn(t)
    lambda_t0 = weighting_fn(t0)
    lambda_t1 = weighting_fn(t1)
    lambda_dt = grad_weighting_fn(t)

    # this is still part of the loss
    loss1 = (2 * torch.sum(xt_score_dt, dim=-1)) * lambda_t
    loss2 = (2 * score_t.squeeze()) * lambda_dt
    loss3 = (score_t.squeeze() ** 2) * lambda_t

    # now the boundary conditions!
    edge1_t = score_fn(batch, t0)  # data is T = 0
    edge2_t = score_fn(px, t1)  # noise is T = 1

    # we only want the scores wrt t
    edge1 = (2 * edge1_t.squeeze()) * lambda_t0
    edge2 = (2 * edge2_t.squeeze()) * lambda_t1

    # loss = loss1 + loss2 + loss3 + edge1 - edge2
    unweighted_loss = loss1 + loss2 + loss3 + edge1 - edge2
    time_loss_no_edges = loss1 + loss2 + loss3

    if iw:
      if train:  # don't reweight for eval
        t_numpy = t.cpu().detach().numpy()
        if interpolate:
          weights = history.weights(t_numpy)
        else:
          # no dependence on t if we are not interpolating
          weights = history.weights()
        # scale to be within [0, 1], otherwise too slow
        weights /= weights.max()  # this shouldn't do anything for the interp

        weights = torch.from_numpy(weights).float().to(unweighted_loss.device)
        # TODO: HACK, this is when the batch size doesn't evenly divide the dataset during training
        if len(weights) != len(batch):
          weights = weights[0:len(batch)]
        weights = weights.view(unweighted_loss.size())

        # all positive weights
        assert (weights > 0).all()
        loss = unweighted_loss * weights

        # update history
        # TODO: annoying, should clean this up
        if not interpolate:
          history.update_with_all_losses(
            ts=t_numpy,
            losses=time_loss_no_edges.squeeze().cpu().detach().numpy())
        else:
          history.update_with_all_losses(
            ts=t_numpy,
            losses=time_loss_no_edges.squeeze().cpu().detach().numpy(),
            weights=weights.cpu().detach().numpy())
      else:
        # for eval, no reweighting bc batch size discrepancy
        loss = unweighted_loss
        weights = torch.ones_like(loss)
    else:
      # if no iw, also just return unweighted loss
      loss = unweighted_loss
      weights = torch.ones_like(loss)

    # variance
    variance = ((loss - torch.mean(loss, dim=0, keepdim=True)) ** 2).mean()

    # collect everything in a loss_dict
    # let's just construct the loss_dict here and return the final loss
    loss_dict = dict(loss=loss.mean().item(),
                     unweighted_loss=unweighted_loss.mean().item(),
                     variance=variance.item(),
                     total_time_loss=loss.mean().item(),
                     dt_time_loss=loss1.mean().item(),
                     time_loss=loss2.mean().item(),
                     time_sq_loss=loss3.mean().item(),
                     edge0=edge1.mean().item(),
                     edge1=-edge2.mean().item(),
                     weights=weights
                     )
    return loss.mean(), loss_dict

  return loss_fn


def get_step_fn(sde, train, algo, joint, z_space, mlp=False, alpha=0, z_interpolate=False, eps=1e-5, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, flow=None, history=None,
                flow_name=None, iw=False, interpolate=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  assert continuous
  if joint:
    loss_fn = get_joint_sde_loss_fn(sde, train, alpha, eps=eps,
                                    reduce_mean=reduce_mean,
                                    continuous=True,
                                    likelihood_weighting=likelihood_weighting)
  else:
    if not z_space:
      # standard time score training loss
      # TODO: will need a diff version of this
      loss_fn = get_time_sde_loss_fn(sde, train, reduce_mean=reduce_mean, eps=eps,
                                      continuous=True, iw=iw, history=history,
                                      interpolate=interpolate,
                                      likelihood_weighting=likelihood_weighting)
    else:
      print('training time scores in z-space!')
      if z_interpolate:
        print('using flow to interpolate samples in z-space!')
        print('iw is set to {}'.format(iw))
        if history:
          assert not likelihood_weighting
        if interpolate:
          print('using interpolation instead of sorting t!')
        loss_fn = get_time_sde_loss_fn_flow_z_interpolate(flow, flow_name, sde, train, mlp=mlp,
                                            reduce_mean=reduce_mean, eps=eps,
                                            continuous=True, iw=iw,
                                            history=history, interpolate=interpolate,
                                            likelihood_weighting=likelihood_weighting,
                                                          )
      else:
        raise NotImplementedError
        # loss_fn = get_time_sde_loss_fn_flow(sde, train,
        #                                     reduce_mean=reduce_mean, eps=eps,
        #                                     continuous=True,
        #                                     likelihood_weighting=likelihood_weighting)


  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss, loss_dict = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss, loss_dict = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss_dict

  return step_fn