import torch
import torch.autograd as autograd
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr,
                           betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay,
                           amsgrad=config.optim.amsgrad)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def toy_optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def joint_loss(scorenet, sde, qx, device, eps=1e-5, likelihood_weighting=False):
  """
  in objective, T = [0, 1]
  px, qx, xt: (batch_size, 1)
  t: (batch_size, 1)
  """
  # sample appropriate data
  n = len(qx)
  t = torch.rand(n, 1) * (1 - eps) + eps
  t = t.to(device)
  px = torch.randn_like(qx).to(device)
  mean, std = sde.marginal_prob(qx, t)
  xt = mean + px * std

  # device things
  px = px.to(device)  # noise
  qx = qx.to(device)  # data
  xt = xt.to(device)  # interp
  t = t.to(device)

  # set up utils for reweighting if needed
  if not likelihood_weighting:
    weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2
  else:
    weighting_fn = lambda t: sde.sde(torch.zeros_like(t), t)[1] ** 2

  def grad_weighting_fn(t):
    with torch.enable_grad():
      t.requires_grad_()
      return autograd.grad(weighting_fn(t).sum(), t)[0]

  # boundary conditions
  t0 = torch.zeros((len(px), 1)).to(px.device) + eps
  t1 = torch.ones((len(qx), 1)).to(qx.device)

  # the appropriate weighting functions
  lambda_t = weighting_fn(t)
  lambda_t0 = weighting_fn(t0)
  lambda_t1 = weighting_fn(t1)
  lambda_dt = grad_weighting_fn(t)

  # reweighted version
  term1 = (2 * scorenet(qx, t0)[-1]) * lambda_t0  # T=0 is data
  term2 = (2 * scorenet(px, t1)[-1]) * lambda_t1  # T=1 is noise

  # need to differentiate score wrt t
  score_x, xt_score = scorenet(xt, t)

  # dsm_loss
  dsm_loss = torch.square(score_x + px / std.to(device))
  dsm_loss = dsm_loss * lambda_t

  with torch.enable_grad():
    t.requires_grad_(True)
    xt_score_dt = autograd.grad(scorenet(xt, t)[-1].sum(), t, create_graph=True)[0]
  term3 = (2 * xt_score_dt) * lambda_t
  term4 = (2 * xt_score) * lambda_dt
  term5 = (xt_score ** 2) * lambda_t

  loss = dsm_loss + term1 - term2 + term3 + term4 + term5

  # 1-d so we can just take the mean rather than summing
  return loss.mean()


# @title Define time-wise loss
def time_loss(scorenet, sde, qx, device, eps=1e-5, likelihood_weighting=False):
  """
  in objective, T = [0, 1]
  px, qx, xt: (batch_size, 1)
  t: (batch_size, 1)
  """
  # sample appropriate data
  n = len(qx)
  t = torch.rand(n, 1) * (1 - eps) + eps
  t = t.to(device)
  px = torch.randn_like(qx).to(qx.device)
  mean, std = sde.marginal_prob(qx, t)
  xt = mean + px * std

  # device things
  px = px.to(device)  # noise
  qx = qx.to(device)  # data
  xt = xt.to(device)  # interp
  t = t.to(device)

  # set up utils for reweighting if needed
  if not likelihood_weighting:
    weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2
  else:
    weighting_fn = lambda t: sde.sde(torch.zeros_like(t), t)[1] ** 2

  def grad_weighting_fn(t):
    with torch.enable_grad():
      t.requires_grad_()
      return autograd.grad(weighting_fn(t).sum(), t)[0]

  # boundary conditions
  t0 = torch.zeros((len(px), 1)).to(px.device) + eps
  t1 = torch.ones((len(qx), 1)).to(qx.device)

  # the appropriate weighting functions
  lambda_t = weighting_fn(t)
  lambda_t0 = weighting_fn(t0)
  lambda_t1 = weighting_fn(t1)
  lambda_dt = grad_weighting_fn(t)

  # reweighted version
  term1 = (2 * scorenet(qx, t0)) * lambda_t0  # T=0 is data
  term2 = (2 * scorenet(px, t1)) * lambda_t1  # T=1 is noise

  # need to differentiate score wrt t
  xt_score = scorenet(xt, t)
  with torch.enable_grad():
    t.requires_grad_(True)
    xt_score_dt = autograd.grad(scorenet(xt, t).sum(), t, create_graph=True)[0]
  term3 = (2 * xt_score_dt) * lambda_t
  term4 = (2 * xt_score) * lambda_dt
  term5 = (xt_score ** 2) * lambda_t

  loss = term1 - term2 + term3 + term4 + term5


  # 1-d so we can just take the mean rather than summing
  return loss.mean()


# TODO: this is used for toy MI exp
def toy_joint_score_estimation(sde, scorenet, qx, eps=1e-5, likelihood_weighting=False):
  """
  in objective, T = [0, 1]
  px, qx, xt: (batch_size, 1)
  t: (batch_size, 1)
  """
  # sample appropriate data
  n = len(qx)
  t = torch.rand(n, 1) * (1 - eps)
  px = torch.randn_like(qx)
  mean, std = sde.marginal_prob(qx, t)
  xt = mean + px * std

  # device things
  px = px.to(device)  # noise
  qx = qx.to(device)  # data
  xt = xt.to(device)  # interp
  t = t.to(device)
  t = t.detach()

  # get data score -- this is SSM!
  xt.requires_grad_()
  vectors = torch.randn_like(xt, device=xt.device)
  score_x, score_t = scorenet(xt, t)
  grad1 = torch.cat([score_x, score_t], dim=-1)
  gradv = torch.sum(score_x * vectors)
  grad2 = autograd.grad(gradv, xt, create_graph=True)[0]

  # set up utils for reweighting if needed
  if not likelihood_weighting:
    weighting_fn = lambda t: torch.ones_like(t)
  else:
    weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2

  def grad_weighting_fn(t):
    with torch.enable_grad():
      t.requires_grad_()
      return autograd.grad(weighting_fn(t).sum(), t)[0]

  # boundary conditions
  t0 = torch.zeros((len(px), 1)).to(px.device) + eps
  t1 = torch.ones((len(qx), 1)).to(qx.device) - eps

  # the appropriate weighting functions
  lambda_t = weighting_fn(t)
  lambda_t0 = weighting_fn(t0)
  lambda_t1 = weighting_fn(t1)
  if not likelihood_weighting:
    lambda_dt = 0.
  else:
    lambda_dt = grad_weighting_fn(t)

  # SSM loss (technically has the s(x,t)**2 term in there too)
  ssm_loss1 = (torch.sum(grad1 * grad1, dim=-1) / 2.).view(
    lambda_t.size()) * lambda_t
  ssm_loss2 = torch.sum(vectors * grad2, dim=-1).view(
    lambda_t.size()) * lambda_t
  ssm_loss = ssm_loss1 + ssm_loss2
  # rw_ssm_loss = ssm_loss * ssm_alpha

  # reweighted version
  term1 = (scorenet(px, t0)[-1]) * lambda_t0  # T=0 is noise
  term2 = (scorenet(qx, t1)[-1]) * lambda_t1  # T=1 is data

  # need to differentiate score wrt t
  with torch.enable_grad():
    t.requires_grad_(True)
    xt_score_dt = \
    autograd.grad(scorenet(xt, t)[-1].sum(), t, create_graph=True)[0]
  term3 = (xt_score_dt) * lambda_t
  term4 = score_t * lambda_dt

  time_loss = term1 - term2 + term3 + term4
  loss = ssm_loss + time_loss

  # 1-d so we can just take the mean rather than summing
  return loss.mean()


# TODO: this is used for toy timewise exp
def toy_timewise_score_estimation(scorenet, samples, t, eps=1e-5, reweight=False):
  """
  in objective, T = [0, 1]
  px, qx, xt: (batch_size, 1)
  t: (batch_size, 1)

  we are reweighting the output of the score network (most recent version)
  """
  px, qx, xt = samples
  px = px.to(device)
  qx = qx.to(device)
  xt = xt.to(device)
  t = t.to(device)

  # reweighted version
  t0 = torch.zeros((len(px), 1)).to(px.device) + eps
  t1 = torch.ones((len(qx), 1)).to(qx.device)

  if reweight:
    lambda_t = (1 - t ** 2).squeeze()
    lambda_t0 = (1 - t0.squeeze() ** 2)
    lambda_t1 = (1 - t1.squeeze() ** 2 + eps ** 2)
    lambda_dt = (-2 * t.squeeze())
  else:
    lambda_t = lambda_t0 = lambda_t1 = 1
    lambda_dt = 0

  term1 = (2 * scorenet(px, t0)).squeeze() * lambda_t0
  term2 = (2 * scorenet(qx, t1)).squeeze() * lambda_t1

  # need to differentiate score wrt t
  t.requires_grad_(True)
  xt_score = scorenet(xt, t)  # dim = 1
  xt_score_dt = autograd.grad(xt_score.sum(), t, create_graph=True)[0]
  term3 = (2 * xt_score_dt).squeeze() * lambda_t
  term4 = (xt_score).squeeze() * lambda_dt
  term5 = (xt_score ** 2).squeeze() * lambda_t

  loss = term1 - term2 + term3 + term4 + term5

  # 1-d so we can just take the mean rather than summing
  return loss.mean(), term3.mean(), term4.mean(), term5.mean(), term1.mean(), term2.mean()


def time_loss_with_constant(scorenet, true_score, sde, qx, device, eps=1e-5,
                              likelihood_weighting=False):
  """
  in objective, T = [0, 1]
  px, qx, xt: (batch_size, 1)
  t: (batch_size, 1)
  """
  # sample appropriate data
  n = len(qx)
  t = torch.rand(n, 1) * (1 - eps) + eps
  px = torch.randn_like(qx)
  mean, std = sde.marginal_prob(qx, t)
  xt = mean + px * std

  # device things
  px = px.to(device)  # noise
  qx = qx.to(device)  # data
  xt = xt.to(device)  # interp
  t = t.to(device)

  # set up utils for reweighting if needed
  if not likelihood_weighting:
    weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2
  else:
    weighting_fn = lambda t: sde.sde(torch.zeros_like(t), t)[1] ** 2

  def grad_weighting_fn(t):
    with torch.enable_grad():
      t = t.detach()  # detaching or not doesn't make a difference
      t.requires_grad_()
      return autograd.grad(weighting_fn(t).sum(), t)[0]

  # boundary conditions
  t0 = torch.zeros((len(px), 1)).to(px.device) + eps
  t1 = torch.ones((len(qx), 1)).to(qx.device)

  # the appropriate weighting functions
  lambda_t = weighting_fn(t)
  lambda_t0 = weighting_fn(t0)
  lambda_t1 = weighting_fn(t1)
  lambda_dt = grad_weighting_fn(t)

  # reweighted version
  term1 = (2 * scorenet(x=qx, t=t0)) * lambda_t0  # T=0 is data
  term2 = (2 * scorenet(x=px, t=t1)) * lambda_t1  # T=1 is noise

  # need to differentiate score wrt t
  xt_score = scorenet(x=xt, t=t)
  t = t.detach()
  with torch.enable_grad():
    t.requires_grad_(True)
    xt_score_dt = autograd.grad(scorenet(x=xt, t=t).sum(), t,
                                create_graph=True)[0]
  term3 = (2 * xt_score_dt) * lambda_t
  term4 = (2 * xt_score) * lambda_dt
  term5 = (xt_score ** 2) * lambda_t

  loss = term1 - term2 + term3 + term4 + term5

  true_xt_score_dt = true_score(x=xt, t=t) ** 2
  constant = lambda_t * true_xt_score_dt
  return (loss + constant).mean()


def time_loss_squared_dist(scorenet, true_score, sde, qx, device, eps=1e-5,
                           likelihood_weighting=False):
  """
  in objective, T = [0, 1]
  px, qx, xt: (batch_size, 1)
  t: (batch_size, 1)
  """
  # sample appropriate data
  n = len(qx)
  t = torch.rand(n, 1) * (1 - eps) + eps
  px = torch.randn_like(qx)
  mean, std = sde.marginal_prob(qx, t)
  xt = mean + px * std

  # device things
  xt = xt.to(device)  # interp
  t = t.to(device)

  # set up utils for reweighting if needed
  if not likelihood_weighting:
    weighting_fn = lambda t: sde.marginal_prob(torch.zeros_like(t), t)[1] ** 2
  else:
    weighting_fn = lambda t: sde.sde(torch.zeros_like(t), t)[1] ** 2

  lambda_t = weighting_fn(t)
  squared_dist = (scorenet(x=xt, t=t) - true_score(x=xt, t=t)) ** 2
  return (lambda_t * squared_dist).mean()


def get_step_fn(sde, train, joint=False, optimize_fn=None, reweight=False):
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
  if not joint:
    # loss_fn = time_loss
    loss_fn = toy_timewise_score_estimation
  else:
    # loss_fn = joint_loss
    loss_fn = toy_joint_score_estimation

  if reweight:
    print('reweighting loss function!')

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
      model.train()
      optimizer = state['optimizer']
      optimizer.zero_grad()
      if joint:
        # TODO: lmao not good
        # loss, loss1, loss2, loss3, loss4, edge1, edge2 = loss_fn(model, batch, t)
        loss = loss_fn(sde, model, batch, likelihood_weighting=reweight)
      else:
        loss, loss1, loss2, loss3, edge1, edge2 = loss_fn(sde, model, batch, likehood_weighting=reweight)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
    else:
      model.eval()
      with torch.no_grad():
        if joint:
          # loss, loss1, loss2, loss3, loss4, edge1, edge2 = loss_fn(model, batch, t)
          loss = loss_fn(sde, model, batch, likelihood_weighting=reweight)
        else:
          loss, loss1, loss2, loss3, edge1, edge2 = loss_fn(sde, model, batch, likelihood_weighting=reweight)
    # return loss in a single dictionary
    loss_dict = {
      'loss': loss.item(),
      # 'loss1': loss1.item(),
      # 'loss2': loss2.item(),
      # 'loss3': loss3.item(),
      # 'edge1': edge1.item(),
      # 'edge2': edge2.item()
    }
    # ugh
    # if joint:
    #   loss_dict['loss4'] = loss4.item()
    return loss_dict

  return step_fn
