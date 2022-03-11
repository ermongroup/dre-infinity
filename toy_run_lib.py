import gc
import io
import os
import copy
import pickle

import numpy as np
import tensorflow as tf
import logging

import sde_lib
from models.toy_networks import *
import toy_losses, toy_mi_losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import toy_datasets
import density_ratios
from absl import flags
import torch
import torch.autograd as autograd
from utils import save_checkpoint, restore_checkpoint
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')

import wandb
FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  # Initialize model.
  score_model = mutils.create_model(config, name=config.model.name)
  assert config.model.ema is False  # this is overkill
  ema = None

  optimizer = toy_losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # create figures directory
  figures_dir = os.path.join(workdir, "figures")
  tf.io.gfile.makedirs(figures_dir)
  metrics_dir = os.path.join(workdir, "metrics")
  tf.io.gfile.makedirs(metrics_dir)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds = toy_datasets.get_dataset(config)

  # Build one-step training and evaluation functions
  optimize_fn = toy_losses.toy_optimization_manager(config)
  joint = config.training.joint
  eps = config.data.eps
  if joint:
    print('Using joint training!')
  sde = sde_lib.ToyInterpXt()

  # see if using a learning rate scheduler helps
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   config.training.n_iters//config.training.eval_freq,
                                                   eta_min=0, last_epoch=-1, verbose=False)

  # get appropriate functions
  if config.data.dataset == 'GaussiansforMI':
    train_step_fn = toy_mi_losses.get_step_fn(sde=sde, train=True, joint=joint,
                                          optimize_fn=optimize_fn,
                                          reweight=config.training.reweight)
  else:
    train_step_fn = toy_losses.get_step_fn(sde=sde, train=True, joint=joint,
                                          optimize_fn=optimize_fn,
                                          reweight=config.training.reweight)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  # in case we are estimating mutual information
  if config.data.dataset == 'GaussiansforMI':
    mi_db = []
    mi_metrics = {'step': [], 'mi': [], 'true_mi': train_ds.true_mutual_info}
  best_diff = np.inf
  best_step = 0

  for step in range(initial_step, num_train_steps + 1):
    n = config.training.batch_size
    if config.data.dataset == 'GaussiansforMI':
      batch = train_ds.sample_data(n)
      loss_dict = train_step_fn(state, batch.detach())
    else:
      # TODO: what is going on??
      eps = 1e-5
      t = torch.rand(n, 1) * (1 - eps)
      batch = train_ds.sample(n, t)
      # TODO: there are also some differences. right now timewise should work, but not joint
      loss_dict = train_step_fn(state, batch, t)

    # Execute one training step
    # loss_dict = train_step_fn(state, batch.detach())
    loss_dict['step'] = step
    wandb.log(loss_dict)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.4f" % (step, loss_dict['loss']))

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0 and step > 0:
      if config.data.dataset != 'GaussiansforMI':
        visualize(config, train_ds, score_model, savefig=figures_dir, step=step,
                  device=config.device)
      else:
        mi_db.append(estimate_mi(config, score_model, train_ds))
        visualize_mi(config, mi_db, train_ds.true_mutual_info,
                     savefig=figures_dir)
        # also save metrics
        mi_metrics['step'].append(step)
        mi_metrics['mi'] = mi_db

        # should you save checkpoints?
        diff = np.abs(mi_db[-1] - train_ds.true_mutual_info)
        if diff <= best_diff:
          best_diff = diff
          best_step = step
          mi_metrics['best_diff'] = best_diff
          mi_metrics['best_step'] = best_step
          fpath = os.path.join(checkpoint_dir, 'best_ckpt.pth')
        else:
          fpath = os.path.join(checkpoint_dir, 'ckpt.pth')
        torch.save(score_model.state_dict(), fpath)

        # save metrics
        with open(os.path.join(metrics_dir, 'metrics.p'), 'wb') as fp:
          pickle.dump(mi_metrics, fp)

        # take a scheduler step
        if config.optim.scheduler:
          scheduler.step()


def visualize(config, dataset, model, savefig=None, step=None, device=None):
  model.eval()

  with torch.no_grad():
    # Build density ratio estimation functions
    density_ratio_fn = density_ratios.get_toy_density_ratio_fn(
      eps=config.data.eps)

    if config.data.dataset == 'PeakedGaussians':
      grid_size = 10000
      left_bound = -2
      right_bound = 2
      mesh = torch.linspace(left_bound, right_bound, grid_size).view(-1, 1).to(device)
    else:
      # what if instead of a mesh you just sampled from both datasets
      qs = dataset.q.sample((5000, config.data.dim))
      ps = dataset.p.sample((5000, config.data.dim))
      mesh = torch.cat([qs, ps])
      # if device is not None:
      #   mesh = mesh.to(device)

    plt.figure(figsize=(8, 5))

    # plot data
    logr_true = dataset.log_density_ratios(mesh.to('cpu')).squeeze().numpy()

    # plot estimated ratios
    est_logr, _ = density_ratio_fn(model.to(device), mesh,
                                score_type=config.model.type)

    if config.data.dataset in ['GaussiansforMI', 'PeakedGaussians']:
      plt.scatter(mesh.squeeze().cpu().numpy(), est_logr, label='est', s=10)
      plt.scatter(mesh.squeeze().cpu().numpy(), logr_true, label='true', s=10)
    else:
      plt.hist(est_logr, bins=50, label='est', alpha=0.7)
      plt.hist(logr_true, bins=50, label='true', alpha=0.7)
    plt.legend()
    sns.despine()
    plt.tight_layout()

    if savefig is not None:
      plt.savefig(savefig + "/{}_log_ratios.png".format(step),
                  bbox_inches='tight')
      plt.close()
    else:
      plt.show()

    print('-----')
    print('true log ratios:', np.min(logr_true), np.max(logr_true))
    print('est. log ratios:', np.min(est_logr), np.max(est_logr))
    print('-----')


def estimate_mi(config, model, teacher):
  # Build density ratio estimation functions
  density_ratio_fn = density_ratios.get_toy_density_ratio_fn(
    eps=config.data.eps)

  model.eval()
  mi_true = teacher.true_mutual_info
  print('computing mutual information estimates...')

  with torch.no_grad():
    n = 5000
    samples = teacher.sample_data(n).to(device)
    emp_mi = teacher.empirical_mutual_info(samples.cpu().detach().numpy())
    est_mi, _ = density_ratio_fn(model.to(device), samples,
                                 score_type=config.model.type)
    est_mi = np.mean(est_mi)

    print('-----')
    print('true MI', mi_true)
    print('empirical MI', emp_mi)
    print('est MI', est_mi)
    print('-----')

  return est_mi


def visualize_mi(config, mi_db, mi_true, savefig=None):

  plt.figure(figsize=(12, 5))
  plt.plot(range(0, len(mi_db) * config.training.eval_freq,config.training.eval_freq),
           mi_db, '-o',label='est. MI')
  plt.hlines(mi_true, xmin=0,
             xmax=int(len(mi_db) * config.training.eval_freq),
             color='black', label='true MI')
  plt.legend(loc='lower right')
  sns.despine()
  plt.tight_layout()

  if savefig is not None:
    plt.savefig(savefig + "/mi.png", bbox_inches='tight')
    plt.close()
  else:
    plt.show()


def time_score(log_prob_fn, x, t):
  t.requires_grad_(True)
  y = log_prob_fn(x, t).sum()
  return autograd.grad(y, t, create_graph=True)[0]