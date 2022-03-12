# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
import copy

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
# from models import ddpm, ncsnv2, ncsnpp
from models import ncsn_unet
# from models import ncsnpp
from models import ncsn_flow
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from evaluations import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import wandb
import density_ratios

FLAGS = flags.FLAGS


# first, load some flow-specific code


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(),
                                 decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, config.training.n_iters // config.training.snapshot_freq,
    eta_min=0, last_epoch=-1, verbose=False)
  optimize_fn = losses.optimization_manager(config)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0,
               scheduler=scheduler)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta",
                                     "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  if not config.training.z_space:
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization)
  else:
    logging.info('Loading MNIST dataset to be encoded using the flow!')
    train_ds, eval_ds = datasets.get_dataset_for_flow(
      config,
      uniform_dequantization=config.data.uniform_dequantization)
  # Create data normalizer and its inverse
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # load pre-trained normalizing flow checkpoint
  if config.training.z_space:
    logging.info('Loading pre-trained flow checkpoint...')
    flow = ncsn_flow.load_pretrained_flow(config)
    flow.eval()  # no training
  else:
    flow = None

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                        beta_max=config.model.beta_max,
                        N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'z_vpsde':
    # TODO: check if we need to feed in the flow
    assert flow is not None
    sde = sde_lib.Z_VPSDE(flow, beta_min=config.model.beta_min,
                        beta_max=config.model.beta_max,
                        N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                           beta_max=config.model.beta_max,
                           N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                        sigma_max=config.model.sigma_max,
                        N=config.model.num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'interpxt':
    sde = sde_lib.InterpXt(t_min=config.training.eps, t_max=1.,
                           N=config.model.num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'flow_interpxt':
    assert config.training.z_space
    assert config.training.invert_flow
    assert flow is not None
    sde = sde_lib.FlowInterpXt(flow, t_min=config.training.eps, t_max=1.,
                               N=config.model.num_scales)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  train_eps = config.training.eps

  # get appropriate loss function
  if config.training.sde.lower() in ['interpxt', 'flow_interpxt']:
    logging.info(
      'using loss function from special linear interpolation scheme!')
    from interp_losses import get_step_fn
  else:
    from losses import get_step_fn

  # Build one-step training and evaluation functions
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  joint = config.training.joint
  alpha = config.optim.alpha
  algo = config.training.algo
  perturb = config.training.perturb_data
  invert_flow = config.training.invert_flow
  z_space = config.training.z_space
  centered = config.data.centered  # [-1, 1]
  z_interpolate = config.training.z_interpolate
  mlp = True if 'mlp' in config.model.name else False
  if perturb:
    print('perturbing MNIST data with a small amount of Gaussian noise!')
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = get_step_fn(sde, train=True, algo=algo, joint=joint,
                              z_space=z_space, mlp=mlp, alpha=alpha,
                              z_interpolate=z_interpolate, 
                              optimize_fn=optimize_fn, eps=train_eps,
                              reduce_mean=reduce_mean, continuous=continuous,
                              likelihood_weighting=likelihood_weighting,
                              flow=flow)
  eval_step_fn = get_step_fn(sde, train=False, algo=algo, joint=joint,
                             z_space=z_space, mlp=mlp, alpha=alpha,
                             z_interpolate=z_interpolate,
                             optimize_fn=optimize_fn, eps=train_eps,
                             reduce_mean=reduce_mean, continuous=continuous,
                             likelihood_weighting=likelihood_weighting,
                             flow=flow)
  likelihood_fn = likelihood.get_likelihood_fn_flow(sde, inverse_scaler)
  if config.training.algo != 'baseline':
    if config.training.sde.lower() in ['interpxt', 'flow_interpxt']:
      if not config.training.z_space:
        density_ratio_fn = density_ratios.get_interp_density_ratio_fn(sde,
                                                                      inverse_scaler)
      else:
        density_ratio_fn = density_ratios.get_interp_density_ratio_fn_flow(sde,
                                                                           inverse_scaler)
    else:
      if not config.training.z_space:
        density_ratio_fn = density_ratios.get_density_ratio_fn(sde,
                                                               inverse_scaler,
                                                               eps=train_eps)
      else:
        if z_interpolate:
          density_ratio_fn = density_ratios.get_z_interp_density_ratio_fn_flow(
            sde,
            inverse_scaler,
            mlp=mlp)
        else:
          density_ratio_fn = density_ratios.get_density_ratio_fn_flow(sde,
                                                                      inverse_scaler,
                                                                      eps=train_eps)
  if config.training.dre_bpd_v2:
    if config.training.sde.lower() in ['interpxt', 'flow_interpxt']:
      density_ratio_fn_v2 = density_ratios.get_interp_v2_density_ratio_fn(sde,
                                                                          inverse_scaler,
                                                                          eps=train_eps)
    else:
      density_ratio_fn_v2 = density_ratios.get_v2_density_ratio_fn(sde,
                                                                   inverse_scaler,
                                                                   eps=train_eps)
  if config.training.from_xscore:
    estimated_density_ratio_fn = density_ratios.get_density_ratios_from_data_scores(
      sde, inverse_scaler, eps=train_eps, rtol=1e-3,
      atol=1e-6)  # default values

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape,
                                           inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  logging.info("Using model type %s." % config.model.name)
  if joint:
    logging.info("Using alpha %.3e for joint training" % (alpha))
  if config.training.rescale_t:
    logging.info('rescaling output of time score network!')

  for step in range(initial_step, num_train_steps + 1):
    if not config.training.z_space:
      # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
      batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(
        config.device).float()
      batch = batch.permute(0, 3, 1, 2)
      batch = scaler(batch)
    else:  # only pytorch, this is for training that uses a flow
      try:
        batch, _ = next(train_iter)  # ignore labels
      except StopIteration:
        train_iter = iter(train_ds)
        batch, _ = next(train_iter)
      batch = batch.to(config.device).float()

      # add uniform noise, then rescale to [-1, +1]
      # NOTE: should flip the order for adding gaussian noise
      # TODO: you have a separate file for this atm
      if 'rq_nsf' in config.model.name:
        # for this flow, it assumes that data is uniformly dequantized BUT
        # still between [0, 256]
        batch = batch * 255.
        batch += torch.rand_like(batch)
      else:
        batch = batch * 255. / 256.
        batch += torch.rand_like(batch) / 256.
      import pdb
      pdb.set_trace()

      if invert_flow or z_interpolate:  # p(x) = flow trained on MNIST
        # rescale to [-1, 1]
        batch = scaler(batch)
      else:
        # in this case, q(x) will be transformed to N(0,I) via the flow
        batch = datasets.logit_transform(batch, config.data.lambda_logit)
        with torch.no_grad():
          batch, _ = flow(batch)

        # since we're feeding it into an MLP, reshape
        if 'mlp' in config.model.name:
          batch = batch.view(batch.size(0), -1)
        else:
          batch = batch.view(batch.size(0),
                             config.data.num_channels,
                             config.data.image_size, config.data.image_size)

    # Execute one training step
    # loss = train_step_fn(state, batch)
    summary = train_step_fn(state, batch.detach())
    summary['step'] = step
    wandb.log(summary)
    if step % config.training.log_freq == 0:
      logging.info(
        "step: %d, training_loss: %.5e" % (step, summary['loss']))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      if not config.training.z_space:
        eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(
          config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
      else:
        try:
          eval_batch, _ = next(eval_iter)
        except StopIteration:
          eval_iter = iter(eval_ds)
          eval_batch, _ = next(eval_iter)
        eval_batch = eval_batch.to(config.device).float()

        # uniform dequantization then [-1, 1] rescaling
        if 'rq_nsf' in config.model.name:
          # for this flow, it assumes that data is uniformly dequantized BUT
          # still between [0, 256]
          eval_batch = eval_batch * 255.
          eval_batch += torch.rand_like(eval_batch)
        else:
          eval_batch = eval_batch * 255. / 256.
          eval_batch += torch.rand_like(eval_batch) / 256.

        if invert_flow or z_interpolate:  # p(x) = flow
          eval_batch = scaler(eval_batch)
          log_det_logit = torch.zeros(len(eval_batch), device=config.device)
          flow_log_det = torch.zeros_like(log_det_logit)
        else:
          # need to do data transformation with the flow
          eval_batch = datasets.logit_transform(eval_batch, config)

          # get log-det-logit
          log_det_logit = F.softplus(-eval_batch).sum() + F.softplus(
            eval_batch).sum() + np.prod(
            eval_batch.shape) * np.log(1 - 2 * config.data.lambda_logit)

          # run through flow
          with torch.no_grad():
            eval_batch, flow_log_det = flow(eval_batch)

          # note: we aren't rescaling to [-1, 1] when using the flow
          # since we're feeding it into an MLP, reshape
          if 'mlp' in config.model.name:
            eval_batch = eval_batch.view(eval_batch.size(0), -1)
          else:
            eval_batch = eval_batch.view(eval_batch.size(0),
                               config.data.num_channels,
                               config.data.image_size, config.data.image_size)

      # NOTE: no additional dequantization on z embeddings!
      dre_eval_batch = copy.copy(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      summary = dict(
        test_loss=eval_loss['loss'],
        step=step
      )
      logging.info(
        "step: %d, eval_loss: %.5e" % (step, eval_loss['loss']))

      # only compute density ratios when network is sufficiently smooth
      if step > 100 and step % config.training.ratio_freq == 0:
        if config.eval.enable_bpd:
          # use EMA for ratio computation
          ema.store(score_model.parameters())
          ema.copy_to(score_model.parameters())

          # different types of density ratios for energy-based modeling
          if config.training.pf_ode_bpd:
            bpd = likelihood_fn(score_model, dre_eval_batch, flow_log_det, log_det_logit)[0]
            # bpd = bpd.detach().cpu().numpy().reshape(-1)
            # summary['test_bpds'] = bpd.mean()
            summary['test_bpds'] = bpd.item()
            logging.info("step: %d, eval_bpd: %.5f" % (step, bpd.mean()))
          if config.training.dre_bpd:
            dre_bpd = \
                density_ratio_fn(score_model=score_model, flow=flow, x=dre_eval_batch)[0]
            # dre_bpd = dre_bpd.reshape(-1)
            # summary['test_dre_bpds'] = dre_bpd.mean()
            summary[
              'test_dre_bpds'] = dre_bpd.item()  # TODO: changed this to sum
            logging.info(
              "step: %d, eval_dre_bpd: %.5f" % (step, dre_bpd.mean()))
          if config.training.dre_bpd_v2:
            raise NotImplementedError
            dre_bpd_v2 = density_ratio_fn_v2(score_model, dre_eval_batch)[0]
            dre_bpd_v2 = dre_bpd_v2.reshape(-1)
            summary['test_dre_bpds_v2'] = dre_bpd_v2.mean()
            logging.info(
              "step: %d, eval_dre_bpd_v2: %.5f" % (step, dre_bpd_v2.mean()))
          if config.training.from_xscore:
            raise NotImplementedError
            est_bpd = estimated_density_ratio_fn(score_model, dre_eval_batch)[0]
            est_bpd = est_bpd.reshape(-1)
            summary['est_bpds'] = est_bpd.mean()
            logging.info("step: %d, eval_bpd via fokker-planck: %.5f" % (
            step, est_bpd.mean()))

          ema.restore(score_model.parameters())

      wandb.log(summary)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(
        os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # update optimizer scheduler
      scheduler.step()
      print('learning rate is now: {}'.format(scheduler._last_lr))

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        # log generations to wandb
        wandb.log({"samples": [wandb.Image(i) for i in sample[0:64]]})
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0,
                         255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # load pre-trained normalizing flow checkpoint
  if config.training.z_space:
    logging.info('Loading pre-trained flow checkpoint...')
    flow = ncsn_flow.load_pretrained_flow(config)
    flow.eval()  # no training
  else:
    flow = None

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(),
                                 decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                        beta_max=config.model.beta_max,
                        N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'z_vpsde':
    assert flow is not None
    sde = sde_lib.Z_VPSDE(flow, beta_min=config.model.beta_min,
                        beta_max=config.model.beta_max,
                        N=config.model.num_scales)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting,
                                   flow=flow)

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True,
                                                      evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    # bpd_num_repeats = 5
    bpd_num_repeats = 1  # let's just do once for now lol
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn_flow(sde, inverse_scaler)
    if config.training.algo != 'baseline':
      if config.training.sde.lower() in ['interpxt', 'flow_interpxt']:
        if not config.training.z_space:
          density_ratio_fn = density_ratios.get_interp_density_ratio_fn(sde,
                                                                        inverse_scaler)
        else:
          density_ratio_fn = density_ratios.get_interp_density_ratio_fn_flow(sde,
                                                                        inverse_scaler)
      else:
        if not config.eval.ais:
          density_ratio_fn = density_ratios.get_z_interp_density_ratio_fn_flow(
            sde,
            inverse_scaler)
        else:
          fancy = config.eval.ais_fancy_prior
          n_ais_steps = config.eval.ais_steps
          n_ais_samples = config.eval.ais_samples
          assert config.training.z_interpolate
          print('using AIS for bpd computation!')
          density_ratio_fn = density_ratios.get_density_ratio_fn_ais(sde,
                                                                     inverse_scaler,
                                                                     n_ais_steps,
                                                                     n_ais_samples,
                                                                     fancy=fancy)
    else:
      # this is a vanilla baseline
      print('WARNING: WE ARE NOT USING THE PROBABILITY FLOW ODE!!!!')
      print('We are estimating the time scores via the fokker-planck equation!')
      print('relaxing rtol and atol from 1e-5 to 1e-3 for speed...')
      density_ratio_fn = density_ratios.get_density_ratios_from_data_scores(sde,
                                                                            inverse_scaler,
                                                                            rtol=1e-3,
                                                                            atol=1e-3)
    if config.training.dre_bpd_v2:
      if config.training.sde.lower() in ['interpxt', 'flow_interpxt']:
        density_ratio_fn_v2 = density_ratios.get_interp_v2_density_ratio_fn(sde,
                                                                            inverse_scaler)
      else:
        density_ratio_fn_v2 = density_ratios.get_v2_density_ratio_fn(sde,
                                                                     inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape,
                                           inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir,
                                 "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(
          config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"),
                             "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses,
                            mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    print('starting density ratio estimation')
    if config.eval.enable_bpd:
      total_bpd = 0
      total_n_data = 0
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(
            config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = density_ratio_fn(flow=flow, score_model=score_model, x=eval_batch)[0]
          # NOTE: we've converted bpds from a list to average bpd per batch
          total_bpd += bpd.item() * eval_batch.shape[0]
          total_n_data += eval_batch.shape[0]
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (
            ckpt, repeat, batch_id, total_bpd / total_n_data))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          if config.eval.ais:
            fname = f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"
          else:
            fname = f"vanilla_{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"
          with tf.io.gfile.GFile(os.path.join(eval_dir, fname), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, total_bpd, total_n_data)
            fout.write(io_buffer.getvalue())

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0,
                          255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size,
           config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(
        os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[
                     :config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())
