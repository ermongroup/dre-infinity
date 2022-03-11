import os
import sys
import yaml
import logging
from models.ncsn_unet import (
  SinusoidalPosEmb,
  GaussianFourierProjection,
  Dense
)

# need to import things from top-level directory for pretrained flow models
top_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(top_path, 'nsf'))
from nsf.nde import distributions, transforms, flows

sys.path.append(os.path.join(top_path, 'mintnet'))
from mintnet.models.cnn_flow import DataParallelWithSampling
from mintnet.models.cnn_flow import Net
from mintnet.models.nice import NICE
import argparse

from . import utils
import torch
import torch.nn as nn


def get_act(config):
  """Get activation functions from the config file."""

  if config.model.nonlinearity.lower() == 'elu':
    return nn.ELU()
  elif config.model.nonlinearity.lower() == 'relu':
    return nn.ReLU()
  elif config.model.nonlinearity.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif config.model.nonlinearity.lower() == 'swish':
    return nn.SiLU()
  elif config.model.nonlinearity.lower() == 'tanh':
    return nn.Tanh()
  else:
    raise NotImplementedError('activation function does not exist!')


def dict2namespace(config):
  namespace = argparse.Namespace()
  for key, value in config.items():
    if isinstance(value, dict):
      new_value = dict2namespace(value)
    else:
      new_value = value
    setattr(namespace, key, new_value)
  return namespace


# NOTE: these are all trained on MNIST
def load_pretrained_flow(config, test=False):
  name = config.training.z_space_model
  print('loading flow model: {}'.format(name))
  config_path = os.path.join(top_path, 'mintnet', 'configs')
  if name in ['mintnet', 'nice']:
    if name == 'mintnet':
      config_path = os.path.join(config_path, 'mnist_mintnet.yml')
      model_cls = Net
      ckpt_path = os.path.join(top_path, 'flow_ckpts', 'mintnet_checkpoint.pth')
    else:
      config_path = os.path.join(config_path, 'mnist_nice.yml')
      model_cls = NICE
      ckpt_path = os.path.join(top_path, 'flow_ckpts', 'bs32_nice_checkpoint.pth')
    print('loading model from checkpoint: {}'.format(ckpt_path))
    logging.info('loading model from checkpoint: {}'.format(ckpt_path))
    with open(os.path.join('configs', config_path), 'r') as f:
      config = yaml.load(f)
    new_config = dict2namespace(config)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
      'cpu')
    new_config.device = device

    # load pretrained model
    net = model_cls(new_config).to(new_config.device)
    net = DataParallelWithSampling(net)

    # doesn't seem to have been trained with EMA
    # if new_config.training.ema:
    #   ema_helper = EMAHelper(mu=0.999)
    #   ema_helper.register(net)

    # load checkpoint
    states = torch.load(ckpt_path, map_location=new_config.device)
    net.load_state_dict(states[0])
  elif name == 'rq_nsf':
    ckpt_path = os.path.join(top_path, 'flow_ckpts', 'rq_nsf_best.pt')
    print('loading model from checkpoint: {}'.format(ckpt_path))

    # annoying data transforms
    c = 1
    h = w = 28
    spline_params = {
    "apply_unconditional_transform": False,
    "min_bin_height": 0.001,
    "min_bin_width": 0.001,
    "min_derivative": 0.001,
    "num_bins": 8,
    "tail_bound": 3.0
    }
    distribution = distributions.StandardNormal((c * h * w,))
    # TODO (HACK): get rid of hardcoding
    from nsf.experiments.images import create_transform
    transform = create_transform(c, h, w,
                                 levels=2, hidden_channels=64, steps_per_level=8, alpha=0.000001,
                                 num_bits=8, preprocessing="realnvp_2alpha", multi_scale=False,
                                 actnorm=True, coupling_layer_type="rational_quadratic_spline",
                                 spline_params=spline_params,
                                 use_resnet=False, num_res_blocks=2, resnet_batchnorm=False, dropout_prob=0.0)

    net = flows.Flow(transform, distribution)

    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint)
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
  elif name == 'rq_nsf_copula':
    import pickle
    from nsf.experiments.images_centering_copula import create_transform
    ckpt_path = os.path.join(top_path, 'flow_ckpts', 'copula_best.pt')
    print('loading model from checkpoint: {}'.format(ckpt_path))

    # load data stats, no need to load a separate checkpoint
    if not test:
      with open(os.path.join(top_path, 'flow_ckpts', 'data_means.p'), 'rb') as fp:
          data_stats = pickle.load(fp)
      val_mean = data_stats['val_mean']
    else:  # val_mean = test_mean here
      print('loading test statistics for flow evaluation!')
      with open(os.path.join(top_path, 'flow_ckpts', 'test_data_means.p'), 'rb') as fp:
          data_stats = pickle.load(fp)
      val_mean = data_stats['test_mean']

    # stats
    train_mean = data_stats['train_mean']
    train_std = data_stats['train_std']

    # doesn't matter, can just get it from the data_stats object
    # from torchvision.datasets import MNIST
    # from datasets import logit_transform
    # import torchvision
    # data_dir = '/atlas/u/kechoi/time-score-dre/'
    # test_transform = torchvision.transforms.Compose([
    #   torchvision.transforms.Resize(config.data.image_size),
    #   torchvision.transforms.ToTensor()
    # ])
    # dataset = MNIST(os.path.join(data_dir, 'datasets', 'mnist_test'),
    #                 train=False,
    #                 download=True,
    #                 transform=test_transform)
    # data = dataset.data.unsqueeze(1).float()
    # # dequantize
    # data = (data + torch.rand_like(data)) / 256.
    # data = logit_transform(data)
    # val_mean = data.mean(0)  # lol will this make a diff? (no)

    # annoying data transforms
    c = 1
    h = w = 28
    spline_params = {
    "apply_unconditional_transform": False,
    "min_bin_height": 0.001,
    "min_bin_width": 0.001,
    "min_derivative": 0.001,
    "num_bins": 128,
    "tail_bound": 3.0
    }
    distribution = distributions.StandardNormal((c * h * w,))

    train_transform, val_transform, transform = create_transform(
      c, h, w, train_mean, val_mean, train_std, levels=2, hidden_channels=64,
      steps_per_level=8, alpha=0.000001, num_bits=8, preprocessing="realnvp_2alpha",
      multi_scale=False, actnorm=True, coupling_layer_type="rational_quadratic_spline",
      spline_params=spline_params, use_resnet=False, num_res_blocks=2,
      resnet_batchnorm=False, dropout_prob=0.0)

    # net = flows.Flow(transform, distribution)
    net = flows.FlowDataTransform(transform, distribution, train_transform, val_transform)

    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    # net.load_state_dict(checkpoint)

    # TODO: this is only for testing purposes! will go away if you train again
    xs = {'_train_transform.'+k: v for k, v in train_transform.state_dict().items()}
    ys = {'_val_transform.'+k: v for k, v in val_transform.state_dict().items()}
    new_state_dict = {**checkpoint, **xs, **ys}
    net.load_state_dict(new_state_dict)

    net = net.to('cuda')
    net = torch.nn.DataParallel(net)

  elif name == 'rq_nsf_noise':
    import pickle
    from nsf.experiments.images_noise import create_transform

    # load data stats, no need to load a separate checkpoint
    with open(os.path.join(top_path, 'flow_ckpts', 'test_data_stats.p'), 'rb') as fp:
        data_stats = pickle.load(fp)

    train_mean = data_stats['train_mean'].to('cuda')
    test_mean = data_stats['test_mean'].to('cuda')
    val_mean = data_stats['val_mean'].to('cuda')
    train_cov = data_stats['train_cov_cholesky'].to('cuda')
    val_cov = data_stats['val_cov_cholesky'].to('cuda')

    # annoying data transforms
    c = 1
    h = w = 28
    spline_params = {
    "apply_unconditional_transform": False,
    "min_bin_height": 0.001,
    "min_bin_width": 0.001,
    "min_derivative": 0.001,
    "num_bins": 128,
    "tail_bound": 3.0
    }
    distribution = distributions.StandardNormal((c * h * w,))
    # TODO (HACK): get rid of hardcoding

    train_transform, val_transform, transform = create_transform(
      c, h, w, train_mean, val_mean, train_cov, val_cov, levels=2, hidden_channels=64,
      steps_per_level=8, alpha=0.000001, num_bits=8, preprocessing="realnvp_2alpha",
      multi_scale=False, actnorm=True, coupling_layer_type="rational_quadratic_spline",
      spline_params=spline_params, use_resnet=False, num_res_blocks=2,
      resnet_batchnorm=False, dropout_prob=0.0)

    # map this "flow" onto the device
    # net = flows.Flow(transform, distribution)
    net = flows.FlowDataTransform(transform, distribution, train_transform, val_transform)

    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
  else:
    raise NotImplementedError

  return net


@utils.register_model(name='ncsn_mlp')
class NCSNMLP(nn.Module):
  """
  Simple MLP-based score network. This model is intended for
  toy Gaussian problems and pre-encoded data (e.g. via a flow).
  This does not use any special positional encoding.
  """

  def __init__(self, config, embed_dim=256):
    super().__init__()
    self.config = config
    # because you flatten your inputs
    self.in_dim = (config.data.image_size * config.data.image_size)
    self.h_dim = config.model.h_dim
    self.act = get_act(config)

    # build mlp
    self.net = [nn.Linear(self.in_dim + 1, self.h_dim), self.act]
    for _ in range(config.model.n_hidden_layers):
      self.net.append(nn.Linear(self.h_dim, self.h_dim))
      self.net.append(self.act)
    self.net.append(nn.Linear(self.h_dim, 1))
    self.net = nn.Sequential(*self.net)

  def forward(self, x, t):
    xt = torch.cat([x, t.unsqueeze(-1)], dim=-1)
    h = self.net(xt)
    return h


@utils.register_model(name='ncsn_mlpv2')
class NCSNMLPv2(nn.Module):
  """
  Simple MLP-based score network. This model is intended for
  toy Gaussian problems and pre-encoded data (e.g. via a flow).

  This uses the DDPM sinusoidal positional embedding.
  """

  def __init__(self, config, embed_dim=256):
    super().__init__()
    self.config = config

    # because you flatten your inputs
    self.in_dim = (config.data.image_size * config.data.image_size)
    self.h_dim = config.model.h_dim
    self.act = get_act(config)
    self.temb_act = lambda x: x * torch.sigmoid(x)

    self.embed = nn.Sequential(
        SinusoidalPosEmb(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim)
    )

    # build mlp
    self.net = [nn.Linear(self.in_dim, self.h_dim)]
    for _ in range(config.model.n_hidden_layers):
      self.net.append(nn.Linear(self.h_dim, self.h_dim))
    self.net = nn.ModuleList(self.net)

    self.dense = nn.ModuleList([Dense(self.h_dim, self.h_dim) for _ in range(config.model.n_hidden_layers+1)])

    self.final_fc = nn.Linear(self.h_dim, 1)

  def forward(self, x, t):
    t = t.squeeze()
    embed = self.temb_act(self.embed(t))

    for layer, dense in zip(self.net, self.dense):
      x = layer(x) + dense(embed).squeeze()  # TODO(HACK): bc no convolutions here
      x = self.act(x)
    x = self.final_fc(x)

    return x


# TODO
@utils.register_model(name='ncsn_mlp_xt')
class NCSNMLP_xt(nn.Module):
  """
  Simple MLP-based score network. This model is intended for
  toy Gaussian problems and pre-encoded data (e.g. via a flow)
  """

  def __init__(self, config, embed_dim=256):
    super().__init__()
    self.config = config
    # because you flatten your inputs
    self.in_dim = (config.data.image_size * config.data.image_size)
    self.h_dim = config.model.h_dim
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
    xt = torch.cat([x, t.unsqueeze(-1)], dim=-1)
    h = self.net(xt)
    return h