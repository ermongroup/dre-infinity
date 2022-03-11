from configs.deeper_default_mnist_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.batch_size = 64
  training.sde = 'z_vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.eps = 1e-4
  training.snapshot_sampling = False
  training.from_xscore = False
  training.z_space = True
  training.z_space_model = 'rq_nsf'
  training.z_interpolate = True
  training.iw = False
  training.buffer_size = 10
  training.interpolate = False  # refers to time interpolation
  training.resume_ckpt = 0
  training.n_iters = 750001
  training.snapshot_freq = 2000
  training.ratio_freq = 2000

  # losses
  training.joint = False
  training.algo = 'dsm'
  training.pf_ode_bpd = False
  training.dre_bpd = True  # baseline
  training.dre_bpd_v2 = False  # testing

  # eval
  eval = config.eval
  eval.batch_size = 1000
  eval.begin_ckpt = 95
  eval.end_ckpt = 97
  eval.ais = False
  eval.ais_method = 'ais'
  eval.ais_fancy_prior = False
  eval.ais_steps = 10

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnunet_t'
  model.n_hidden_layers = 2
  model.h_dim = 256
  model.nonlinearity = 'swish'
  model.embedding_type = 'linear'
  model.embedding_scale = False
  # others
  model.scale_by_sigma = False
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nf = 32  # 64
  model.ch_mult = (1, 2, 2)
  # this configuration has 10 resblocks total
  # model.ch_mult = (1, 2)
  model.num_res_blocks = 2 # 4
  model.attn_resolutions = (7,)  # (14,) maybe start with this
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # optimization
  optim = config.optim
  optim.amsgrad = False
  optim.alpha = 0.
  optim.manager = 'v1'

  return config