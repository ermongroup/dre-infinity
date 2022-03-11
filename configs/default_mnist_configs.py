import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 130001
  training.snapshot_freq = 4000
  training.eval_freq = 100
  training.log_freq = 50
  training.ratio_freq = 4000
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.eps = 1e-4
  training.rescale_t = False
  training.rescale_method = 'default'
  training.perturb_data = False
  training.from_xscore = False
  # z-space training
  training.z_space = False
  training.invert_flow = False

  # losses
  training.joint = False
  training.algo = 'dsm'

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = False
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  # evaluate.begin_ckpt = 9
  # evaluate.end_ckpt = 26
  evaluate.begin_ckpt = 60
  evaluate.end_ckpt = 65
  # evaluate.batch_size = 1024
  evaluate.batch_size = 64
  # evaluate.enable_sampling = False
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.ais = False
  evaluate.ais_steps = 1000
  evaluate.ais_samples = 10000

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.image_size = 28
  data.random_flip = False
  data.centered = False
  data.uniform_dequantization = False
  data.logit_transform = False
  data.num_channels = 1
  data.lambda_logit = 1e-6

  # model
  config.model = model = ml_collections.ConfigDict()
  model.energy = False
  model.ema = True
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0.
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.
  optim.alpha = 0.
  optim.amsgrad = False
  optim.rescale_t = False

  config.seed = 42
  config.device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config