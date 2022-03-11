import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 200001
  training.snapshot_freq = 4000
  training.log_freq = 100
  training.eval_freq = 100
  training.ratio_freq = 4000
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.importance_weighting = False
  training.continuous = True
  training.n_jitted_steps = 10
  training.reduce_mean = False
  training.smallest_time = 1e-5
  training.regularization = 'none'
  training.reg_coeff = 1e-3
  training.eps = 1e-4
  training.rescale_t = False
  training.perturb_data = False
  # z-space training
  training.z_space = False
  training.from_xscore = False
  training.invert_flow = False
  training.iw = False
  training.IS = False
  training.interpolate = False
  training.resume_ckpt = 0

  # losses
  training.joint = False
  training.algo = 'dsm'
  training.method = 'score'

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 9
  evaluate.end_ckpt = 26
  evaluate.batch_size = 1024
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = False
  evaluate.enable_bpd = True
  evaluate.bpd_dataset = 'test'
  evaluate.ais = False
  evaluate.ais_method = 'ais'
  evaluate.ais_steps = 1000
  evaluate.ais_samples = 10000

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.image_size = 28
  data.random_flip = False
  data.horizontal_flip = False
  data.centered = False
  data.uniform_dequantization = False
  data.logit_transform = False
  data.num_channels = 1
  data.lambda_logit = 1e-6

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.trainable_embedding = True
  model.data_init = False
  # TRE-specific
  model.head_type = 'linear'
  model.infinite = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0.
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.
  optim.alpha = 0.
  optim.rescale_t = False
  optim.amsgrad = False
  optim.manager = 'v1'

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config