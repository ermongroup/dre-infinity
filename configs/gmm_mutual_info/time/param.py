from configs.default_toy_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.joint = False
  training.n_iters = 20000
  training.batch_size = 512
  training.reweight = False

  # data
  data = config.data
  data.dataset = 'GaussiansforMI'
  data.eps = 1e-5
  data.dim = 40  # [40, 80, 160, 320]
  data.rho = 0.8
  data.sigmas = [0.001, 1.]
  data.centered = False

  # model
  model = config.model
  model.type = 'time'
  model.param = True
  model.name = 'toy_param_mvn_mi'
  model.nf = 64
  model.z_dim = 128

  # optimizer
  optim = config.optim
  optim.lr = 1e-3

  return config
