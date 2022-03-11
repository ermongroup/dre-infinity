import os
import time

import numpy as np
import sacred
import pickle

import torch
from torch import nn

from sacred import Experiment, observers
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments import autils
from experiments.autils import Conv2dSameSize, LogProbWrapper
from experiments.images_data import get_data, Preprocess

from data import load_num_batches
from torchvision.utils import make_grid, save_image

from nsf.nde import distributions, transforms, flows
import nsf.nsf_utils as nsf_utils
import optim
import nn as nn_

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Capture job id on the cluster
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append('SLURM_JOB_ID')

runs_dir = os.path.join(nsf_utils.get_data_root(), 'runs/images')
ex = Experiment('decomposition-flows-images')

fso = observers.FileStorageObserver.create(runs_dir, priority=1)
# I don't like how sacred names run folders.
ex.observers.extend([fso, autils.NamingObserver(runs_dir, priority=2)])

# For num_workers > 0 and tensor datasets, bad things happen otherwise.
torch.multiprocessing.set_start_method("spawn", force=True)

# noinspection PyUnusedLocal
@ex.config
def config():
    # Dataset
    dataset = 'fashion-mnist'
    num_workers = 0
    valid_frac = 0.01

    # Pre-processing
    preprocessing = 'glow'
    alpha = .05
    num_bits = 8
    pad = 2 # For mnist-like datasets

    # Model architecture
    steps_per_level = 10
    levels = 3
    multi_scale=True
    actnorm = True

    # Coupling transform
    coupling_layer_type = 'rational_quadratic_spline'
    spline_params = {
        'num_bins': 4,
        'tail_bound': 1.,
        'min_bin_width': 1e-3,
        'min_bin_height': 1e-3,
        'min_derivative': 1e-3,
        'apply_unconditional_transform': False
    }

    # Coupling transform net
    hidden_channels = 256
    use_resnet = False
    num_res_blocks = 5 # If using resnet
    resnet_batchnorm = True
    dropout_prob = 0.

    # Optimization
    batch_size = 256
    learning_rate = 5e-4
    cosine_annealing = True
    eta_min=0.
    warmup_fraction = 0.
    num_steps = 100000
    temperatures = [0.5, 0.75, 1.]

    # Training logistics
    use_gpu = True
    multi_gpu = False
    run_descr = ''
    flow_checkpoint = None
    optimizer_checkpoint = None
    start_step = 0

    intervals = {
        'save': 1000,
        'sample': 1000,
        'eval': 1000,
        'reconstruct': 1000,
        'log': 10 # Very cheap.
    }

    # For evaluation
    num_samples = 64
    samples_per_row = 8
    num_reconstruct_batches = 10

@ex.capture
def create_transform_step(train_std,
                          num_bins, actnorm, coupling_layer_type, spline_params,
                          use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob):

    step_transforms = []
    # cholesky = transforms.AffineTransformCopula(shape=(784,))
    # try this now?
    cholesky = transforms.NaiveLinear(784, orthogonal_initialization=False)
    # cholesky = transforms.NaiveLinear(784)
    # cholesky = transforms.NaiveLinearCholesky(784)

    affine_transform1 = transforms.AffineTransform(shape=(784,))
    coupling_layer = transforms.PiecewiseRationalQuadraticCDF(
        shape=(784,),
        tails='linear',
        tail_bound=spline_params['tail_bound'],
        identity_init=True,
        num_bins=spline_params['num_bins'],
        min_bin_width=spline_params['min_bin_width'],
        min_bin_height=spline_params['min_bin_height'],
        min_derivative=spline_params['min_derivative']
    )
    affine_transform2 = transforms.AffineTransform(shape=(784,))
    # Transformv3 has no trainable parameters, only rescales the data
    rescaling = transforms.AffineTransformv3(shape=(784,), scale=1./train_std.view(-1))

    step_transforms = [rescaling, affine_transform1, coupling_layer, affine_transform2, cholesky]

    return transforms.CompositeTransform(step_transforms)


@ex.capture
# def create_transform(c, h, w,
#                      levels, hidden_channels, steps_per_level, alpha, num_bits, preprocessing,
#                      multi_scale):
# TODO(HACK): 4th and 5th lines of command line args are just to make this compatible with time-score code
# def create_transform(c, h, w,
#                      levels, hidden_channels, steps_per_level, alpha, num_bits, preprocessing,
#                      multi_scale,
#                      actnorm, coupling_layer_type, spline_params,
#                      use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob,
#                      ):
def create_transform(c, h, w, train_mean, val_mean, train_std,
                     levels, hidden_channels, steps_per_level, alpha, num_bits, preprocessing,
                     multi_scale,
                     actnorm, coupling_layer_type, spline_params,
                     use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob,
                     ):
    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    all_transforms = []

    reshape_transform = transforms.ReshapeTransform(
        input_shape=(c, h, w),
        output_shape=(c*h*w,))

    transform_level = transforms.CompositeTransform(
         [create_transform_step(train_std, spline_params['num_bins'], actnorm,
                                 coupling_layer_type, spline_params,
             use_resnet, num_res_blocks, resnet_batchnorm, dropout_prob)]
    )
    all_transforms.append(transform_level)

    mct = transforms.CompositeTransform(all_transforms)

    # Inputs to the model in [0, 2 ** num_bits]

    if preprocessing == 'glow':
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits),
                                                                shift=-0.5)
    elif preprocessing == 'tre':
        # TODO: this is to match TRE's preprocessing!
        preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - 2. * alpha)),
            transforms.Logit(),
        ])
    elif preprocessing == 'realnvp':
        preprocess_transform = transforms.CompositeTransform([
            # Map to [0,1]
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            # Map into unconstrained space as done in RealNVP
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - alpha)),
            transforms.Logit()
        ])

    elif preprocessing == 'realnvp_2alpha':
        train_preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - 2. * alpha)),
            transforms.Logit(),
            transforms.AffineScalarTransform(shift=-train_mean),
            reshape_transform
        ])
        val_preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - 2. * alpha)),
            transforms.Logit(),
            transforms.AffineScalarTransform(shift=-val_mean),
            reshape_transform
        ])
        # ok, what we're going to do is return those transformations separately
        return train_preprocess_transform, val_preprocess_transform, mct
    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    return transforms.CompositeTransform([preprocess_transform, mct])

@ex.capture
def create_flow(c, h, w,
                train_mean, val_mean, train_std,
                flow_checkpoint, _log):
    distribution = distributions.StandardNormal((c * h * w,))
    # transform = create_transform(c, h, w)
    train_transform, val_transform, transform = create_transform(c, h, w, train_mean, val_mean, train_std)

    flow = flows.Flow(transform, distribution)

    _log.info('There are {} trainable parameters in this model.'.format(
        nsf_utils.get_num_parameters(flow)))

    if flow_checkpoint is not None:
        flow.load_state_dict(torch.load(flow_checkpoint))
        _log.info('Flow state loaded from {}'.format(flow_checkpoint))

    return train_transform, val_transform, flow

@ex.capture
def train_flow(flow, train_dataset, val_dataset, dataset_dims, device, weight_decay,
               batch_size, num_steps, learning_rate, cosine_annealing, warmup_fraction,
               temperatures, num_bits, num_workers, intervals, multi_gpu, actnorm,
               optimizer_checkpoint, start_step, eta_min, _log,
               train_transform, val_transform):
    run_dir = fso.dir

    flow = flow.to(device)

    summary_writer = SummaryWriter(run_dir, max_queue=100)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers)

    if val_dataset:
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers)
    else:
        val_loader = None

    # Random batch and identity transform for reconstruction evaluation.
    random_batch, _ = next(iter(DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=0 # Faster than starting all workers just to get a single batch.
    )))
    # TODO: also edited this to add train transform
    identity_transform = transforms.CompositeTransform([
        train_transform,
        flow._transform,
        transforms.InverseTransform(flow._transform),
        transforms.InverseTransform(train_transform)
    ])

    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(torch.load(optimizer_checkpoint))
        _log.info('Optimizer state loaded from {}'.format(optimizer_checkpoint))

    if cosine_annealing:
        if warmup_fraction == 0.:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=num_steps,
                last_epoch=-1 if start_step == 0 else start_step,
                eta_min=eta_min
            )
        else:
            scheduler = optim.CosineAnnealingWarmUpLR(
                optimizer=optimizer,
                warm_up_epochs=int(warmup_fraction * num_steps),
                total_epochs=num_steps,
                last_epoch=-1 if start_step == 0 else start_step,
                eta_min=eta_min
            )
    else:
        scheduler = None

    def nats_to_bits_per_dim(x):
        c, h, w = dataset_dims
        return autils.nats_to_bits_per_dim(x, c, h, w)

    _log.info('Starting training...')

    best_val_log_prob = None
    start_time = None
    num_batches = num_steps - start_step

    # hook
    # grads?
    def get_zero_grad_hook(mask):
        def hook(grad):
            return grad * mask

        return hook

    # TODO
    mask = torch.tril(torch.ones(784, 784))
    # Register with hook
    # reversed
    # flow._transform._transforms[-1]._transforms[-1]._transforms[-1]._scale.register_hook(get_zero_grad_hook(mask))

    # not reversed
    # flow._transform._transforms[-1]._transforms[-1]._transforms[0]._scale.register_hook(get_zero_grad_hook(mask))

    # epoch_interval = 50000 // batch_size
    for step, (batch, _) in enumerate(load_num_batches(loader=train_loader,
                                                       num_batches=num_batches),
                                      start=start_step):
        # if step == 0:
        start_time = time.time() # Runtime estimate will be more accurate if set here.

        flow.train()

        optimizer.zero_grad()

        # data is uniformly dequantized, [0, 256]
        batch = batch.to(device)
        # TODO: now apply transformation and get the logdet
        batch, logabsdet = train_transform(batch)

        # TODO: modified this
        log_density = flow.log_prob(batch)
        # add in logabsdet from transformation
        log_density = log_density + logabsdet

        loss = -nats_to_bits_per_dim(torch.mean(log_density))

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            summary_writer.add_scalar('learning_rate', scheduler.get_lr()[0], step)

        summary_writer.add_scalar('loss', loss.item(), step)
        if best_val_log_prob:
            summary_writer.add_scalar('best_val_log_prob', best_val_log_prob, step)

        flow.eval() # Everything beyond this point is evaluation.

        if step % intervals['log'] == 0:
            elapsed_time = time.time() - start_time
            progress = autils.progress_string(elapsed_time, step, num_steps)
            _log.info("It: {}/{} loss: {:.3f} [{}]".format(step, num_steps, loss, progress))

        # if step % epoch_interval == 0 and step > 0:
        #     # this is the end of an "epoch"
        #     if scheduler is not None:
        #         scheduler.step()
        #         summary_writer.add_scalar('learning_rate',
        #                                   scheduler.get_lr()[0], step)

        if step % intervals['sample'] == 0:
            fig, axs = plt.subplots(1, len(temperatures), figsize=(4 * len(temperatures), 4))
            for temperature, ax in zip(temperatures, axs.flat):
                with torch.no_grad():
                    noise = flow._distribution.sample(64) * temperature
                    samples, _ = flow._transform.inverse(noise)
                    # samples, _ = flow._transform(noise)
                    # TODO: adding back in the mean shift
                    samples, _ = val_transform.inverse(samples)
                    samples = Preprocess(num_bits).inverse(samples)

                autils.imshow(make_grid(samples, nrow=8), ax)

                ax.set_title('T={:.2f}'.format(temperature))

            summary_writer.add_figure(tag='samples', figure=fig, global_step=step)

            plt.close(fig)

        if step > 0 and step % intervals['eval'] == 0 and (val_loader is not None):
            if multi_gpu:
                def log_prob_fn(batch):
                    return nn.parallel.data_parallel(LogProbWrapper(flow),
                                                     batch.to(device))
            else:
                def log_prob_fn(batch):
                    return flow.log_prob(batch.to(device))

            # val_log_prob = autils.eval_log_density(log_prob_fn=log_prob_fn,
            #                                        data_loader=val_loader)
            val_log_prob = autils.eval_log_density_transform(
                log_prob_fn=log_prob_fn,
                data_loader=val_loader,
                transform=val_transform,
                device=device)
            val_log_prob = nats_to_bits_per_dim(val_log_prob).item()

            _log.info("It: {}/{} val_log_prob: {:.3f}".format(step, num_steps, val_log_prob))
            summary_writer.add_scalar('val_log_prob', val_log_prob, step)

            if best_val_log_prob is None or val_log_prob > best_val_log_prob:
                best_val_log_prob = val_log_prob

                torch.save(flow.state_dict(), os.path.join(run_dir, 'flow_best.pt'))
                _log.info('It: {}/{} best val_log_prob improved, saved flow_best.pt'
                          .format(step, num_steps))

        if step > 0 and (step % intervals['save'] == 0 or step == (num_steps - 1)):
            torch.save(optimizer.state_dict(), os.path.join(run_dir, 'optimizer_last.pt'))
            torch.save(flow.state_dict(), os.path.join(run_dir, 'flow_last.pt'))
            _log.info('It: {}/{} saved optimizer_last.pt and flow_last.pt'.format(step, num_steps))

        if step > 0 and step % intervals['reconstruct'] == 0:
            with torch.no_grad():
                random_batch_ = random_batch.to(device)
                random_batch_rec, logabsdet = identity_transform(random_batch_)

                max_abs_diff = torch.max(torch.abs(random_batch_rec - random_batch_))
                max_logabsdet = torch.max(logabsdet)

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            autils.imshow(make_grid(Preprocess(num_bits).inverse(random_batch[:36, ...]),
                                    nrow=6), axs[0])
            autils.imshow(make_grid(Preprocess(num_bits).inverse(random_batch_rec[:36, ...]),
                                    nrow=6), axs[1])
            summary_writer.add_figure(tag='reconstr', figure=fig, global_step=step)
            plt.close(fig)

            summary_writer.add_scalar(tag='max_reconstr_abs_diff',
                                      scalar_value=max_abs_diff.item(),
                                      global_step=step)
            summary_writer.add_scalar(tag='max_reconstr_logabsdet',
                                      scalar_value=max_logabsdet.item(),
                                      global_step=step)
            # summary = dict(max_reconstr_abs_diff=max_abs_diff.item(),
            #                max_reconstr_logabsdet=max_logabsdet.item(),
            #                step=step)
            # wandb.log(summary)

@ex.capture
def set_device(use_gpu, multi_gpu, _log):
    # Decide which device to use.
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError('use_gpu is True but CUDA is not available')

    if use_gpu:
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    if multi_gpu and torch.cuda.device_count() == 1:
        raise RuntimeError('Multiple GPU training requested, but only one GPU is available.')

    if multi_gpu:
        _log.info('Using all {} GPUs available'.format(torch.cuda.device_count()))

    return device

@ex.capture
def get_train_valid_data(dataset, num_bits, valid_frac):
    return get_data(dataset, num_bits, train=True, valid_frac=valid_frac)

@ex.capture
def get_test_data(dataset, num_bits):
    return get_data(dataset, num_bits, train=False)

@ex.command
def sample_for_paper(seed):
    run_dir = fso.dir

    sample(output_path=os.path.join(run_dir, 'samples_small.png'),
           num_samples=30,
           samples_per_row=10)

    sample(output_path=os.path.join(run_dir, 'samples_big.png'),
           num_samples=100,
           samples_per_row=10,
           seed=seed + 1)


@ex.command(unobserved=True)
def eval_on_test(batch_size, num_workers, seed, _log, flow_checkpoint):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = set_device()

    # get other things (TODO: this is redundant)
    train_dataset, val_dataset, (c, h, w) = get_train_valid_data()

    # load data stats, no need to load a separate checkpoint
    pickle_path = '/'.join(flow_checkpoint.split('/')[:-1])
    # with open(os.path.join(pickle_path, 'data_means.p'), 'rb') as fp:
    #     data_stats = pickle.load(fp)
    #
    # train_mean = data_stats['train_mean']
    # train_std = data_stats['train_std']

    test_stats_path = os.path.join(pickle_path, 'test_data_means.p')
    test_dataset, (c, h, w) = get_test_data()
    train_mean, test_mean, train_std = get_data_stats(train_dataset,
                                                      test_dataset,
                                                      train=False)

    # TODO: Check logic
    if not os.path.exists(test_stats_path):
        # save these stats
        d = {
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean
        }
        with open(os.path.join(pickle_path, 'test_data_means.p'), 'wb') as fp:
            pickle.dump(d, fp)
    else:
        print('test data statistics already exist...loading!')
        with open(os.path.join(pickle_path, 'test_data_means.p'), 'rb') as fp:
            d = pickle.load(fp)

    # train_mean = train_mean
    test_mean = d['test_mean']

    # train_transform, val_transform, flow = create_flow(c, h, w)
    train_transform, test_transform, flow = create_flow(c, h, w, train_mean, test_mean, train_std)
    flow = flow.to(device)

    _log.info('Test dataset size: {}'.format(len(test_dataset)))
    _log.info('Image dimensions: {}x{}x{}'.format(c, h, w))

    flow.eval()

    def log_prob_fn(batch):
        return flow.log_prob(batch.to(device))

    test_loader=DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           num_workers=num_workers)
    test_loader = tqdm(test_loader)

    # get bpd
    def nats_to_bits_per_dim(x, c, h, w):
        return autils.nats_to_bits_per_dim(x, c, h, w)

    test_log_prob = autils.eval_log_density_transform(
        log_prob_fn=log_prob_fn,
        data_loader=test_loader,
        transform=test_transform,
        device=device)
    test_log_prob = nats_to_bits_per_dim(test_log_prob, c, h, w).item()
    print('Test log probability (bits/dim): {:.4f}'.format(test_log_prob))


# NOTE: what you just had!
# @ex.command(unobserved=True)
# def eval_on_test(batch_size, num_workers, seed, _log, flow_checkpoint):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     device = set_device()
#
#     # get other things (TODO: this is redundant)
#     train_dataset, val_dataset, (c, h, w) = get_train_valid_data()
#
#     # load data stats, no need to load a separate checkpoint
#     pickle_path = '/'.join(flow_checkpoint.split('/')[:-1])
#     with open(os.path.join(pickle_path, 'data_means.p'), 'rb') as fp:
#         data_stats = pickle.load(fp)
#
#     train_mean = data_stats['train_mean']
#     train_std = data_stats['train_std']
#
#     test_stats_path = os.path.join(pickle_path, 'test_data_means.p')
#     test_dataset, (c, h, w) = get_test_data()
#     _, test_mean, _ = get_data_stats(test_dataset, test_dataset, train=False)
#
#     # TODO: Check logic
#     if not os.path.exists(test_stats_path):
#         # save these stats
#         d = {
#             'train_mean': train_mean,
#             'train_std': train_std,
#             'test_mean': test_mean
#         }
#         with open(os.path.join(pickle_path, 'test_data_means.p'), 'wb') as fp:
#             pickle.dump(d, fp)
#     else:
#         print('test data statistics already exist...loading!')
#         with open(os.path.join(pickle_path, 'test_data_means.p'), 'rb') as fp:
#             d = pickle.load(fp)
#
#     # train_mean = train_mean
#     test_mean = d['test_mean']
#
#     # train_transform, val_transform, flow = create_flow(c, h, w)
#     train_transform, test_transform, flow = create_flow(c, h, w, train_mean, test_mean, train_std)
#     flow = flow.to(device)
#
#     _log.info('Test dataset size: {}'.format(len(test_dataset)))
#     _log.info('Image dimensions: {}x{}x{}'.format(c, h, w))
#
#     flow.eval()
#
#     def log_prob_fn(batch):
#         return flow.log_prob(batch.to(device))
#
#     test_loader=DataLoader(dataset=test_dataset,
#                            batch_size=batch_size,
#                            num_workers=num_workers)
#     test_loader = tqdm(test_loader)
#
#     # get bpd
#     def nats_to_bits_per_dim(x, c, h, w):
#         return autils.nats_to_bits_per_dim(x, c, h, w)
#
#     test_log_prob = autils.eval_log_density_transform(
#         log_prob_fn=log_prob_fn,
#         data_loader=test_loader,
#         transform=test_transform,
#         device=device)
#     test_log_prob = nats_to_bits_per_dim(test_log_prob, c, h, w).item()
#     print('Test log probability (bits/dim): {:.4f}'.format(test_log_prob))

# @ex.command(unobserved=True)
# def eval_on_test(batch_size, num_workers, seed, _log):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     device = set_device()
#     test_dataset, (c, h, w) = get_test_data()
#     _log.info('Test dataset size: {}'.format(len(test_dataset)))
#     _log.info('Image dimensions: {}x{}x{}'.format(c, h, w))
#
#     flow = create_flow(c, h, w).to(device)
#
#     flow.eval()
#
#     def log_prob_fn(batch):
#         return flow.log_prob(batch.to(device))
#
#     test_loader=DataLoader(dataset=test_dataset,
#                            batch_size=batch_size,
#                            num_workers=num_workers)
#     test_loader = tqdm(test_loader)
#
#     mean, err = autils.eval_log_density_2(log_prob_fn=log_prob_fn,
#                                           data_loader=test_loader,
#                                           c=c, h=h, w=w)
#     print('Test log probability (bits/dim): {:.2f} +/- {:.4f}'.format(mean, err))


def get_data_stats(train_dataset, val_dataset=None, train=True):
    """
    HACKY code for computing the train/val mean of the data after applying logit transform
    :param train_dataset:
    :param val_dataset:
    :param flow:
    :return:
    """
    def logit_transform(image, lambd=1e-6):
        image = lambd + (1 - 2 * lambd) * image
        return torch.log(image) - torch.log1p(-image)

    # if train:
    #     data = train_dataset.dataset.data[train_dataset.indices].unsqueeze(1)
    # else:
    #     # originally i was feeding in the test set for both train_dataset and val_dataset,
    #     # but maybe i'll try feeding in train,test this time
    #     data = train_dataset.data.unsqueeze(1)
    data = train_dataset.dataset.data[train_dataset.indices].unsqueeze(1)

    # dequantize then logit transform (data is already in [0, 255])
    data = data.float() / 256.
    data += torch.rand_like(data) / 256.
    data = logit_transform(data)
    train_mean = data.mean(0)

    tmp = data - train_mean
    train_std = tmp.std(0)

    # do the same for validation set
    if train:
        val_data = val_dataset.dataset.data[val_dataset.indices].unsqueeze(1)
    else:
        # TODO: HACK
        # val_data = train_dataset.data.unsqueeze(1)
        val_data = val_dataset.data.unsqueeze(1)
    # dequantize then logit transform
    val_data = val_data.float() / 256.
    val_data += torch.rand_like(val_data) / 256.
    val_data = logit_transform(val_data)
    val_mean = val_data.mean(0)

    return train_mean, val_mean, train_std


@ex.command(unobserved=True)
def sample(seed, num_bits, num_samples, samples_per_row, _log, output_path=None):
    torch.set_grad_enabled(False)

    if output_path is None:
        output_path = 'samples.png'

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = set_device()

    _, _, (c, h, w) = get_train_valid_data()

    flow = create_flow(c, h, w).to(device)
    flow.eval()

    preprocess = Preprocess(num_bits)

    samples = flow.sample(num_samples)
    samples = preprocess.inverse(samples)

    save_image(samples.cpu(), output_path,
               nrow=samples_per_row,
               padding=0)

@ex.command(unobserved=True)
def num_params(_log):
    _, _, (c, h, w) = get_train_valid_data()
    # c, h, w = 3, 256, 256
    create_flow(c, h, w)

@ex.command(unobserved=True)
def eval_reconstruct(num_bits, batch_size, seed, num_reconstruct_batches, _log, output_path=''):
    torch.set_grad_enabled(False)

    device = set_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset, _, (c, h, w) = get_train_valid_data()

    flow = create_flow(c, h, w).to(device)
    flow.eval()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    identity_transform = transforms.CompositeTransform([
        flow._transform,
        transforms.InverseTransform(flow._transform)
    ])

    first_batch = True
    abs_diff = []
    for batch,_ in tqdm(load_num_batches(train_loader, num_reconstruct_batches),
                        total=num_reconstruct_batches):
        batch = batch.to(device)
        batch_rec, _ = identity_transform(batch)
        abs_diff.append(torch.abs(batch_rec - batch))

        if first_batch:
            batch = Preprocess(num_bits).inverse(batch[:36, ...])
            batch_rec = Preprocess(num_bits).inverse(batch_rec[:36, ...])

            save_image(batch.cpu(), os.path.join(output_path, 'invertibility_orig.png'),
                       nrow=6,
                       padding=0)

            save_image(batch_rec.cpu(), os.path.join(output_path, 'invertibility_rec.png'),
                       nrow=6,
                       padding=0)

            first_batch = False

    abs_diff = torch.cat(abs_diff)

    print('max abs diff: {:.4f}'.format(torch.max(abs_diff).item()))


@ex.command(unobserved=True)
def profile(batch_size, num_workers):
    train_dataset, _, _ = get_train_valid_data()

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers)
    for _ in tqdm(load_num_batches(train_loader, 1000),
                  total=1000):
        pass

@ex.command(unobserved=True)
def plot_data(num_bits, num_samples, samples_per_row, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset, _, _ = get_train_valid_data()

    samples = torch.cat([train_dataset[i][0] for i in np.random.randint(0, len(train_dataset),
                                                                        num_samples)])
    samples = Preprocess(num_bits).inverse(samples)

    save_image(samples.cpu(),
               'samples.png',
               nrow=samples_per_row,
               padding=0)

@ex.automain
def main(seed, _log):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = set_device()

    # TODO: for this setup, we're going to try to get 1.40 on the test set
    # as in the TRE paper
    train_dataset, _, _ = get_train_valid_data()
    test_dataset, (c, h, w) = get_test_data()

    _log.info('Training dataset size: {}'.format(len(train_dataset)))

    if test_dataset is None:
        _log.info('No test dataset')
    else:
        _log.info('Test dataset size: {}'.format(len(test_dataset)))

    _log.info('Image dimensions: {}x{}x{}'.format(c, h, w))

    train_mean, test_mean, train_std = get_data_stats(train_dataset,
                                                      test_dataset,
                                                      train=False)

    # save these stats
    d = {
        'train_mean': train_mean,
        'test_mean': test_mean,
        'train_std': train_std
    }
    with open(os.path.join(fso.dir, 'data_means.p'), 'wb') as fp:
        pickle.dump(d, fp)
    train_mean = train_mean.to(device)
    test_mean = test_mean.to(device)
    train_std = train_std.to(device)

    # train_transform, val_transform, flow = create_flow(c, h, w)
    train_transform, test_transform, flow = create_flow(c, h, w, train_mean, test_mean, train_std)

    # train_flow(flow, train_dataset, val_dataset, (c, h, w), device)
    train_flow(flow, train_dataset, test_dataset, (c, h, w), device,
               train_transform=train_transform, val_transform=test_transform)
