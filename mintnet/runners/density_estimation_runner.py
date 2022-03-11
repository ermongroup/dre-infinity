import getpass

from models.cnn_flow import Net
from models.nice import NICE
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import shutil
# import tensorboardX
import getpass
import wandb
import logging
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from models.cnn_flow import DataParallelWithSampling
from torchvision.utils import save_image, make_grid
import torch.autograd as autograd
import torch
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import math
import pickle
from models.utils import EMAHelper
sns.set()


class DensityEstimationRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.sample_dir = config.sample_dir

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad,
                              eps=self.config.optim.adam_eps)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        elif self.config.optim.optimizer == 'Adamax':
            return optim.Adamax(parameters, lr=self.config.optim.lr, betas=(self.config.optim.beta1, 0.999),
                                weight_decay=self.config.optim.weight_decay)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def get_model(self):
        if self.config.model.name == 'mintnet':
            model_cls = Net
        elif self.config.model.name == 'nice':
            model_cls = NICE
        else:
            raise NotImplementedError
        return model_cls

    def logit_transform(self, image):
        lambd = self.config.data.lambda_logit
        image = lambd + (1 - 2 * lambd) * image
        return torch.log(image) - torch.log1p(-image)

    def sigmoid_transform(self, samples):
        lambd = self.config.data.lambda_logit
        samples = torch.sigmoid(samples)
        samples = (samples - lambd) / (1 - 2 * lambd)
        return samples

    def train(self):
        if self.config.data.horizontal_flip:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])

        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        test_transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'MNIST':
            userid = getpass.getuser()
            dataset = MNIST(os.path.join('/atlas/u/{}/datasets'.format(userid), 'mnist'),
                            train=True, download=True,
                            transform=train_transform)
            test_dataset = MNIST(os.path.join('/atlas/u/{}/datasets'.format(userid), 'mnist_test'),
                                 train=False, download=True,
                                 transform=test_transform)
        else:
            raise NotImplementedError
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4,
                                drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)
        test_iter = iter(test_loader)

        model_cls = self.get_model()
        net = model_cls(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        if self.config.training.ema:
            ema_helper = EMAHelper(mu=0.999)
            ema_helper.register(net)

        optimizer = self.get_optimizer(net.parameters())
        # tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        # if os.path.exists(tb_path):
        #     shutil.rmtree(tb_path)
        # tb_logger = tensorboardX.SummaryWriter(logdir=tb_path)

        def flow_loss(u, log_jacob, size_average=True):
            log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
            log_jacob = log_jacob.sum()
            loss = -(log_probs + log_jacob)

            if size_average:
                loss /= u.size(0)
            return loss

        if self.config.data.dataset == 'MNIST':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 350, 450], gamma=0.1) #original
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.training.n_epochs, eta_min=0.)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'),
                                map_location=self.config.device)

            net.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            begin_epoch = states[2]
            step = states[3]
            scheduler.load_state_dict(states[4])
            if self.config.training.ema:
                ema_helper.load_state_dict(states[5])
        else:
            step = 0
            begin_epoch = 0

        # Train the model

        for epoch in range(begin_epoch, self.config.training.n_epochs):

            for batch_idx, (data, _) in enumerate(dataloader):
                net.train()
                # Transform to logit space since pixel values ranging from 0-1
                data = data.to(self.config.device) * 255. / 256.
                data += torch.rand_like(data) / 256.
                if self.config.data.logit_transform:
                    data = self.logit_transform(data)
                    log_det_logit = F.softplus(-data).sum() + F.softplus(data).sum() + np.prod(
                        data.shape) * np.log(1 - 2 * self.config.data.lambda_logit)
                else:
                    log_det_logit = 0.

                output, log_det = net(data)
                loss = flow_loss(output, log_det)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.config.training.ema:
                    ema_helper.update(net)

                bpd = (loss.item() * data.shape[0] - log_det_logit) / (np.log(2) * np.prod(data.shape)) + 8

                # validation
                # Do EMA
                if step % self.config.training.log_interval == 0:
                    if self.config.training.ema:
                        net_test = ema_helper.ema_copy(net)
                    else:
                        net_test = net
                    net_test.eval()
                    with torch.no_grad():
                        try:
                            test_data, _ = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            test_data, _ = next(test_iter)

                        test_data = test_data.to(self.config.device) * 255. / 256.
                        test_data += torch.rand_like(test_data) / 256.
                        if self.config.data.logit_transform:
                            test_data = self.logit_transform(test_data)

                            test_log_det_logit = F.softplus(
                                -test_data).sum() + F.softplus(
                                test_data).sum() + np.prod(
                                test_data.shape) * np.log(
                                1 - 2 * self.config.data.lambda_logit)
                        else:
                            test_log_det_logit = 0.

                        test_output, test_log_det = net_test(test_data)
                        test_loss = flow_loss(test_output, test_log_det)
                        test_bpd = (test_loss.item() * test_data.shape[0] - test_log_det_logit) * (
                                1 / (np.log(2) * np.prod(test_data.shape))) + 8

                    summary = dict(
                        training_loss=loss,
                        training_bpd=bpd,
                        test_loss=test_loss,
                        test_bpd=test_bpd,
                        global_step=step
                    )
                    wandb.log(summary)
                    logging.info(
                        "epoch: {}, batch: {}, training_loss: {}, test_loss: {}, training_bpd: {}, test_bpd: {}".format(
                            epoch, batch_idx, loss.item(), test_loss.item(), bpd.item(), test_bpd.item()))
                step += 1

                if self.config.data.dataset == 'ImageNet':
                    scheduler.step()
                    if step % self.config.training.snapshot_interval == 0:
                        states = [
                            net.state_dict(),
                            optimizer.state_dict(),
                            epoch + 1,
                            step,
                            scheduler.state_dict(),
                        ]
                        if self.config.training.ema:
                            states.append(ema_helper.state_dict())
                        torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                        'checkpoint_batch_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))

                    if step == self.config.training.maximum_steps:
                        states = [
                            net.state_dict(),
                            optimizer.state_dict(),
                            epoch + 1,
                            step,
                            scheduler.state_dict(),
                        ]
                        if self.config.training.ema:
                            states.append(ema_helper.state_dict())
                        torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                        'checkpoint_last_batch.pth'))
                        torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))

                        return 0

            if self.config.data.dataset != 'ImageNet':
                scheduler.step()

                # also do a set of sampling
                net_test.eval()
                logging.info("Generating samples")
                z = torch.randn(64,
                                self.config.data.channels * self.config.data.image_size * self.config.data.image_size,
                                device=self.config.device)
                samples = net_test.sampling(z)
                if self.config.data.logit_transform:
                    samples = self.sigmoid_transform(samples)
                samples = torch.clamp(samples, min=0., max=1.)
                # save samples
                wandb.log({"samples": [wandb.Image(i) for i in samples[0:40]]})

                samples = make_grid(samples, 8)
                save_image(samples,
                           os.path.join(self.sample_dir, 'samples_{}.png'.format(epoch+1)))

            if self.config.data.dataset != 'ImageNet' and (epoch + 1) % self.config.training.snapshot_interval == 0:
                states = [
                    net.state_dict(),
                    optimizer.state_dict(),
                    epoch + 1,
                    step,
                    scheduler.state_dict(),
                ]
                if self.config.training.ema:
                    states.append(ema_helper.state_dict())
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc,
                                                'checkpoint_epoch_{}.pth'.format(epoch + 1)))
                torch.save(states, os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'))



    def test(self):
        import time
        torch.cuda.synchronize()
        start_time = time.time()

        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'MNIST':
            userid = getpass.getuser()
            test_dataset = MNIST(os.path.join('/atlas/u/{}/datasets'.format(userid), 'mnist'),
                                 train=False, download=False,
                                 transform=transform)

        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=False)

        net = Net(self.config).to(self.config.device)
        net = DataParallelWithSampling(net)
        if self.config.training.ema:
            ema_helper = EMAHelper(mu=0.999)
        optimizer = self.get_optimizer(net.parameters())

        def flow_loss(u, log_jacob, size_average=True):
            log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
            log_jacob = log_jacob.sum()
            loss = -(log_probs + log_jacob)

            if size_average:
                loss /= u.size(0)
            return loss

        states = torch.load(os.path.join(self.args.run, 'logs', self.args.doc, 'checkpoint.pth'),
                            map_location=self.config.device)

        net.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        loaded_epoch = states[2]
        if self.config.training.ema:
            ema_helper.load_state_dict(states[5])
            ema_helper.ema(net)

        logging.info(
            "Loading the model from epoch {}".format(loaded_epoch))

        # Test the model
        net.eval()
        total_loss = 0
        total_bpd = 0
        total_n_data = 0

        logging.info("Generating samples")
        z = torch.randn(64, self.config.data.channels * self.config.data.image_size * self.config.data.image_size,
                       device=self.config.device)
        samples = net.sampling(z)
        samples = self.sigmoid_transform(samples)

        samples = make_grid(samples, 8)
        save_image(samples,
                   os.path.join(self.sample_dir, '{}_samples.png'.format(self.config.data.dataset)))

        logging.info("Calculating overall bpd")

        with torch.no_grad():
            for batch_idx, (test_data, _) in enumerate(tqdm.tqdm(test_loader)):
                test_data = test_data.to(self.config.device) * 255. / 256.
                test_data += torch.rand_like(test_data) / 256.
                if self.config.data.logit_transform:
                    test_data = self.logit_transform(test_data)

                    test_log_det_logit = F.softplus(
                        -test_data).sum() + F.softplus(
                        test_data).sum() + np.prod(
                        test_data.shape) * np.log(
                        1 - 2 * self.config.data.lambda_logit)
                else:
                    test_log_det_logit = 0.

                test_output, test_log_det = net(test_data)
                test_loss = flow_loss(test_output, test_log_det)

                test_bpd = (test_loss.item() * test_data.shape[0] - test_log_det_logit) * (
                        1 / (np.log(2) * np.prod(test_data.shape))) + 8

                total_loss += test_loss * test_data.shape[0]
                total_bpd += test_bpd * test_data.shape[0]
                total_n_data += test_data.shape[0]
        logging.info(
            "Total batch:{}\nTotal loss: {}\nTotal bpd: {}".format(batch_idx + 1, total_loss.item() / total_n_data,
                                                                   total_bpd.item() / total_n_data))

        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print("Run-Time: %.4f s" % time_taken)

