from torch.autograd import Variable
import torchvision.utils as tvut

from tensorboardX import SummaryWriter
import shutil
from tqdm import tqdm
import numpy as np
import utils

import os
import pdb
import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc

from utils import create_dir


# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Trainer:
    def __init__(self, model, optimizer, loss, train_loader, test_loader, opts):
        self.model = model
        self.opts = opts
        self.summary_dir = '/'.join([opts.base_path, 'runs', opts.run, 'logs'])
        self.checkpoint_dir = '/'.join([opts.base_path, 'runs', opts.run, 'checkpoints'])
        self.sample_dir = '/'.join([opts.base_path, 'runs', opts.run, 'samples'])
        create_dir(self.summary_dir)
        create_dir(self.checkpoint_dir)
        create_dir(self.sample_dir)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.fixed_X = utils.to_var(iter(test_loader).next()[0])
        print(self.fixed_X.size())

        self.loss = loss
        self.optimizer = optimizer
        # print()
        self.summary_writer = SummaryWriter(log_dir=self.summary_dir)
        print("Using cuda?", next(model.parameters()).is_cuda)

    def train(self, opts):
        self.model.train()
        last_loss = 10000000
        save_new_checkpoint = False
        for epoch in range(self.opts.start_epoch, self.opts.epochs):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
                if torch.cuda.is_available():
#                    print('using GPU')
                    data = data.cuda()
                data = Variable(data)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model.forward1(data)
                loss = self.loss(recon_batch, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            epoch_avg_loss = np.mean(loss_list)/self.opts.batch_size
            print("epoch {}: - training loss: {}".format(epoch, epoch_avg_loss))
            new_lr = self.adjust_learning_rate(epoch)
            print('learning rate:', new_lr)

            if epoch % opts.test_every == 0:
                new_loss = self.test(epoch)
                if epoch % opts.checkpoint_every == 0:
                    if new_loss < last_loss:
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, filename=self.opts.new_chkpt_fname)
                        print("Saved new checkpoint!", epoch)
                last_loss = new_loss
                self.summary_writer.add_scalar('training/loss', epoch_avg_loss, epoch)
                self.summary_writer.add_scalar('training/learning_rate', new_lr, epoch)
                # self.save_checkpoint({
                #     'epoch': epoch + 1,
                #     'state_dict': self.model.state_dict(),
                #     'optimizer': self.optimizer.state_dict(),
                # })
                # self.print_image("training/epoch"+str(epoch))
                self.save_samples(epoch)


    def test(self, cur_epoch):
        print('testing...')
        self.model.eval()
        test_loss = 0
        mse_loss = 0
        kld_loss = 0
        for i, (data, _) in enumerate(self.test_loader):
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)
            recon_batch, mu, logvar = self.model.forward1(data)
            test_loss += self.loss(recon_batch, data, mu, logvar).item()
            mse_loss += self.loss.mse(recon_batch, data).item()
            kld_loss += self.loss.kld(mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        mse_loss /= len(self.test_loader.dataset)
        kld_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Test set MSE loss: {:.4f}'.format(mse_loss))
        print('====> Test set KLD loss: {:.4f}'.format(kld_loss))
        self.summary_writer.add_scalar('testing/loss', test_loss, cur_epoch)
        self.summary_writer.add_scalar('testing/mseloss', mse_loss, cur_epoch)
        self.summary_writer.add_scalar('testing/kldloss', kld_loss, cur_epoch)
        self.model.train()
        return test_loss

    def merge_images(self, sources, targets, k=10):
        """Creates a grid consisting of pairs of columns, where the first column in
        each pair contains images source images and the second column in each pair
        contains images generated by the CycleGAN from the corresponding images in
        the first column.
        """
        _, _, h, w = sources.shape
        # row = int(np.sqrt(10)) # TODO: 10 is the hardcoded batch size
        row = 4
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1, 2, 0)


    def save_samples(self, epoch):
        """Saves samples from both generators X->Y and Y->X.
        """

        mu, logvar = self.model.encode(self.fixed_X)
        fake_X = self.model.decode(self.model.reparmaterize(mu, logvar))

        X, fake_X = utils.to_data(self.fixed_X), utils.to_data(fake_X)

        merged = self.merge_images(X, fake_X)
        path = '/'.join([self.sample_dir,'sample-{:06d}.png'.format(epoch)])
        scipy.misc.imsave(path, merged)
        print('Saved {}'.format(path))

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.opts.lr * (self.opts.lr_decay ** (epoch//self.opts.lr_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        return learning_rate

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        '''
        a function to save checkpoint of the training
        :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        :param is_best: boolean to save the checkpoint aside if it has the best score so far
        :param filename: the name of the saved file
        '''
        torch.save(state, '/'.join([self.checkpoint_dir,filename]))
        # if is_best:
        #     shutil.copyfile('/'.join([self.checkpoint_dir,filename]),
        #                     '/'.join([self.checkpoint_dir,'model_best.pth.tar']))
