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

import math

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc

class CNNTrainer:
    def __init__(self, model, optimizer, loss, train_loader, test_loader, opts):
        self.model = model
        self.opts = opts
        self.cuda = opts.cuda
        self.summary_dir = '/'.join([opts.base_path, 'cnn_runs', opts.run, 'logs'])
        self.checkpoint_dir = '/'.join([opts.base_path, 'cnn_runs', opts.run, 'checkpoints'])
        self.sample_dir = '/'.join([opts.base_path, 'cnn_runs', opts.run, 'samples'])
        utils.create_dir(self.summary_dir)
        utils.create_dir(self.checkpoint_dir)
        utils.create_dir(self.sample_dir)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss = loss
        self.optimizer = optimizer
        # print()
        self.summary_writer = SummaryWriter(log_dir=self.summary_dir)
        print("Using cuda?", next(model.parameters()).is_cuda)

    def train(self, opts):
        self.model.train()
        last_loss = 10000000
        last_accuracy = 0
        for epoch in range(self.opts.start_epoch, self.opts.epochs):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader)):
                if self.cuda:
#                    print('using GPU')
                    data = data.cuda()
                    labels = labels.cuda()
                data = Variable(data)
                self.optimizer.zero_grad()
                linear_pred = self.model(data)
                loss = self.loss(linear_pred, labels)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            epoch_avg_loss = np.mean(loss_list)/self.opts.batch_size
            print("epoch {}: - training loss: {}".format(epoch, epoch_avg_loss))

            if epoch % opts.test_every == 0:
                new_loss, new_accuracy = self.test(epoch)
                if epoch % opts.checkpoint_every == 0:
                    if (new_loss < last_loss) or (new_accuracy > last_accuracy):
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, filename=self.opts.new_chkpt_fname)
                        print("Saved new checkpoint!", epoch)
                last_loss = new_loss
                last_accuracy = new_accuracy
                self.summary_writer.add_scalar('training/loss', epoch_avg_loss, epoch)

    def test(self, cur_epoch):
        print('testing...')
        self.model.eval()
        test_loss = 0
        correctly_classified = 0
        for i, (data, labels) in enumerate(self.test_loader):
            if self.cuda:
                data = data.cuda()
                labels = labels.cuda()
            data = Variable(data)
            linear_pred = self.model(data)
            test_loss += self.loss(linear_pred, labels).item()
            correctly_classified += (linear_pred.max(dim = 1)[1] == labels).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = correctly_classified/len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Test set accuracy: {:.6f}'.format(accuracy))
        self.summary_writer.add_scalar('testing/loss', test_loss, cur_epoch)
        self.summary_writer.add_scalar('testing/accuracy', accuracy, cur_epoch)
        self.model.train()
        return test_loss, accuracy

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
