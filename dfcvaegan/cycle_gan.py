# CSC 321, Assignment 4
#
# This is the main training file for CycleGAN.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to samples_cyclegan/):
#       python cycle_gan.py
#
#    To train with cycle consistency loss (saves results to samples_cyclegan_cycle/):
#       python cycle_gan.py --use_cycle_consistency_loss
#
#
#    For optional experimentation:
#    -----------------------------
#    If you have a powerful computer (ideally with a GPU), then you can obtain better results by
#    increasing the number of filters used in the generator and/or discriminator, as follows:
#      python cycle_gan.py --g_conv_dim=64 --d_conv_dim=64

import os
import pdb
import pickle
import argparse
import math

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

# Local imports
import utils
from models import CycleGenerator, DCDiscriminator


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



def create_model(opts):
    """Builds the generators and discriminators.
    """
    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)
    
    if torch.cuda.is_available():
        G_XtoY.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, D_Y


def checkpoint(iteration, G_XtoY, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_Y, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY.pkl')
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_XtoY_path = os.path.join(opts.load, 'G_XtoY.pkl')
    D_Y_path = os.path.join(opts.load, 'D_Y.pkl')

    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)
    
    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage), strict=False)
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))
    
    if torch.cuda.is_available():
        G_YtoX.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY,D_Y


def merge_images(sources, targets, opts, k=10):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    row = 4
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged.transpose(1, 2, 0)


def save_samples(iteration, fixed_Y, G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_Y = G_XtoY(fixed_Y)

    X, fake_Y = utils.to_data(fixed_Y), utils.to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    scipy.misc.imsave(path, merged)
    print('Saved {}'.format(path))


def training_loop(dataloader_X, test_dataloader_X, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    if opts.load:
        G_XtoY, D_Y = load_checkpoint(opts)
    else:
        G_XtoY, D_Y = create_model(opts)
    #G_XtoY = G_YtoX
    g_params = list(G_XtoY.parameters())  # Get generator parameters
    d_params = list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)

    test_iter_X = iter(test_dataloader_X)
    cross_entropy = nn.CrossEntropyLoss()

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_Y = utils.to_var(test_iter_X.next()[0])

    iter_per_epoch = len(iter_X)

    for iteration in range(1, opts.train_iters+1):
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)

        images_X, labels_X = iter_X.next()
        images_X, labels_X = utils.to_var(images_X), utils.to_var(labels_X).long().squeeze()

        d_optimizer.zero_grad()
        D_Y_loss = -math.log(D_Y(images_X))

        d_real_loss = D_Y_loss
        d_real_loss.backward()
        d_optimizer.step()

        d_optimizer.zero_grad()


        fake_Y, _, _ = G_XtoY(images_X)

        D_Y_loss = -math.log(D_Y(images_X) - 1)

        d_fake_loss = D_Y_loss
        d_fake_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()


        fake_Y, logvar_Y, mu_Y = G_XtoY(images_X)

        g_loss = ((fake_Y - images_X)**2).sum() #MSE
        g_loss += -0.5 * torch.sum(1 + logvar_Y - mu_Y.pow(2) - logvar_Y.exp())

        

        g_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | '
                  'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    iteration, opts.train_iters, d_real_loss.item(), D_Y_loss.item(), 
                    d_fake_loss.item(), g_loss.item()))


        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, G_XtoY, opts)


        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, D_Y, opts)


def main(opts, dataloader_X, test_dataloader_X):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, test_dataloader_X, opts)
            


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--use_cycle_consistency_loss', action='store_true', default=False, help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument('--init_zero_weights', action='store_true', default=False, help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=2000, help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=11, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=100)
    parser.add_argument('--checkpoint_every', type=int , default=800)

    parser.add_argument('--movements', dest='movements', nargs='*', default=ALL_MOVEMENTS)
    parser.add_argument('--data_path', default='/'.join(['', 'dfcvaegan','data','wikiart']))
    parser.add_argument('--balance_classes', type=int, default=1)
    parser.add_argument('--random_crop', type=int, default=0)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    if opts.use_cycle_consistency_loss:
        opts.sample_dir = 'samples_cyclegan_cycle'

    if opts.load:
        opts.sample_dir = '{}_pretrained'.format(opts.sample_dir)
        opts.sample_every = 20

    print_opts(opts)
    main(opts)
