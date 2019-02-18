import torch
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

from model import DCDiscriminator

class Loss(nn.Module):
    def __init__(self, opts):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)
        self.DFCNet = DCDiscriminator(conv_dim=64)
        self.DFCNet.load_state_dict(torch.load('/'.join([opts.base_path, 'D_Xfull.pkl']), map_location=lambda storage, loc: storage))
        if torch.cuda.is_available():
        	self.DFCNet.cuda()
        for param in self.DFCNet.parameters():
        	param.requires_grad = False
        self.DFCNet.eval()

    def forward(self, recon_x, x, mu, logvar):
        return self.mse(recon_x, x) + self.kld(mu, logvar)

    def kld(self, mu, logvar):
        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # -KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def mse(self, recon_x, x):
        x = F.relu(self.DFCNet.conv1(x))    # BS x 64 x 16 x 16
        recon_x = F.relu(self.DFCNet.conv1(recon_x))
        MSE = self.mse_loss(recon_x, x)
        # print(MSE)
        x = F.relu(self.DFCNet.conv2(x))  # BS x 128 x 8 x 8
        recon_x = F.relu(self.DFCNet.conv2(recon_x))
        MSE += self.mse_loss(recon_x, x)
        # print(MSE)
        x = F.relu(self.DFCNet.conv3(x))
        recon_x = F.relu(self.DFCNet.conv3(recon_x))
        MSE += self.mse_loss(recon_x, x)
        return MSE


# Note batch norms ommitted since just used for eval anyways
# class DCDiscriminator(nn.Module):
#     def __init__(self, conv_dim=64):
#         super(DCDiscriminator, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(in_channels=conv_dim * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)

#     def forward(self, x):

#         out = F.relu(self.conv1(x))    # BS x 64 x 16 x 16
#         out = F.relu(self.conv2(out))  # BS x 128 x 8 x 8
#         out = F.relu(self.conv3(out))  # BS x 256 x 4 x 4

#         out = self.conv4(out).squeeze()
#         out = F.sigmoid(out)
#         return out
