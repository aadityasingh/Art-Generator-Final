import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torchvision.utils as tvut
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os

from model import VAE
from data_loader import load_data
from train import Trainer
from loss import Loss

ALL_MOVEMENTS = ['Early_Renaissance', 'Analytical_Cubism', 'Mannerism_Late_Renaissance', 
					'Expressionism', 'Contemporary_Realism', 'Fauvism', 'Northern_Renaissance', 
					'Rococo', 'Ukiyo_e', 'Pop_Art', 'High_Renaissance', 'Minimalism', 
					'Art_Nouveau_Modern', 'Action_painting', 'Color_Field_Painting', 
					'Symbolism', 'Realism', 'Romanticism', 'Cubism', 'Impressionism', 
					'New_Realism', 'Baroque', 'Post_Impressionism', 'Abstract_Expressionism', 
					'Pointillism', 'Synthetic_Cubism', 'Naive_Art_Primitivism']

def make_model(opts):
	train_loader, test_loader = load_data(opts)
	model = VAE()
	if torch.cuda.is_available():
		model.cuda()
		print('Using GPU')

	return model, train_loader, test_loader

def train(model, train_loader, test_loader, opts):
	optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
	loss = Loss(opts)

	if opts.load_from_chkpt != None:
		checkpoint = torch.load('/'.join([os.path.dirname(__file__),opts.run,'checkpoint.pth.tar']))
		model.load_state_dict(checkpoint['state_dict'])
		opts.start_epoch = checkpoint['epoch']
		optimizer.load_state_dict(checkpoint['optimizer'])

	trainer = Trainer(model, optimizer, loss, train_loader, test_loader, opts)

	# print(trainer)

	trainer.train(opts)

# CAUTION: Note that vae_params must be the same as the checkpoint... perhaps we can save this with the checkpoint for future
def gen_image(checkpoint_file):
	checkpoint = torch.load(checkpoint_file)

	model.load_state_dict(checkpoint['state_dict'])
	print(len(data.train_set))
	# print(data.train_set[0][0])
	print(data.train_set[0][0].size())
	batch1 = data.train_set[0][0].unsqueeze(0)
	print(batch1.size())
	print(batch1)
	to_save = data.un_norm(data.train_set[0][0])
	tvut.save_image(to_save, "goal1.png")
	model.eval()
	mu, logvar = model.encode(batch1)
	lat = model.reparamaterize(mu, logvar)
	print(model.decode(mu)[0])
	print(mu)
	tvut.save_image(data.un_norm(model.decode(mu)[0]), "generated_image1.png")
	
def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', dest='epochs', type=int, default = 10000)
	parser.add_argument('--test_every', dest='test_every', type = int, default = 50)
	parser.add_argument('--load_from_chkpt', dest='load_from_chkpt', default=None)
	parser.add_argument('--lr', dest='lr', type=float, default=0.001)
	parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.995)
	parser.add_argument('--lr_step', dest='lr_step', type=int, default=1)
	parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.3)
	parser.add_argument('--run', default='run')
	parser.add_argument('--image_size', type=int, default=128)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--start_epoch', type=int, default=0)
	parser.add_argument('--base_path', default='/dfcvaegan')

	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--movements', dest='movements', nargs='*', default=ALL_MOVEMENTS)
	parser.add_argument('--data_path', default='/'.join(['', 'dfcvaegan','data','wikiart']))
	parser.add_argument('--balance_classes', type=int, default=1)
	parser.add_argument('--random_crop', type=int, default=0)
	return parser


if __name__ == "__main__":
	parser = create_parser()
	opts = parser.parse_args()
	print(opts.data_path)
	print("Using movements", opts.movements)
	model, train_loader, test_loader = make_model(opts)
	# counts = [0, 0]
	# for i in range(2):
	# 	for batch_idx, (data, labels) in enumerate(train_loader):
	# 		print(labels)
	# 		if batch_idx > 5:
	# 			break
	# 	print('bla')
	# print(batch_idx)
	# print(counts)
	train(model, train_loader, test_loader, opts)
	# gen_image('./runs/checkpoints/checkpoint2.pth.tar')
