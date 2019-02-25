import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torchvision.utils as tvut
import torchvision.models as models
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os

from model import VAE, DCDiscriminator
from data_loader import load_data
from train import Trainer
from cnn_train import CNNTrainer
from loss import Loss
from evaluate import Evaluator


ALL_MOVEMENTS = ['Early_Renaissance', 'Analytical_Cubism', 'Mannerism_Late_Renaissance', 
					'Expressionism', 'Contemporary_Realism', 'Fauvism', 'Northern_Renaissance', 
					'Rococo', 'Ukiyo_e', 'Pop_Art', 'High_Renaissance', 'Minimalism', 
					'Art_Nouveau_Modern', 'Action_painting', 'Color_Field_Painting', 
					'Symbolism', 'Realism', 'Romanticism', 'Cubism', 'Impressionism', 
					'New_Realism', 'Baroque', 'Post_Impressionism', 'Abstract_Expressionism', 
					'Pointillism', 'Synthetic_Cubism', 'Naive_Art_Primitivism']

def make_model(opts):
	train_loader, test_loader = load_data(opts)
	model = VAE(conv_dim=opts.conv_dim, latent_vector=opts.latent_dim)
	if opts.cuda:
		model.cuda()
		print('Using GPU')

	return model, train_loader, test_loader

def make_cnn(opts):
	if opts.cnn_type == 'resnet18':
		opts.normalize = 'resnet'
		model = models.resnet18(pretrained=True)
	elif opts.cnn_type == 'resnet50':
		opts.normalize = 'resnet'
		model = models.resnet50(pretrained=True)
	else:
		raise NotImplementedError

	PYTORCH_RESNET_LAYERS = 10
	PYTORCH_REAL_RESNET_LAYERS = 8 # Last two layers are avg_pool and fc
	fixed_layers = PYTORCH_REAL_RESNET_LAYERS - opts.retrainable_layers
	for i, child in enumerate(model.children()):
		if i < fixed_layers:
			for param in child.parameters():
				param.requires_grad = False

	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs,len(opts.movements))

	if opts.cuda:
		model.cuda()
		print("Using GPU for CNN")
	train_loader, test_loader = load_data(opts)
	return model, train_loader, test_loader

def dfcvae_train(model, train_loader, test_loader, opts):
	optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
	loss = Loss(opts)

	if opts.load_from_chkpt != None:
		checkpoint = torch.load('/'.join([opts.base_path,'runs',opts.run,'checkpoints',opts.load_from_chkpt]))
		model.load_state_dict(checkpoint['state_dict'])
		opts.start_epoch = checkpoint['epoch']
		print("Start epoch", opts.start_epoch)
		optimizer.load_state_dict(checkpoint['optimizer'])

	trainer = Trainer(model, optimizer, loss, train_loader, test_loader, opts)

	# print(trainer)

	trainer.dfcvae_train(opts)

def vaegan_train(model, train_loader, test_loader, opts):
	optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
	loss = Loss(opts)

	discriminator = DCDiscriminator(conv_dim=opts.d_conv_dim)
	if opts.cuda:
		discriminator.cuda()
		print('Using GPU on discrim')
	d_optimizer = optim.Adam(discriminator.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

	if opts.load_from_chkpt != None:
		checkpoint = torch.load('/'.join([opts.base_path,'runs',opts.run,'checkpoints',opts.load_from_chkpt]))
		model.load_state_dict(checkpoint['state_dict'])
		opts.start_epoch = checkpoint['epoch']
		print("Start epoch", opts.start_epoch)
		optimizer.load_state_dict(checkpoint['optimizer'])

		d_checkpoint = torch.load('/'.join([opts.base_path,'runs',opts.run,'checkpoints','discrim_'+opts.load_from_chkpt]))
		discriminator.load_state_dict(checkpoint['state_dict'])
		opts.start_epoch = checkpoint['epoch']
		print("Start epoch", opts.start_epoch) # Should be the same as above
		d_optimizer.load_state_dict(checkpoint['optimizer'])

	trainer = Trainer(model, optimizer, loss, train_loader, test_loader, opts)

	# print(trainer)

	trainer.vaegan_train(discriminator, d_optimizer, opts)

def cnn_train(model, train_loader, test_loader, opts):
	# From keras script (thanks albert!)
	# Batch size: 32 (passed in)
	# Best optim: AMSGrad
	# lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opts.lr, amsgrad=True)
	loss = nn.CrossEntropyLoss()

	if opts.load_from_chkpt != None:
		checkpoint = torch.load('/'.join([opts.base_path,'cnn_runs',opts.run,'checkpoints',opts.load_from_chkpt]))
		model.load_state_dict(checkpoint['state_dict'])
		opts.start_epoch = checkpoint['epoch']
		print("Start epoch", opts.start_epoch)
		optimizer.load_state_dict(checkpoint['optimizer'])

	trainer = CNNTrainer(model, optimizer, loss, train_loader, test_loader, opts)

	# print(trainer)

	trainer.train(opts)

def eval(model, classifier, test_loader, opts):
	checkpoint = torch.load('/'.join([opts.base_path,'runs',opts.run,'checkpoints',opts.load_from_chkpt]), map_location='cpu')
	model.load_state_dict(checkpoint['state_dict'])

	run_data_type = opts.run.split("Class")[0]
	checkpoint = torch.load('/'.join([opts.base_path,'cnn_runs',run_data_type+"ClassRes18R0",'checkpoints',"checkpoint.pth.tar"]), map_location='cpu')
	classifier.load_state_dict(checkpoint['state_dict'])

	evaluator = Evaluator(model, classifier, test_loader, opts)

	if opts.eval_type == 'all':
		evaluator.generate()
		evaluator.cluster()
		evaluator.interpolate()
		evaluator.four_by_four()
		evaluator.latent_grid()
	elif opts.eval_type == 'randgen':
		evaluator.generate()
	elif opts.eval_type == 'cluster':
		evaluator.cluster()
	elif opts.eval_type == 'transform':
		evaluator.interpolate()
	elif opts.eval_type == 'real_cluster':
		evaluator.latent_grid()
	else:
		raise NotImplementedError

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
	parser.add_argument('--mode', dest='mode', default='train_dfcvae', help='Choose from train_dfcvae, train_vaegan, train_cnn, eval')
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--normalize', default='standard')


	parser.add_argument('--epochs', dest='epochs', type=int, default = 10000)
	parser.add_argument('--test_every', dest='test_every', type = int, default = 2)
	parser.add_argument('--checkpoint_every', dest='checkpoint_every', type = int, default = 4)
	parser.add_argument('--load_from_chkpt', dest='load_from_chkpt', default=None) # Also used for eval
	parser.add_argument('--new_chkpt_fname', dest='new_chkpt_fname', default="checkpoint.pth.tar")
	parser.add_argument('--lr', dest='lr', type=float, default=0.001)
	parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.99)
	parser.add_argument('--lr_step', dest='lr_step', type=int, default=1)
	parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.3)
	parser.add_argument('--run', default='run')
	parser.add_argument('--image_size', type=int, default=128)
	parser.add_argument('--conv_dim', type=int, default=64)
	parser.add_argument('--latent_dim', type=int, default=800)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--start_epoch', type=int, default=0)
	parser.add_argument('--base_path', default='/dfcvaegan')

	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--movements', dest='movements', nargs='*', default=ALL_MOVEMENTS) # Note if in eval mode, these are the movements to eval on (can be subset of trained movements)
	parser.add_argument('--data_path', default='/'.join(['', 'dfcvaegan','data','wikiart']))
	parser.add_argument('--balance_classes', type=int, default=1)
	parser.add_argument('--random_crop', type=int, default=0)

	parser.add_argument('--dfc', default='discrim', help='Pick from discrim, encoder')
	parser.add_argument('--dfc_path', default='D_Xfull.pkl', help='Path beyond base_path to get to pkl or checkpoint file... example: runs/fiveClassTrueTrue/checkpoints/checkpoint.pth.tar')

	# Evaluation arguments
	parser.add_argument('--eval_type', dest='eval_type', default='all', help='Choose from randgen, cluster, transform, real_cluster, all')
	# Note for eval, we always use the Res18R1 CNN trained on that dataset
	parser.add_argument('--op_samples', dest='op_samples', type=int, default=400, help='Number of images to do eval_type on and plot')
	parser.add_argument('--pca_components', dest='pca_components', type=int, default=6, help='Number of PCA components to show')
	parser.add_argument('--random_pick', dest='random_pick', type=int, default=0, help="Whether or not to pick images to interpolate randomly")
	# TODO make intermediate a passable parameter... not enough time rn tho
	# parser.add_argument('--intermediate', dest='intermediate', type=int, default=16, help='How many intermediate images to have in interpolation')

	# For vaegan
	parser.add_argument('--d_conv_dim', type=int, default=64)
	parser.add_argument('--d_weight', type=float, default=1, help='How much to weight the cross entropy term in VAEGAN')
	parser.add_argument('--dfc_weight', type=float, default=1, help='How much to weight dfc/mse term in VAEGAN')
	parser.add_argument('--use_dfc', type=int, default=1, help='Either use dfc loss with gan or use mse loss. Defaults to dfc')

	# For CNN
	parser.add_argument('--cnn_type', default='resnet18', help='Choose from resnet18, resnet50')
	parser.add_argument('--retrainable_layers', type=int, default=0, help='Choose from [0,1,2] using pytorch layers in resnet... Last layer is always retrained')

	# Training hyper-parameters
	# parser.add_argument('--train_iters', type=int, default=2000, help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
	# parser.add_argument('--beta1', type=float, default=0.5)
	# parser.add_argument('--beta2', type=float, default=0.999)

	return parser


if __name__ == "__main__":
	parser = create_parser()
	opts = parser.parse_args()
	print(opts.data_path)
	print("Using movements", opts.movements)
	opts.cuda = 1 if torch.cuda.is_available() else 0
	# counts = [0, 0]
	# for i in range(2):
	# 	for batch_idx, (data, labels) in enumerate(train_loader):
	# 		print(labels)
	# 		if batch_idx > 5:
	# 			break
	# 	print('bla')
	# print(batch_idx)
	# print(counts)
	if opts.mode == 'train_dfcvae':
		model, train_loader, test_loader = make_model(opts)
		dfcvae_train(model, train_loader, test_loader, opts)
	elif opts.mode == 'train_vaegan':
		model, train_loader, test_loader = make_model(opts)
		vaegan_train(model, train_loader, test_loader, opts)
	elif opts.mode == 'train_cnn':
		opts.image_size = 224
		model, train_loader, test_loader = make_cnn(opts)
		cnn_train(model, train_loader, test_loader, opts)
	elif opts.mode == 'eval':
		opts.cuda = 0
		opts.batch_size = opts.op_samples
		model, _, test_loader = make_model(opts)
		classifier, _, _ = make_cnn(opts)
		eval(model, classifier, test_loader, opts)
	else:
		raise NotImplementedError
	# gen_image('./runs/checkpoints/checkpoint2.pth.tar')
