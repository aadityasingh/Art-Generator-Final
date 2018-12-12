import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torchvision.utils as tvut
import argparse
import torch as nn
from torch.autograd import Variable

from model import VAE
from data_loader import load_data
from train import Trainer
from loss import Loss

image_size = 128

data_load_params = {'batch_size' : 10, 'shuffle' : True, 'cuda': False, 'num_workers' : 1, 'pin_memory' : True,
					'image_size': image_size, 'valid_split': 0.2, 'data_path': './data/touse/',
					'normalize': [[0.57867116, 0.53378844, 0.47493997], [0.2312678, 0.21923426, 0.22560729]]}

train_loader, test_loader = load_data("Pointillism", data_load_params)
model = VAE()
if torch.cuda.is_available():
	model.cuda()
	print('Using GPU')

def train(opts):
	training_params = {'num_epochs' : opts.epochs, 'learning_rate' : opts.lr, 'weight_decay' : 0.3, 'learning_rate_decay' : opts.decay, 'cuda' : False, 
		'summary_dir' : './runs/logs/', 'checkpoint_dir' : './runs/checkpoints/'}

	if opts.checkpoint == 1:
		checkpoint = torch.load('./runs/checkpoints/checkpoint2.pth.tar')
		model.load_state_dict(checkpoint['state_dict'])

	loss = Loss()

	trainer = Trainer(model, loss, train_loader, test_loader, training_params)

	# print(trainer)

	trainer.train(opts)

# CAUTION: Note that vae_params must be the same as the checkpoint... perhaps we can save this with the checkpoint for future
def gen_image(checkpoint_file):
	checkpoint = torch.load(checkpoint_file)

	model.load_state_dict(checkpoint['state_dict'])
	print(len(data.train_set))
	# print(data.train_set[0][0])
# <<<<<<< HEAD
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
# =======
# 	# print(data.train_set[0][0].size())
# 	batch1 = data.train_set[9][0].unsqueeze(0).cuda()
# 	# print(batch1.size())

# 	inp = Variable(batch1)
# 	tvut.save_image(batch1, "goal2.png")
# 	# print(inp)
# 	model.eval()
# 	mu, logvar = model.encode(batch1)
# 	lat = model.reparamaterize(mu, logvar)
# >>>>>>> gans
	
	# generated = model.decode(mu)

	# print(mu, logvar, generated)
	# tvut.save_image(generated, "generated_image2.png")
	
def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type = int, default = 10000)
	parser.add_argument('--test_every', type = int, default = 100)
	parser.add_argument('--checkpoint', type=int, default=0)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--decay', type=float, default=0.995)
	return parser


if __name__ == "__main__":
	parser = create_parser()
	opts = parser.parse_args()
	
	train(opts)
	# gen_image('./runs/checkpoints/checkpoint2.pth.tar')
