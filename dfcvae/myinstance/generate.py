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

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

image_size = 128
num_images = 700
color = ['r', 'g', 'b']

data_load_params = {'batch_size' : num_images, 'shuffle' : True, 'cuda': False, 'num_workers' : 1, 'pin_memory' : True,
					'image_size': image_size, 'valid_split': 0.2, 'data_path': './data/touse/',
					'normalize': [[0.57867116, 0.53378844, 0.47493997], [0.2312678, 0.21923426, 0.22560729]]}

train_loader, test_loader = load_data("Pointillism", data_load_params)
model = VAE()
if torch.cuda.is_available():
	model.cuda()
	print('Using GPU')
checkpoint = torch.load('./runs/checkpoints/checkpoint3.pth.tar', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
for param in model.parameters():
	param.requires_grad = False
model.eval()

images = iter(train_loader).next()
latent_vectors = model.encode(images[0])[0].data.numpy()
categories = images[1].data.numpy()
plt.figure(1)
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=[color[i] for i in categories])
print("Made first figure")
# print(latent_vectors)
# print(latent_vectors.shape)
# print(categories)
# print(categories.shape)
# print(np.mean(latent_vectors, axis=0))
# # print(np.mean(latent_vectors).size())
# print(np.std(latent_vectors, axis=0))

latent_df = pd.DataFrame(latent_vectors)
# latent_df = StandardScaler().fit_transform(latent_df)

# print(latent_df)
# print()
comps = 10
pca = PCA(n_components=comps)
principalComponents = pca.fit_transform(latent_df)
principalDf = pd.DataFrame(data = principalComponents)
print(principalDf.head(100))
pca_latent = principalDf.values

print(pca_latent.shape)
print(pca.components_)

plt.figure(2)
for i in range(comps):
	plt.subplot(comps, 1, i+1)
	plt.scatter(pca_latent[:, i], categories, c=[color[i] for i in categories])
# plt.subplot(2, 2, 2)
# plt.scatter(pca_latent[:, 2], pca_latent[:, 3], c=[color[i] for i in categories])
# plt.subplot(2, 2, 3)
# plt.scatter(pca_latent[:, 4], pca_latent[:, 5], c=[color[i] for i in categories])
# plt.subplot(2, 2, 4)
# plt.scatter(pca_latent[:, 6], pca_latent[:, 7], c=[color[i] for i in categories])
plt.show()


# def train(opts):
# 	training_params = {'num_epochs' : opts.epochs, 'learning_rate' : opts.lr, 'weight_decay' : 0.3, 'learning_rate_decay' : opts.decay, 'cuda' : False, 
# 		'summary_dir' : './runs/logs/', 'checkpoint_dir' : './runs/checkpoints/'}

# 	if opts.checkpoint == 1:
# 		checkpoint = torch.load('./runs/checkpoints/checkpoint2.pth.tar')
# 		model.load_state_dict(checkpoint['state_dict'])

# 	loss = Loss()

# 	trainer = Trainer(model, loss, train_loader, test_loader, training_params)

# 	# print(trainer)

# 	trainer.train(opts)

# # CAUTION: Note that vae_params must be the same as the checkpoint... perhaps we can save this with the checkpoint for future
# def gen_image(checkpoint_file):
# 	checkpoint = torch.load(checkpoint_file)

# 	model.load_state_dict(checkpoint['state_dict'])
# 	print(len(data.train_set))
# 	# print(data.train_set[0][0])
# # <<<<<<< HEAD
# 	print(data.train_set[0][0].size())
# 	batch1 = data.train_set[0][0].unsqueeze(0)
# 	print(batch1.size())
# 	print(batch1)
# 	to_save = data.un_norm(data.train_set[0][0])
# 	tvut.save_image(to_save, "goal1.png")
# 	model.eval()
# 	mu, logvar = model.encode(batch1)
# 	lat = model.reparamaterize(mu, logvar)
# 	print(model.decode(mu)[0])
# 	print(mu)
# 	tvut.save_image(data.un_norm(model.decode(mu)[0]), "generated_image1.png")
# # =======
# # 	# print(data.train_set[0][0].size())
# # 	batch1 = data.train_set[9][0].unsqueeze(0).cuda()
# # 	# print(batch1.size())

# # 	inp = Variable(batch1)
# # 	tvut.save_image(batch1, "goal2.png")
# # 	# print(inp)
# # 	model.eval()
# # 	mu, logvar = model.encode(batch1)
# # 	lat = model.reparamaterize(mu, logvar)
# # >>>>>>> gans
	
# 	# generated = model.decode(mu)

# 	# print(mu, logvar, generated)
# 	# tvut.save_image(generated, "generated_image2.png")
	
# def create_parser():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--images', type = int, default = 100)
# 	return parser


# if __name__ == "__main__":
# 	parser = create_parser()
# 	opts = parser.parse_args()
	
# 	pca_plot(opts)
	# gen_image('./runs/checkpoints/checkpoint2.pth.tar')
