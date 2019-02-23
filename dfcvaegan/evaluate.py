import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torchvision.utils as tvut
import argparse
import torch.nn as nn
from torch.autograd import Variable

from model import VAE
from data_loader import load_data
from train import Trainer
from loss import Loss
import utils

# import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc
import os
from tqdm import tqdm

COLORS = ['r', 'g', 'b', 'c', 'm', 'k']

class Evaluator:
	def __init__(self, model, classifier, test_loader, opts):
		self.eval_dir = '/'.join([opts.base_path, 'runs', opts.run, 'eval'])
		self.pca_dir = '/'.join([self.eval_dir, 'cluster'])
		self.gen_dir = '/'.join([self.eval_dir, 'random'])
		self.trans_dir = '/'.join([self.eval_dir, 'transform'])
		utils.create_dir(self.eval_dir)
		utils.create_dir(self.pca_dir)
		utils.create_dir(self.gen_dir)
		utils.create_dir(self.trans_dir)

		self.model = model
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.eval()

		self.test_loader = test_loader
		print("Getting samples")
		self.sample_set = iter(self.test_loader).next()
		print("Samples in memory")
		self.images = self.sample_set[0]
		print(self.images.shape)
		
		self.latents = self.model.encode(self.images)[0].data.numpy() # Note the [0] is to extract the mean
		print("Encoded latents")
		self.labels = self.sample_set[1].data.numpy()
		self.latent_means = None
		self.latent_stds = None
		self.normalized_latents = None

		self.classifier = classifier
		self.classifier.eval()
		self.num_movements = len(opts.movements)

		self.sfmax = nn.Softmax(dim=1)

		self.opts = opts


	def cluster(self):
		print("Conducting PCA")
		if self.normalized_latents:
			pass
		else:
			self.latent_means = np.mean(self.latents, axis=0)
			self.latent_stds = np.std(self.latents, axis=0)
			print(self.latent_means.shape)
			print(self.latent_stds.shape)
			self.normalized_latents = (self.latents-self.latent_means)/self.latent_stds
			print(self.normalized_latents.shape)

		pca = PCA(n_components=self.opts.pca_components)
		pca.fit(self.normalized_latents)
		print("Finished PCA fit... moving to graphing")

		fig1, axes1 = plt.subplots(self.opts.pca_components, 1, sharex=True, sharey=True)
		plt.setp(axes1, xlim=(-10, 10), ylim=(-1, self.num_movements), xticks=[-5, 0, 5], yticks=list(range(self.num_movements)), yticklabels=self.opts.movements)
		fig2, axes2 = plt.subplots(1, 1)
		plt.setp(axes2, xlim=(-10, 10), ylim=(-10, 10))
		for batch_idx, (data, labels) in enumerate(tqdm(self.test_loader)):
			labels = labels.data.numpy()
			pca_latent = pca.transform(self.model.encode(data)[0].data.numpy())
			axes2.scatter(pca_latent[:, 0], pca_latent[:, 1], c=[COLORS[i] for i in labels])
			for i in range(self.opts.pca_components):
				# plt.subplot(comps, 1, i+1)
				axes1[i].scatter(pca_latent[:, i], labels, c=[COLORS[i] for i in labels])
				# axes[i].xlim(-10, 10)
				# axes[i].ylim(-0.5, 2.5)
				# axes[i].yticks([])
				# ax.set_ylabel('Component ' + str(i+1))
				# ax.set_yticklabels([])
			if batch_idx > 4:
				break
		print("saving plot, done with PCA")
		fig1.savefig('/'.join([self.pca_dir, f'S{self.opts.op_samples}_C{self.opts.pca_components}']))
		fig2.savefig('/'.join([self.pca_dir, f'Y2_X1']))

	def generate(self):
		self.save_image(torch.randn(self.num_movements, self.opts.latent_dim), 'random')
		# save_samples(torch.randn(10, 800), 'random3.png')
		print("Saved random images")

		centers = np.zeros((self.num_movements, self.opts.latent_dim))
		noisy = np.zeros((self.num_movements, self.opts.latent_dim))
		for i in range(self.num_movements):
			# print(np.where(self.labels == i))
			centers[i, :] = np.mean(self.latents[np.where(self.labels == i)], axis=0)
			noisy[i, :] = centers[i, :] + np.random.randn(self.opts.latent_dim)*0.5
			# print(centers[i, :])

		self.save_image(torch.from_numpy(centers).float(), '/'.join([self.gen_dir, 'center_']), movements=True)
		print("Saved center images")
		self.save_image(torch.from_numpy(noisy).float(), '/'.join([self.gen_dir, 'noisy_center_']), movements=True)
		print("Saved noisy center images")

	def interpolate(self):
		movement_latents = [0]*self.num_movements
		saved = [False]*self.num_movements
		if self.opts.random_pick:
			for i, movement in enumerate(self.opts.movements):
				filtered_latents = self.latents[np.where(self.labels == i)]
				if len(filtered_latents):
					print("NO MOVEMENT LATENT FOUND SAD")
					return
				movement_latents[i] = filtered_latents[np.random.randint(len(filtered_latents))]
		else:
			for i, label in enumerate(self.labels):
				if not saved[label]:
					movement_latents[label] = self.latents[i]
					saved[label] = True
				if False not in saved:
					break
			if False in saved:
				print("NO MOVEMENT LATENT FOUND SAD")
				return
		print("Found movement latents, interpolating now...")
		# First part is generating paired crosses, their line graphs, and also bar plots (just in case i feel like gif'ing)
		for i in range(self.num_movements):
			for j in range(i+1, self.num_movements):
				diff = movement_latents[j] - movement_latents[i]
				to_stack = []
				to_stack.append(movement_latents[i])
				for k in range(15):
					to_stack.append(movement_latents[i]+diff*(k+1)/15)
				stack_tensor = torch.from_numpy(np.stack(to_stack)).float()

				name = self.opts.movements[i] + "To" + self.opts.movements[j]
				utils.create_dir('/'.join([self.trans_dir, name]))
				self.save_samples(stack_tensor, '/'.join([self.trans_dir, name]))
				self.save_samples(stack_tensor, '/'.join([self.trans_dir, name+"Horizontal"]), rows=1, columns=16)
				tensors = self.save_image(stack_tensor, '/'.join([self.trans_dir, name, '/']))
				print(tensors.shape)
				classes = utils.to_data(self.sfmax(self.classifier(tensors)))
				fig, axes = plt.subplots(1, 1)
				plt.setp(axes2, xlim=(0, 1), ylim=(0, 1))
				for i in range(self.num_movements):
					axes.plot(np.linspace(0, 1, 16), classes[:, i], COLORS[i]+'-')
				axes.savefig('/'.join([self.trans_dir, name+"Graph.png"]))
				print(name, 'done')

	def four_by_four(self):
		pass

	def latent_grid(self):
		pass

	def merge_images(self, sources, rows=4, columns=4):
		"""Creates a grid consisting of pairs of columns, where the first column in
		each pair contains images source images and the second column in each pair
		contains images generated by the CycleGAN from the corresponding images in
		the first column.
		"""
		# print(sources.shape)
		_, _, h, w = sources.shape
		# row = int(np.sqrt(10)) # TODO: 10 is the hardcoded batch size
		merged = np.zeros([3, rows*h, columns*w])
		# print(merged.shape)
		for idx, s in enumerate(sources):
			print(s.shape)

			i = idx // columns
			j = idx % rows
			# print(j*w)
			# print((j+1)*w)
			# print(merged[:, i*h:(i+1)*h, j*w:(j+1)*w].shape)

			merged[:, i*h:(i+1)*h, j*w:(j+1)*w] = s
			# merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
		return merged.transpose(1, 2, 0)


	def save_samples(self, latent, filename, rows=4, columns=4):
		fake_X = self.model.decode(latent)

		fake_X = utils.to_data(fake_X)

		merged = self.merge_images(fake_X, rows=rows, columns=columns)
		# path = os.path.join(os.path.join(os.path.dirname(__file__),'generated'), filename)
		scipy.misc.imsave(filename+".png", merged)
		# print('Saved {}'.format(path))

	def save_image(self, latent, fileroot, movements=False):
		generated_tensors = self.model.decode(latent)
		print(generated_tensors.shape)

		generated = utils.to_data(generated_tensors)
		if movements:
			for i, image in enumerate(generated):
				scipy.misc.imsave(fileroot+self.opts.movements[i]+".png", image.transpose(1, 2, 0))
		else:
			for i, image in enumerate(generated):
				scipy.misc.imsave(fileroot+str(i)+".png", image.transpose(1, 2, 0))
		return generated_tensors

# translation = False
# if translation:
# 	for ind1 in range(10):
# 		for ind2 in range(ind1+1, 10):
# 			diff = latent_vectors[ind2:(ind2+1), :] - latent_vectors[ind1:(ind1+1), :]
# 			stack = latent_vectors[ind1:(ind1+1), :]
# 			for i in range(15):
# 				stack = np.vstack((stack, latent_vectors[ind1:(ind1+1), :]+diff*(i+1)/15))
# 			stack_tensor = torch.from_numpy(stack).float()

# 			if not os.path.isdir("generated/"+movements[categories[ind1]]+"To"+movements[categories[ind2]]):
# 				os.mkdir("generated/"+movements[categories[ind1]]+"To"+movements[categories[ind2]])
# 				save_samples(stack_tensor, movements[categories[ind1]]+"To"+movements[categories[ind2]]+".png")
# 				save_image(stack_tensor, movements[categories[ind1]]+"To"+movements[categories[ind2]]+"/")
# 			else:
# 				print("already done!")

# 			randn = np.random.randn(800)*0.5
# 			diff = latent_vectors[ind2:(ind2+1), :]+np.random.randn(800)*0.5 - latent_vectors[ind1:(ind1+1), :]-randn
# 			stack = latent_vectors[ind1:(ind1+1), :]+randn
# 			for i in range(15):
# 				stack = np.vstack((stack, latent_vectors[ind1:(ind1+1), :]+diff*(i+1)/15))
# 			stack_tensor = torch.from_numpy(stack).float()
# 			if not os.path.isdir("generated/"+"Noisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]]):
# 				os.mkdir("generated/"+"Noisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]])
# 				save_samples(stack_tensor, "Noisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]]+".png")
# 				save_image(stack_tensor, "Noisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]]+"/")
# 			else:
# 				print('already done noisy!')

# 			randn = np.random.randn(800)*1
# 			diff = latent_vectors[ind2:(ind2+1), :]+np.random.randn(800)*1 - latent_vectors[ind1:(ind1+1), :]-randn
# 			stack = latent_vectors[ind1:(ind1+1), :]+randn
# 			for i in range(15):
# 				stack = np.vstack((stack, latent_vectors[ind1:(ind1+1), :]+diff*(i+1)/15))
# 			stack_tensor = torch.from_numpy(stack).float()
# 			if not os.path.isdir("generated/"+"VeryNoisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]]):
# 				os.mkdir("generated/"+"VeryNoisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]])
# 				save_samples(stack_tensor, "VeryNoisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]]+".png")
# 				save_image(stack_tensor, "VeryNoisy"+movements[categories[ind1]]+"To"+movements[categories[ind2]]+"/")
# 			else:
# 				print('already done noisy!')

# # merged = merge_images(fake_X)
# # path = os.path.join(os.path.join(os.path.dirname(__file__),'generated'), filename)
# # scipy.misc.imsave(path, merged)
# # plt.show()