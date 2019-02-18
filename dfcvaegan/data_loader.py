import os
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision import datasets
from torchvision import transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ArtDataset(Dataset):
    def __init__(self, root, opts target_transform=None):
        self.image_size = opts.image_size
        self.resize_transform = transforms.Compose([
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #fix normalization
                    ])

        self.crop_transform = transforms.Compose([
                        transforms.RandomCrop((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.random_crop = opts.random_crop

        self.samples = []
        self.class_counts = [0]*len(opts.movements)
        for i, movement in enumerate(opts.movements):
            for fname in os.listdir('/'.join([root, movement])):
                if fname[-4:] in ['.jpg', '.png']:
                    self.samples.append(('/'.join([root, movement, fname]), i))
                    self.class_counts[i] += 1

        self.target_transform = target_transform
        print("Length of dataset:", len(self.samples))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if (sample.size[0] < self.image_size) or (sample.size[1] < self.image_size):
            print("Small image so resizing up", path)
            sample = self.resize_transform(sample)
        elif self.random_crop:
            sample = self.crop_transform(sample)
        else:
            sample = self.resize_transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

def load_data(opts):
    """Creates training and test data loaders.
    """

    train_path = '/'.join([opts.data_path,'train'])
    test_path = '/'.join([opts.data_path,'test'])

    train_dataset = ArtDataset(train_path, opts)
    test_dataset = ArtDataset(test_path, opts)

    train_sampler = WeightedRandomSampler([1/train_dataset.class_counts[sample[1]] for sample in train_dataset.samples], len(train_dataset.samples))
    test_sampler = WeightedRandomSampler([1/test_dataset.class_counts[sample[1]] for sample in test_dataset.samples], len(test_dataset.samples))

    if opts.balance_classes:
        print("Using WeightedRandomSampler")
        train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, sampler=train_sampler, num_workers=opts.num_workers)
        test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, sampler=test_sampler, num_workers=opts.num_workers)
    else:
        print("Not weighting classes")
        train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
        test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return train_dloader, test_dloader
