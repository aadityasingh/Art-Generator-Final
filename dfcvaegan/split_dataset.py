import os
import shutil
from utils import create_dir
from tqdm import tqdm
import numpy as np

# This file creates a 20/80 val/train split

movements = os.listdir('./data/wikiart')
print(movements)

create_dir('./data/wikiart/train')
create_dir('./data/wikiart/test')

for movement in tqdm(movements):
	images = os.listdir('./data/wikiart/'+movement)
	shuffled = np.random.permutation(images)
	create_dir('/'.join(['./data/wikiart','test', movement]))
	create_dir('/'.join(['./data/wikiart','train', movement]))
	for i, imname in enumerate(shuffled):
		if i <= len(images)//5:
			os.rename('/'.join(['./data/wikiart',movement, imname]), '/'.join(['./data/wikiart','test', movement, imname]))
		else:
			os.rename('/'.join(['./data/wikiart',movement, imname]), '/'.join(['./data/wikiart','train', movement, imname]))

