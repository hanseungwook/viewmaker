import os
import copy
import getpass
from PIL import Image
import numpy as np
import random

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from src.datasets.root_paths import DATA_ROOTS


class Plankton(data.Dataset):
    NUM_CLASSES = 121
    NUM_CHANNELS = 3
    FILTER_SIZE = 64
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['plankton'],
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.image_transforms = image_transforms
        
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')


        if isinstance(image_transforms, list):
            self.dataset = datasets.ImageFolder(
                root
            )
        else:
            self.dataset = datasets.ImageFolder(
                root,
                transform=image_transforms
            )

    def __getitem__(self, index):
        
        if isinstance(self.image_transforms, list):
            # pick random number
            neg_index = np.random.choice(np.arange(self.__len__()))
            img_data, label = self.dataset.__getitem__(index)
            img2_data, _ = self.dataset.__getitem__(index)
            neg_data, _ = self.dataset.__getitem__(neg_index)
            
            img_data = self.image_transforms[0](img_data)
            img2_data = self.image_transforms[1](img2_data)
            neg_data = self.image_transforms[0](neg_data)
            # build this wrapper such that we can return index
            data = [index, img_data.float(), img2_data.float(), 
                    neg_data.float(), label]
            return tuple(data)
        else:
            # pick random number
            neg_index = np.random.choice(np.arange(self.__len__()))
            img_data, label = self.dataset.__getitem__(index)
            img2_data, _ = self.dataset.__getitem__(index)
            neg_data, _ = self.dataset.__getitem__(neg_index)
            # build this wrapper such that we can return index
            data = [index, img_data.float(), img2_data.float(), 
                    neg_data.float(), label]
            return tuple(data)

    def __len__(self):
        return len(self.dataset)