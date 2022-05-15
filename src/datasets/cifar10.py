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


class CIFAR10(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.image_transforms = image_transforms
        
        # Two different sets of transformations for the two views
        if isinstance(image_transforms, list):
            self.dataset = datasets.cifar.CIFAR10(
                root,
                train=train,
                download=True)
        else:
            self.dataset = datasets.cifar.CIFAR10(
                root, 
                train=train,
                download=True,
                transform=image_transforms,
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

class CIFAR10A2(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.image_transforms = image_transforms
        
        self.dataset = datasets.cifar.CIFAR10(
            root,
            train=train,
            download=True)

    def __getitem__(self, index):
        if isinstance(self.image_transforms, list):
            # pick random number
            img_data, label = self.dataset.__getitem__(index)
            img2_data, _ = self.dataset.__getitem__(index)
            img3_data, _ = self.dataset.__getitem__(index)
            
            img_data = self.image_transforms[0](img_data)
            img2_data = self.image_transforms[0](img2_data)
            img3_data = self.image_transforms[1](img3_data)
            # build this wrapper such that we can return index
            data = [index, img_data.float(), img2_data.float(), 
                    img3_data.float(), label]

            return tuple(data)
        else:
            img_data, label = self.dataset.__getitem__(index)
            img2_data, _ = self.dataset.__getitem__(index)
            img3_data, _ = self.dataset.__getitem__(index)
            
            img_data = self.image_transforms(img_data)
            img2_data = self.image_transforms(img2_data)
            img3_data = self.image_transforms(img3_data)
            # build this wrapper such that we can return index
            data = [index, img_data.float(), img2_data.float(), 
                    img3_data.float(), label]

            return tuple(data)
    def __len__(self):
        return len(self.dataset)


class CIFAR10Corners(data.Dataset):
    '''Creates a four-corners mosaic of different CIFAR images.'''
    
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
        self,
        root=DATA_ROOTS['cifar10'],
        train=True,
        image_transforms=None,
    ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root,
            train=train,
            download=True # Don't apply transformations yet
        )
        self.train = train
        self.transforms = image_transforms

    def get_random_cifar(self):
        idx = random.randint(0, len(self.dataset) - 1)
        return self.dataset[idx][0]


    def paste_random_cifar_square(self, base_img, x, y):
        img = self.get_random_cifar()
        img_crop = img.crop((x, y, x + 16, y + 16))
        base_img.paste(img_crop, (x, y))
        return base_img


    def get_cifar_corners(self):
        base_img = self.get_random_cifar()
        base_img = self.paste_random_cifar_square(base_img, 16, 0)
        base_img = self.paste_random_cifar_square(base_img, 16, 16)
        base_img = self.paste_random_cifar_square(base_img, 0, 16)
        return base_img


    def __getitem__(self, index):
        if not self.train:
            img_data, label = self.dataset.__getitem__(index)
            img2_data, _ = self.dataset.__getitem__(index)
            # build this wrapper such that we can return index
            data = [index, self.transforms(img_data).float(), self.transforms(img2_data).float(), label, label]
        else:
            img_data = self.get_cifar_corners()
            img2_data = img_data
            # No labels for pano.
            data = [index, self.transforms(img_data).float(), self.transforms(img2_data).float(), 0, 0]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)