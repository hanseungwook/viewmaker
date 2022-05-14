import torch
import random
from torchvision import transforms
from PIL import ImageFilter, Image

from src.datasets.cifar10 import CIFAR10, CIFAR10Corners
from src.datasets.cifar100 import CIFAR100, CIFAR100Corners
from src.datasets.tinyimagenet import TinyImageNet, TinyImageNetCorners

from src.datasets.meta_datasets.aircraft import Aircraft
from src.datasets.meta_datasets.cu_birds import CUBirds
from src.datasets.meta_datasets.dtd import DTD
from src.datasets.meta_datasets.fashionmnist import FashionMNIST
from src.datasets.meta_datasets.fungi import Fungi
from src.datasets.meta_datasets.mnist import MNIST
from src.datasets.meta_datasets.mscoco import MSCOCO as MSCOCO2
from src.datasets.meta_datasets.traffic_sign import TrafficSign
from src.datasets.meta_datasets.vgg_flower import VGGFlower
from src.datasets.data_statistics import get_data_mean_and_stdev

DATASET = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'cifar100_corners': CIFAR100Corners,
    'cifar10_corners': CIFAR10Corners,
    'tinyin': TinyImageNet,
    'tinyin_corners': TinyImageNetCorners,
    'meta_aircraft': Aircraft,
    'meta_cu_birds': CUBirds,
    'meta_dtd': DTD,
    'meta_fashionmnist': FashionMNIST,
    'meta_fungi': Fungi,
    'meta_mnist': MNIST,
    'meta_mscoco': MSCOCO2,
    'meta_traffic_sign': TrafficSign,
    'meta_vgg_flower': VGGFlower,
}


def zscore_image(img_tensor):
    img_tensor -= img_tensor.mean([-1, -2], keepdim=True)
    img_tensor /= img_tensor.std([-1, -2], keepdim=True)
    return img_tensor

def get_image_datasets(
        dataset_name,
        default_augmentations='none',
    ):
    load_transforms = TRANSFORMS[default_augmentations]
    train_transforms, test_transforms = load_transforms(
        dataset=dataset_name, 
    )
    train_dataset = DATASET[dataset_name](
        train=True,
        image_transforms=train_transforms
    )
    val_dataset = DATASET[dataset_name](
        train=False,
        image_transforms=test_transforms,
    )
    return train_dataset, val_dataset


def load_image_transforms(dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.ToTensor()
        test_transforms = transforms.ToTensor()
    elif 'tinyin' in dataset:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms

def load_a1_transforms(dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transforms = transforms.ToTensor()
    elif 'tinyin' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.ToTensor(),
            # transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms

def load_a2_transforms(dataset):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transforms = transforms.ToTensor()
    elif 'tinyin' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.ToTensor(),
            # transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


def load_default_transforms(dataset):
    if 'cifar10' == dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446],
                                std=[0.247, 0.243, 0.261]),
        ])
    elif 'cifar100' == dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])
    elif 'tinyin' == dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
        ])
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        mean, std = get_data_mean_and_stdev(dataset)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return None, None
    
    return train_transforms, test_transforms


def load_default_unnorm_transforms(dataset, **kwargs):
    if 'cifar' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.ToTensor()
    elif dataset in ['mscoco'] or 'meta_' in dataset:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        return None, None

    return train_transforms, test_transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


TRANSFORMS = {
    True: load_default_transforms,
    False: load_image_transforms,
    'all': load_default_transforms,
    'all_unnorm': load_default_unnorm_transforms,
    'none': load_image_transforms, 
    'a1': load_a1_transforms,
    'a2': load_a2_transforms,
}
