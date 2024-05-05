from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from utils import DATASETTRAIN
from utils import DATASETVAL
from utils import build_augmentation_pipeline

@DATASETTRAIN.register()
def CIFAR100_train(root, batch_size, num_workers, shuffle, augmentations=None):
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    if augmentations is not None:
        augmentations = build_augmentation_pipeline(augmentations)
    dataset = CIFAR100(root=root, train=True, download=True, transform=augmentations)
    dataset = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return dataset

@DATASETVAL.register()
def CIFAR100_val(root, batch_size, num_workers, shuffle, augmentations=None):
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    if augmentations is not None:
        augmentations = build_augmentation_pipeline(augmentations)
    dataset = CIFAR100(root=root, train=False, download=True, transform=augmentations)
    dataset = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return dataset