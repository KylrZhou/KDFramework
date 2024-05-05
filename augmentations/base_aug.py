from utils import AUGMENTATION

import torchvision.transforms as transforms

@AUGMENTATION.register()
def RandomCrop(**kwargs):
    return transforms.RandomCrop(**kwargs)

@AUGMENTATION.register()
def RandomHorizontalFlip(**kwargs):
    return transforms.RandomHorizontalFlip(**kwargs)

@AUGMENTATION.register()
def RandomRotation(**kwargs):
    return transforms.RandomRotation(**kwargs)

@AUGMENTATION.register()
def ToTensor():
    return transforms.ToTensor()

@AUGMENTATION.register()
def Normalize(**kwargs):
    return transforms.Normalize(**kwargs)