# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:23:48 2021

@author: csyu
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Data transform
data_transform_Resize = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Training Data


class MNISTDataset(Dataset):
    def __init__(self, path):
        # self.x = x
        # self.y = y
        # self.transform = transform
        dataset = ImageFolder(
            path, transform=data_transform_Resize)
        self.imgs = dataset.imgs
        self.nb_classes = len(dataset.classes)

    def __len__(self):
        return int(len(self.imgs))

    def __getitem__(self, index):

        # image = Image.fromarray(self.x[index][0]*255).convert('RGB')
        # image = self.transform(image)

        path, label = self.imgs[index]

        image = Image.open(path).convert('RGB')
        image = data_transform_Resize(image)

        y = torch.zeros(int(self.nb_classes))
        y[label] = 1

        return image, y

# Testing Data


def TestDataset(image_array, select):
    image = Image.fromarray(image_array*255).convert('RGB')
    if select < 3:
        image = np.array(data_transform_Resize(image))
    else:
        image = np.array(data_transform_Resize(image))
    return image
