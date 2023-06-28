import torch
from torchvision import transforms

from PIL import Image

import os

class ImageDataset:
    def __init__(self, img_paths, labels, transformation = None):
        self.img_paths = img_paths
        self.labels = labels
        self.transformation = transformation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        image = transforms.ToTensor(image)

        label = self.labels[idx]

        if self.transformation:
            image = self.transformation(image)

        sample = {'image': image,
                  'label': label}
        
        return sample



def lowercase_directories(path):
    #renames subdirectories in path to all lowercase
    for file in os.listdir(path):
        os.rename(path + file, path + file.lower())

    print("subdirectories have been converted to lowercase")
