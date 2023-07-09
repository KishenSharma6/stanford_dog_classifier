import torch, torchvision

import os

class ImageDataset:
    def __init__(self, root_directory, transform = None):
        self.root_directory = root_directory
        self.dataset = torchvision.datasets.ImageFolder(root = root_directory, 
                                                      transform = transform)
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, label = sample[0], sample[1]
        
        sample = {'image': image,
                  'label': torch.tensor(label)}
        
        return sample

