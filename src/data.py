import torch, torchvision
import matplotlib.pyplot as plt
import os
import random

class ImageDataset:
    def __init__(self, root_directory, transform = None):
        self.root_directory = root_directory
        self.dataset = torchvision.datasets.ImageFolder(root = root_directory, 
                                                      transform = transform)
        self.class_labels = list(self.dataset.class_to_idx.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, idx_label, str_label = sample[0], sample[1], self.class_labels[int(sample[1])]
        
        sample = {'image': image,
                  'idx_label': torch.tensor(idx_label),
                  'str_label':str_label
                    }
        
        return sample
    
    def display_images(self, rows, cols, num_images, figsize, fontdict):
        image_idxs = random.sample(range(20_000), num_images)

        for i, image_idx in enumerate(image_idxs):
            sample = self.__getitem__(image_idx)
            img = sample['image'].permute(1, 2,0)

            plt.rcParams["figure.figsize"] = figsize
            plt.subplot(rows, cols, i + 1)

            plt.title(label = str(sample['str_label']) + "\n" + str(torchvision.transforms.functional.get_image_size(sample['image'])),
                    fontdict = fontdict)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(img)


def calculate_channel_means(dataset, label):
    #calculates and returns per channel mean of dataset containing image data
    mean = 0
    n_images = dataset.__len__()
    for i in range(n_images):
        mean += torch.mean(dataset.__getitem__(i)[label].float(), dim= [1,2])
    return mean/n_images

def calculate_channel_sds(dataset, label):
    #calculates and returns per channel standard deviation of dataset containing image data
    var = 0
    n_images = dataset.__len__()
    for i in range(n_images):
        var += torch.var(input = dataset.__getitem__(i)[label].float(), dim = [1,2])
    return torch.sqrt(var/n_images)