import torch

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