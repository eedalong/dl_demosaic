import torch

# functions to show an image
import matplotlib.pyplot as plt
import numpy as np


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() # interactive mode


def imshow(img, num=0):
    img   = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(num)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    plt.pause(1)




def add_noise(data, sigma):
    # add noise gaussian random rand(im.shape)
    # sigma : noise standard deviation
    # target image pixel value range is 0-1

    data = data / 2 + 0.5  # unnormalize

    noise      = sigma * torch.rand(data.size())
    noisy_data = data + noise

    # negative to zero
    noisy_data[noisy_data < 0] = 0
    noisy_data[noisy_data > 1] = 1

    noisy_data = (noisy_data - 0.5) * 2  # normalized

    return noisy_data

def remosaic(i_rgb):
    batch_size, ch, patch_height, patch_width = i_rgb.shape
    bayer = torch.zeros(batch_size, patch_height, patch_width)

    bayer[:,  ::2,  ::2] = i_rgb[:, 1,  ::2,  ::2]  # G
    bayer[:,  ::2, 1::2] = i_rgb[:, 0,  ::2, 1::2]  # R
    bayer[:, 1::2,  ::2] = i_rgb[:, 2, 1::2,  ::2]  # B
    bayer[:, 1::2, 1::2] = i_rgb[:, 1, 1::2, 1::2]  # G

    bayer_rgb = torch.zeros(batch_size, 3, patch_height, patch_width)

    bayer_rgb[:, 1,  ::2,  ::2] = bayer[:,  ::2,  ::2]  # G
    bayer_rgb[:, 0,  ::2, 1::2] = bayer[:,  ::2, 1::2]  # R
    bayer_rgb[:, 2, 1::2,  ::2] = bayer[:, 1::2,  ::2]  # B
    bayer_rgb[:, 1, 1::2, 1::2] = bayer[:, 1::2, 1::2]  # G

    return bayer_rgb


import torchvision
def unnormalize(images):
    img = torchvision.utils.make_grid(images)
    img   = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg
