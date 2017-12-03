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

