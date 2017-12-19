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

def remosaic(i_rgb, type):
    batch_size, ch, patch_height, patch_width = i_rgb.shape
    bayer = torch.zeros(batch_size, 1, patch_height, patch_width)

    bayer[:, 0,  ::2,  ::2] = i_rgb[:, 1,  ::2,  ::2]  # G
    bayer[:, 0,  ::2, 1::2] = i_rgb[:, 0,  ::2, 1::2]  # R
    bayer[:, 0, 1::2,  ::2] = i_rgb[:, 2, 1::2,  ::2]  # B
    bayer[:, 0, 1::2, 1::2] = i_rgb[:, 1, 1::2, 1::2]  # G

    bayer_rgb = torch.zeros(batch_size, 3, patch_height, patch_width)

    bayer_rgb[:, 1,  ::2,  ::2] = bayer[:, 0,  ::2,  ::2]  # G
    bayer_rgb[:, 0,  ::2, 1::2] = bayer[:, 0, ::2, 1::2]  # R
    bayer_rgb[:, 2, 1::2,  ::2] = bayer[:, 0, 1::2,  ::2]  # B
    bayer_rgb[:, 1, 1::2, 1::2] = bayer[:, 0, 1::2, 1::2]  # G

    if type == 0:
        o_bayer = bayer
    else:
        o_bayer = bayer_rgb


    return o_bayer


import torchvision
def unnormalize(images):
    img = torchvision.utils.make_grid(images)
    img   = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg



import cv2
def demosaic_cv2(bayer, bseq):
    batch_size, ch, patch_height, patch_width = bayer.shape
    patch_size  = patch_width

    bseq = cv2.COLOR_BAYER_GR2BGR

    bayer_np = bayer.numpy()
    # unnormalize
    bayer_np = ((bayer_np/2+0.5)*255)

    # to uint8
    bayer_u8 = bayer_np.astype('uint8')
    debayer = np.zeros([batch_size, patch_size, patch_size, 3])

    for i in range(batch_size):
        debayer[i] = cv2.cvtColor(bayer_u8[i,0], cv2.COLOR_BAYER_GR2BGR)  # bilinear intp.
        # debayer[i] = cv2.demosaicing(bayer_u8[i,0], cv2.COLOR_BAYER_GR2BGR_EA)  # so so
        # debayer[i] = cv2.demosaicing(bayer_u8[i, 0], cv2.COLOR_BAYER_GR2BGR_VNG)  # bad

    # normalized
    debayer = debayer.astype('float32')/255
    debayer = (debayer-0.5)*2

    # to torch.FloatTensor
    dem0 = torch.from_numpy(debayer)

    # b, h, w, ch =>  b, ch, h, w
    dem = torch.zeros(batch_size, 3, patch_size, patch_size)
    dem[:,0,:,:] = dem0[:,:,:,0]
    dem[:,1,:,:] = dem0[:,:,:,1]
    dem[:,2,:,:] = dem0[:,:,:,2]

    return dem

"""
import numpy as np
from scipy import signal



out = torch.zeros(batch_size, 3, patch_size, patch_size)

for i in range(batch_size):

    img = bayer[i] / 2 + 0.5
    x = img.numpy()


    o_dem = np.zeros((3, patch_size, patch_size))

    w_k = np.array([[1, 4, 6, 4, 1],
                    [4,16,24,16, 4],
                    [6,24,36,24, 6],
                    [4,16,24,16, 4],
                    [1, 4, 6, 4, 1]],
                   dtype='float')/256


    w0 = np.zeros((5,5))
    w1 = np.zeros((5,5))
    w2 = np.zeros((5,5))
    w3 = np.zeros((5,5))

    w0[ ::2,  ::2] = w_k[ ::2,  ::2]
    w1[ ::2, 1::2] = w_k[ ::2, 1::2]
    w2[1::2,  ::2] = w_k[1::2,  ::2]
    w3[1::2, 1::2] = w_k[1::2, 1::2]


    o0 = signal.convolve2d(x, w0, boundary='symm', mode='same')
    o1 = signal.convolve2d(x, w1, boundary='symm', mode='same')
    o2 = signal.convolve2d(x, w2, boundary='symm', mode='same')
    o3 = signal.convolve2d(x, w3, boundary='symm', mode='same')

    o_r = o1*4
    o_g = (o0*4+o3*4)/2
    o_b = o2*4

    o_dem[0] = o_r
    o_dem[1] = o_g
    o_dem[2] = o_b

    out[i] = torch.from_numpy(o_dem)

imshow(torchvision.utils.make_grid(out), 3)
