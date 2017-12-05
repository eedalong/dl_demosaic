# -*- coding: utf-8 -*-
"""
Training a classifier
=====================
This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.
Now you might be thinking,
What about data?
----------------
Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.
-  For images, packages such as Pillow, OpenCV are useful.
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful.
Specifically for ``vision``, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.
This provides a huge convenience and avoids writing boilerplate code.
For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
.. figure:: /_static/img/cifar10.png
   :alt: cifar10
   cifar10
Training an image classifier
----------------------------
We will do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
from utils import *



# paramters
patch_size = 32
batch_size = 64 # 32
sigma      = 32/255
no_epoch   = 64 


# Image Preprocessing

transform = transforms.Compose([
   transforms.Scale(32),
   transforms.RandomHorizontalFlip(),
   transforms.RandomCrop(patch_size),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # This normalize is needed?


# CIFAR-10 Dataset
trainset = dsets.CIFAR10(root='./data/',
                               train     = True,
                               transform = transform,
                               download  = True)

#testset = dsets.CIFAR10(root='./data/',
#                              train     = False,
#                              transform = transforms.ToTensor())

testset = dsets.CIFAR10(root='./data/',
                              train     = False,
                              transform = transform)

# Data Loader (Input Pipeline)
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size = batch_size,
                                           shuffle    = True)

testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size = batch_size,
                                          shuffle    = False)


########################################################################
# Let us show some of the training images, for fun.


# get some random training images
dataiter       = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print('data preparation completed')



########################################################################
# Demosaic, simple gaussian

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

"""



########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# 3x3 Convolution
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=False)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = conv3x3(3, 6)
#         self.conv2 = conv3x3(6, 16)
#         self.fc1 = nn.Linear(16 * patch_size * patch_size, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(-1, 16 * patch_size * patch_size)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ch_l2 = 64 # 64
        self.ch_l3 = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.ch_l2, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.ch_l2, self.ch_l2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch_l2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.ch_l2, self.ch_l3, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x) # use 12~14 layers
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
#       x = self.layer2(x)
        x = self.layer3(x)
        return x


net = Net()
if torch.cuda.is_available():
    net.cuda()


########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize

for epoch in range(no_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data


        # Remosaic : RGB to bayer
        # i_bayer = remosaic(inputs, 1)  # 3ch bayer
        i_bayer = remosaic(inputs, 0)  # 1ch bayer


        # ground truth : noisy image - clean image(noise)
        ground_truth = inputs

        # wrap them in Variable
        if torch.cuda.is_available():
            i_bayer, ground_truth = Variable(i_bayer).cuda(), Variable(ground_truth).cuda()
        else:
            i_bayer, ground_truth = Variable(i_bayer), Variable(ground_truth)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(i_bayer)
        loss = criterion(outputs, ground_truth)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')


# Save the Model
torch.save(net.state_dict(), 'net.pkl')
