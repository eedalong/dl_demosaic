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
patch_size = 31
batch_size = 64 # 32
sigma      = 128/255

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

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# Let us show some of the training images, for fun.


# get some random training images
dataiter       = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print(images.size())
print(images.type())

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# add noise gaussian random
im_noise = add_noise(images, sigma)

# show images
imshow(torchvision.utils.make_grid(im_noise), 1)
print('data preparation completed')


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
        self.ch_l2 = 128 # 64
        self.ch_l3 = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.ch_l2, kernel_size=3, padding=1),
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

for epoch in range(16*4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # add noise gaussian random
        i_noisy = add_noise(inputs, sigma)

        # ground truth : noisy image - clean image(noise)
        ground_truth = i_noisy - inputs

        # wrap them in Variable
        if torch.cuda.is_available():
            i_noisy, ground_truth = Variable(i_noisy).cuda(), Variable(ground_truth).cuda()
        else:
            i_noisy, ground_truth = Variable(i_noisy), Variable(ground_truth)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(i_noisy)
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




########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:


# add noise gaussian random
noisy_image = add_noise(images, sigma)

if torch.cuda.is_available():
    noisy_image = Variable(noisy_image).cuda()
else:

    noisy_image = Variable(noisy_image)


outputs = net(noisy_image)
predicted_noise = outputs.data
noisy_image  = noisy_image.data

out_denoised = noisy_image - predicted_noise
# clip
out_denoised[out_denoised<-1] = -1
out_denoised[out_denoised> 1] =  1


# show images
imshow(torchvision.utils.make_grid(images.cpu()), 0)
print('show inpute test images')

imshow(torchvision.utils.make_grid(noisy_image.cpu()), 1)
print('show noisy images')

imshow(torchvision.utils.make_grid(predicted_noise.cpu()), 2)
print('show predicted noises')

imshow(torchvision.utils.make_grid(out_denoised.cpu()), 3)
print('show output de-noised images')


# Save the Model
torch.save(net.state_dict(), 'net.pkl')

