# -*- coding: utf-8 -*-
"""
Training a classifier
=====================
This is it. You have seen how to define neural modelworks, compute loss and make
updates to the weights of the modelwork.
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
Imagemodel, CIFAR10, MNIST, etc. and data transformers for images, viz.,
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
2. Define a Convolution Neural modelwork
3. Define a loss function
4. Train the modelwork on the training data
5. Test the modelwork on the test data
1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
import cv2
from utils import *
from model import *
from parameters import *


# Image Preprocessing

transform = transforms.Compose([
   transforms.Scale(32),
   transforms.RandomHorizontalFlip(),
   transforms.RandomCrop(patch_size),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # This normalize is needed?


# CIFAR-10 Dataset
testset = dsets.CIFAR10(root='./data/',
                              train     = False,
                              transform = transform)

testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size = batch_size,
                                          shuffle    = False)


########################################################################
# 2. Define a Convolution Neural modelwork
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural modelwork from the Neural modelworks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

model = Net()

# model.load_state_dict(torch.load('./model.pkl'))
# model.load_state_dict(torch.load('./model_weights/model_p32.pkl'))

checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])


if torch.cuda.is_available():
    model.cuda()



########################################################################
# 5. Test the modelwork on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the modelwork for 2 passes over the training dataset.
# But we need to check if the modelwork has learnt anything at all.
#
# Let us look at how the modelwork performs on the whole dataset.


# show images
fig, axs = plt.subplots(2,2)

for data in testloader:
    images, labels = data

    # Remosaic : RGB to bayer
    i_bayer = remosaic(images, bayer_type)

    if torch.cuda.is_available():
        i_bayer = Variable(i_bayer).cuda()
    else:
        i_bayer = Variable(i_bayer)


    outputs = model(i_bayer)
    predicted_dem = outputs.data
    i_bayer  = i_bayer.data

    # clip
    predicted_dem[predicted_dem < -1] = -1
    predicted_dem[predicted_dem >  1] =  1

    # demosaic with CV2
    bayer_1ch = remosaic(images, 0)  # 1ch bayer
    dem_cv2 = demosaic_cv2(bayer_1ch, 0)

    #
    bayer_rgb = remosaic(images, 1)  # 3ch bayer

    images = unnormalize(images.cpu())
    i_bayer = unnormalize(bayer_rgb.cpu())
    predicted_dem = unnormalize(predicted_dem.cpu())
    dem_alg = unnormalize(dem_cv2.cpu())

    # show images
    # fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(images)
    axs[1,0].imshow(i_bayer)
    axs[0,1].imshow(predicted_dem)
    axs[1,1].imshow(dem_alg)

    plt.pause(3)
