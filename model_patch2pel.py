# -*- coding: utf-8 -*-
"""
dl_demosaic model
=====================
"""

########################################################################
# Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

# 3x3 Convolution
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=False)



class Net_pel(nn.Module):
    def __init__(self):
        super(Net_pel, self).__init__()
        self.ch_l1 = 3  # 1
        self.ch_l2 = 64 # 64
        self.ch_l3 = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.ch_l1, self.ch_l2, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.ch_l2, self.ch_l2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ch_l2),
            nn.ReLU(),
            nn.MaxPool2d((2,2)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.ch_l2, self.ch_l3, kernel_size=3, padding=1))
        self.fc1 = nn.Sequential(
            nn.Linear(1*1*self.ch_l3, 3),
            nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x) # use 12~14 layers
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer2(x)

#        x = self.layer2(x)
#        x = self.layer2(x)
#        x = self.layer2(x)
#        x = self.layer2(x)
#
#        x = self.layer2(x)
#        x = self.layer2(x)
#        x = self.layer2(x)
#        x = self.layer2(x)

        x = self.layer3(x)
#        x = x.view(-1, self.num_flat_features(x))  # This FC is hard to train
#        x = self.fc1(x)
        x = torch.squeeze(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features