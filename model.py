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


# 3x3 Convolution
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=False)



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
