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

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
from utils import *
from model import *
from parameters import *


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-batch_size', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-patch_size', '--patch-size', default=32, type=int,
                    metavar='N', help='input-patch size (default: 32)')
parser.add_argument('--epochs', default=64, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_min_loss = 1

def main():
    global args, best_min_loss
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Scale(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.patch_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # This normalize is needed?

    # CIFAR-10 Dataset
    trainset = dsets.CIFAR10(root='./data/',
                             train=True,
                             transform=transform,
                             download=True)

    # validset = dsets.CIFAR10(root='./data/',
    #                              train     = False,
    #                              transform = transforms.ToTensor())

    validset = dsets.CIFAR10(root='./data/',
                             train=False,
                             transform=transform)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    ########################################################################
    # Let us show some of the training images, for fun.

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    print('data preparation completed')

    ########################################################################
    # 2. Define a Convolution Neural modelwork
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural modelwork from the Neural modelworks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).

    model = Net()
    if torch.cuda.is_available():
        model.cuda()

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum
    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_min_loss = checkpoint['best_min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        loss_val = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = loss_val < best_min_loss
        best_min_loss = min(loss_val, best_min_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'best_min_loss': best_min_loss,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Remosaic : RGB to bayer
        i_bayer = remosaic(input, bayer_type)

        # ground truth : noisy image - clean image(noise)
        ground_truth = input

        # wrap them in Variable
        if torch.cuda.is_available():
            i_bayer, ground_truth = Variable(i_bayer).cuda(), Variable(ground_truth).cuda()
        else:
            i_bayer, ground_truth = Variable(i_bayer), Variable(ground_truth)

        # compute output
        output = model(i_bayer)
        loss = criterion(output, ground_truth)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        # Remosaic : RGB to bayer
        i_bayer = remosaic(input, bayer_type)

        # ground truth : noisy image - clean image(noise)
        ground_truth = input

        # wrap them in Variable
        if torch.cuda.is_available():
            i_bayer, ground_truth = Variable(i_bayer).cuda(), Variable(ground_truth).cuda()
        else:
            i_bayer, ground_truth = Variable(i_bayer), Variable(ground_truth)

        # compute output
        output = model(i_bayer)
        loss = criterion(output, ground_truth)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))
    return losses.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()