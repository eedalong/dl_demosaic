# dl_demosaic

* bayer1ch case  
  * weird circle/square issues at center(wrong color)  
  * need to comapare with bayer3ch case  
    * Bayer3ch has no weired center issues.
  * more epochs?   
  * 1ch to 3ch transform without position info. is hard to train.   
* bayer3ch case
  * It can be trained easier than 1ch case.  
* opencv demosaic
  * https://docs.opencv.org/3.0-beta/modules/cudaimgproc/doc/color.html  
    * cvtColor  
    * demosaicing  
    * bilinear interpolation : COLOR_BayerBG2BGR , COLOR_BayerGB2BGR , ...    
    * Malvar-He-Cutler algorithm : COLOR_BayerBG2BGR_MHT, COLOR_BayerGB2BGR_MHT, ...   
      * Pascal Getreuer, Malvar-He-Cutler Linear Image Demosaicking, Image Processing On Line, 2011  
* 4 layer => 8 layers
  * It may be hard to train due to deep layers.  
  * This network seems to need specific schemes to train deeper network(ex. dropout?, batchnorm?)  

* ADAM is better than SGD  

* Batch size : Bigger is better.  

## residual learning
* Loss : 0.0016(slightly better), P32-B128-L4-C64   
* more easily trained  
* ground_truth  = i_bayer - inputs  
* predicted_dem = i_bayer - outputs  

## add training case for another dataset  
* stl10

## add tensorboard 
 * https://github.com/chson0316/dl_info/blob/master/pytorch_visualization.md


## gaussian_dem_rgb : input extra ch  
* use extra rgb 3ch(lpf-dem)
* residual learning
* net.input : rgb_dem - i_bayer_3ch  (or i_bayer_3ch, concat(i_bayer_3ch, rgb_dem))  
* ground_truth : rgb_dem - inputs (or rgb_dem - i_bayer - inputs)  
* predicted_dem: rgb_dem - net.outputs  
* best case  
  * in:rgb_dem - i_bayer_3ch, gt: rgb_dem - inputs  
* plus detail_layer info.?


## Mixed Loss function  : NVIDIA Works
* http://www.mit.edu/~hangzhao/  
### Loss Functions for Neural Networks for Image Processing
* http://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks  
* http://on-demand.gputechconf.com/gtc/2017/presentation/s7447-orazio-gallo-image-restoration-with-neural-networks.pdf   
* http://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/comparison_tci.pdf    
* https://arxiv.org/pdf/1511.08861.pdf  
* http://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf  
* https://pdfs.semanticscholar.org/0434/70c64627c074e1a7634a2508ee7e4ec98cd7.pdf  

### nvidia research
https://www.nvidia.com/en-us/research/  
https://github.com/NVlabs  

### pytorch : MS-SSIM
https://github.com/chson0316/pytorch-image-comp-rnn  
https://github.com/Po-Hsun-Su/pytorch-ssim  
https://github.com/chainer/chainer/issues/2503  

### pytorch custom loss function
https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/8  
https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/13  
https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568  

```

SECTION 5 - CUSTOM LOSS FUNCTIONS

Now that we have our model all in place we can load anything and create any architecture we want. That leaves us with 2 important components in any pipeline - Loading the data, and the training part. Let's take a look at the training part. The two most important components of this step are the optimizer and the loss function. The loss function quantifies how far our existing model is from where we want to be, and the optimizer decides how to update parameters such that we can minimize the loss.

Sometimes, we need to define our own loss functions. And here are a few things to know about this -

    custom Loss functions are defined using a custom class too. They inherit from torch.nn.Module just like the custom model.
    Often, we need to change the dimenions of one of our inputs. This can be done using view() function.
    If we want to add a dimension to a tensor, use the unsqueeze() function.
    The value finally being returned by a loss function MUST BE a scalar value. Not a vector/tensor.
    The value being returned must be a Variable. This is so that it can be used to update the parameters. The best way to do so is to just make sure that both x and y being passed in are Variables. That way any function of the two will also be a Variable.
    A Pytorch Variable is just a Pytorch Tensor, but Pytorch is tracking the operations being done on it so that it can backpropagate to get the gradient.

Here I show a custom loss called Regress_Loss which takes as input 2 kinds of input x and y. Then it reshapes x to be similar to y and finally returns the loss by calculating L2 difference between reshaped x and y. This is a standard thing you'll run across very often in training networks.

Consider x to be shape (5,10) and y to be shape (5,5,10). So, we need to add a dimension to x, then repeat it along the added dimension to match the dimension of y. Then, (x-y) will be the shape (5,5,10). We will have to add over all three dimensions i.e. three torch.sum() to get a scalar.
```



## Depthwise seperable convolution 
* depthwise + point-wise cnn(=1x1 conv.)
* spatial/cross-channel correlation  
* before or after 1x1 conv   
* BN+relu are needed between depthwise and pointwise.  


## using LBCNN


## add extra label for hard patches as another loss
* It can be likely to auxiliary classifier in GoogLeNet at training phase. 
* remove this at test phase  



## use Y-C instead of RGB


## hard negative mining

## RWB, RGBIR


## use dropout in convnet  

## Distilling the Knowledge in a Neural Network
* create more precisely true labels 
* A huge network is training a smaller network

## Patch to pixel  
1. add extra fully con. layer  
* fail to train and high complexity
2. add pooling, fc, relu layer(low compexity)  
* https://github.com/chson0316/dl_info/blob/master/pytorch_classify_summary.md  





   
## how to train   
   
### like moire, hard training patches

### hard negative mining
* adjust a ratio of datasets   
* Using PSNR, SSIM, HDR-VDP2, save patches(hard) to image files with labels.

 
### weighted least squares for loss function
* is this loss function exist in pytorch ?
* https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py  
* class _WeightedLoss(_Loss):  
* https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547/3  

