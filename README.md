# dl_demosaic

1. bayer1ch case  
  * weird circle issue at center(wrong color)  
  * need to comapare with bayer3ch case  



## like moire, hard traing patches

## hard negative mining
 adjust a ratio of datasets 
 
## weighted least squares for loss function
 is this loss function exist in pytorch ?

https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py  
class _WeightedLoss(_Loss):  

https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547/3  

## add extra label for hard patches as another loss
