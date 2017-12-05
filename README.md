# dl_demosaic

* bayer1ch case  
  * weird circle issue at center(wrong color)  
  * need to comapare with bayer3ch case  
    * without weired circle at bayer3ch
  * more epochs?   
* bayer3ch case
  * this can be trained easier than 1ch case.  

## revision list  
 * add model configuration(depth)  
 * remove parameters.py  
 * update main.py and test.py  
 * add training case for another dataset  
 
## like moire, hard traing patches

## hard negative mining
 adjust a ratio of datasets 
 
## weighted least squares for loss function
 is this loss function exist in pytorch ?

https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py  
class _WeightedLoss(_Loss):  

https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547/3  

## add extra label for hard patches as another loss
