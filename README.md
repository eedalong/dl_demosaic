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


## revision list   
 * ~~~remove parameters.py~~~  
 * ~~~update main.py and test.py~~~  
 * add model configuration(depth) 
 * add training case for another dataset  
 
## like moire, hard training patches

## hard negative mining
 adjust a ratio of datasets 
 
## weighted least squares for loss function
 is this loss function exist in pytorch ?

https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py  
class _WeightedLoss(_Loss):  

https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547/3  

## add extra label for hard patches as another loss
