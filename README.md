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
* ground_truth = i_bayer - inputs  
* predicted_dem = i_bayer - outputs  

## Relu -> leacky Relu?
* ?? 



## Patch to pixel  
1. add extra fully con. layer  
2. add pooling, fc, relu layer(low compexity)  

## gaussian_dem_rgb : input extra ch  
* use extra rgb 3ch(lpf-dem)
* + residual learning
  * (i_bayer_rgb - rgb_dem) - inputs

## add training case for another dataset  
* stl10, cifar100, (Coco)
* demosaicnet


## using LBCNN

## dem is only prediction of direction problem? V or H label  




   
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

### add extra label for hard patches as another loss
