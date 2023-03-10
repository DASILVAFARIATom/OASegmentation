## **Simple UNET model** 
This segmentation model is a basic UNET [1] architecture :    
<img src="https://github.com/DASILVAFARIATom/OASegmentation/blob/master/Images/SimpleUnet.png" width="400">

The block "ConvBlock" is made of two 2D convolutional layers (with Batch norm and ReLU activation).  

Training results are available in this [notebook](https://github.com/DASILVAFARIATom/OASegmentation/blob/master/SimpleUnet/main_ntbk.ipynb). The used training parameters are registered in this [config](https://github.com/DASILVAFARIATom/OASegmentation/blob/master/SimpleUnet/CONFIG.py) file.  

The performances of this architecture are quite low. The resulting segmentation maps are not as accurate as the ground truth. However, we can see that some missing vessels in the GT are actually detected by the model. 

## References :
[1] Ronneberger et al. **_U-Net: Convolutional Networks for Biomedical Image Segmentation_** (https://arxiv.org/abs/1505.04597)
