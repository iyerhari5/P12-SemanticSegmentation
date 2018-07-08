# Semantic Segmentation
### Introduction
The goal of the project is to use a Fully Convolutional Network (FCN) to perform semantic segmentation to label the drivable road pixels
in images.

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Description

#### Network architecture
We use a pre-trained VGG-16 network. The network is converted to an FCN by converting the final fully connected layers to a 1x1 convolution. The depth
is set to 2 as the desired labels here are "road" and "no-road".  We further improve the performance by skip connections between layers further back
in the network. For example layer 7 is 1x1 convolved and upsampled before being added to the 1x1 convolution results from Layer 4. L2 regularization is used
with each of the 1x1 convultional layers as well as the deconvolution (upsampling) layers.

#### Optimizer
We use the Adam optimizer. The loss that is minimized is the cross entropy of the binary labels.

#### Hyperparameters 
We use 50 epochs with a batch size of 5. 

#### Results
[image1]: ./Images/umm_000033.png
[image2]: ./Images/umm_000079.png
[image3]: ./Images/umm_000001.png
[image4]: ./Images/um_000017.png
[image5]: ./Images/um_000030.png
[image6]: ./Images/um_000031.png

The results are reasonably good for most of the images. For a few  images, parts of the road are missing. Here are some sample results.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]



