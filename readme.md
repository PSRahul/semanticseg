
# Semantic Segmentation on The Oxford-IIIT Pet Dataset

## Dataset
https://www.robots.ox.ac.uk/~vgg/data/pets/
O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
Cats and Dogs  
IEEE Conference on Computer Vision and Pattern Recognition

## Model
A U-Net based neural network was trained from scratch using Pytorch Lightning wrapper over the Pytorch Framework.

The dropout probability was varied to optimise the network.

### Optimiser 
Adam with the default learning rate of 10e-3.

### Loss   
Cross Entropy Loss of classified pixel labels and ground data.

### Callbacks
Early Stopping , Best Validation loss checkpoints.

## Datasets

Datasets folder contain the Train|Validation|Test split on the dataset.

## Test Examples

The **tester.ipynb** contain the testing notebook and can be used to generate more test samples. 
Some of the best test examples are in Sample Images folder.

![Alt text](sample_images/1.png?raw=true "Sample 1")
![Alt text](sample_images/2.png?raw=true "Sample 1")

## Tensorboard 

Loss values can be found in the tensorboard logs.
To run
> tensorboard --logdir runs/

## Model Checkpoints

Early Stopping Checkpoint are saved in **CKPT** folder
