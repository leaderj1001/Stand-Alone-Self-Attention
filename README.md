# Implementing Stand-Alone Self-Attention in Vision Models using Pytorch (13 Jun 2019)
  - [Stand-Alone Self-Attention in Vision Models paper](https://arxiv.org/abs/1906.05909)
  - Author:
    - Prajit Ramachandran (Google Research, Brain Team)
    - Niki Parmar (Google Research, Brain Team)
    - Ashish Vaswani (Google Research, Brain Team)
    - Irwan Bello (Google Research, Brain Team)
    - Anselm Levskaya (Google Research, Brain Team)
    - Jonathon Shlens (Google Research, Brain Team)
  - Awesome :)

## Method
  - **Attention Layer**<br>
    <img src='https://user-images.githubusercontent.com/22078438/60595767-a7821280-9de2-11e9-891a-38dd49c25377.PNG' height='300' width='500'><br>
    - Equation 1:<br><br>
    ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/22078438/60596611-5a06a500-9de4-11e9-9116-4d1641f4b84d.gif)<br><br>
  - **Relative Position Embedding**<br>
    <img src='https://user-images.githubusercontent.com/22078438/60596076-34c56700-9de3-11e9-9beb-c03f8842d8b8.PNG' height='400'><br>
    - The row and column offsets are associated with an embedding ![CodeCogsEqn (3)](https://user-images.githubusercontent.com/22078438/60596887-da2d0a80-9de4-11e9-936d-73f5159aa8b9.gif) and ![CodeCogsEqn (4)](https://user-images.githubusercontent.com/22078438/60596947-f9c43300-9de4-11e9-8630-7f4674c7f0c8.gif) respectively each with dimension ![CodeCogsEqn (5)](https://user-images.githubusercontent.com/22078438/60597007-182a2e80-9de5-11e9-9d44-c383e19f55b9.gif). The row and column offset embeddings are concatenated to form ![CodeCogsEqn (6)](https://user-images.githubusercontent.com/22078438/60597062-38f28400-9de5-11e9-8010-ee05512222b5.gif). This spatial-relative attention is now defined as below equation.
    - Equation 2:<br><br>
    ![CodeCogsEqn (7)](https://user-images.githubusercontent.com/22078438/60597197-7b1bc580-9de5-11e9-890a-6225db5a1108.gif)
      
    - I refer to the following paper when implementing this part.
      - [Attention Augemnted Convolutional Networks paper](https://arxiv.org/abs/1904.09925)
      
  1. Replacing Spatial Convolutions<br>
    - A **2 × 2 average pooling with stride 2 operation follows the attention layer** whenever spatial downsampling is required.
    - This work applies the transform on the ResNet family of architectures. **The proposed transform swaps the 3 × 3 spatial convolution with a self-attention layer as defined in Equation 3.**
  2. Replacing the Convolutional Stem<br>
    - The initial layers of a CNN, sometimes referred to as **the stem, play a critical role** in learning local features such as edges, which later layers use to identify global objects.
    - The stem performs **self-attention within each 4 × 4 spatial block of the original image, followed by batch normalization and a 4 × 4 max pool operation.**

## Experiments
### Setup
  - Spatial extent: 7
  - Attention heads: 8
  - Layers:
    - ResNet 26: [1, 2, 4, 1]
    - ResNet 38: [2, 3, 5, 2]
    - ResNet 50: [3, 4, 6, 3]
    
| Datasets | Model | Accuracy | Parameters (My Model, Paper Model)
| :---: | :---: | :---: | :---: |
CIFAR-10 | ResNet 26 | 90.94% | 8.30M, -
CIFAR-10 | Naive ResNet 26 | 94.29% | 8.74M
CIFAR-10 | ResNet 26 + stem | 90.22% | 8.30M, -
CIFAR-10 | ResNet 38 (WORK IN PROCESS) | 89.46% | 12.1M, -
CIFAR-10 | Naive ResNet 38 | 94.93% | 15.0M
CIFAR-10 | ResNet 50 (WORK IN PROCESS) | | 16.0M, -
IMAGENET | ResNet 26 (WORK IN PROCESS) | | 10.3M, 10.3M
IMAGENET | ResNet 38 (WORK IN PROCESS) | | 14.1M, 14.1M
IMAGENET | ResNet 50 (WORK IN PROCESS) | | 18.0M, 18.0M

## Usage

## Requirements
  - torch==1.0.1

## Todo
  - Experiments
  - IMAGENET
  - Review relative position embedding, attention stem
  - Code Refactoring

## Reference
  - [ResNet Pytorch CIFAR](https://github.com/kuangliu/pytorch-cifar)
  - [ResNet Pytorch](https://github.com/pytorch/vision/blob/8350645b680b5dc0ef347de82deea5ae3f8ca3dc/torchvision/models/resnet.py)
  - Thank you :)
