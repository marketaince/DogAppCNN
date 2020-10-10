# Dog App CNN
This work is an extension of one of projects required to obtain [Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) by [Udacity](https://www.udacity.com/).

*Submitted: Aug 2019*

## Objective
The purpose of this project is to create a neural network that can classify dog breeds from a provided image.

### Methods Used
* Convolutional Neural Network (**CNN**)
* Transfer Learning - VGG-16

### Technologies
* Python
* PyTorch
* Flask Restful
* Heroku

### Training Data
Data for this project was provided by Udacity. The Training set contains 6680 images of 133 different dog breeds. Validation and Test sets contain 835 and 836 images respectively.

## Project Description
As mentioned above the goal is to create a dog breed classifier. For this purpose a Convolutional Neural Network is a plausible choice. CNN is capable of learning patterns in an image and therefore recognizing differences between different images.

Such classifier usually consists of two parts. Fist part are convolutional layers that are responsible from feature learning. Following are fully connected layers, that are responsible from classification.

| ![Image of Dog App Frontend](<src/cnn.jpg>) |
| --- |
**CNN Architecture Example<sup>1</sup>**

Instead of building the whole model from scratch (even though this approach was also tested in earlier parts of the project), the final architecture is build on a pretrained model that is publicly available to developers.
In this case a [VGG-16](https://arxiv.org/pdf/1409.1556.pdf) with pretrained weights was used. VGG-16 is one of the popular architectures trained on [ImageNet](http://www.image-net.org/) database. ImageNet had 1000 categories of images and a certain portion are dog breed classes.
It is plausible to think that it is a good basis architecture to build on. In order to adapt this model, the original classifier part (fully connected layers that output 1000 categories) are removed and replaced with new layers outputting 133 dog breed classes.

Only the weights of fully connected layers are trained, whereas weights coming from VGG-16 are not retrained anymore.

## Extending project
This project is extended beyond the Udacity submission requirements by creating a API deployed on Heroku. In order to decrease the size of deployed app the model was recreated with Keras from the original model created in PyTorch.

| ![Image of Dog App Frontend](<src/cover_image.jpg>) |
| --- |


## Contact
* Please visit my [website](https://marketaince.com/).

## Sources
[1] Santos GL, Endo PT, Monteiro KHC, et al. [Accelerometer-Based Human Fall Detection Using Convolutional Neural Networks](https://europepmc.org/article/PMC/6480090). Sensors (Basel, Switzerland). 2019 Apr;19(7). DOI: 10.3390/s19071644.

[2] [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
