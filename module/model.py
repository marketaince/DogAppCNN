"""

"""

import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
# from tqdm import tqdm
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets
from collections import OrderedDict
import json

# Allow loading truncated images.
from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import matplotlib.pyplot as plt

# Check if CUDA is available.
use_cuda = torch.cuda.is_available()


# Use VGG16 for transfer learning.
model_transfer = models.vgg16(pretrained=True)

# Freeze parameters of the model.
for param in model_transfer.parameters():
    param.requires_grad = False




# Define new classifier.
dog_breed_classifier = nn.Sequential(OrderedDict([
    ('0', nn.Linear(25088,4096)),
    ('1', nn.ReLU()),
    ('2', nn.Dropout(0.5)),
    ('3', nn.Linear(4096,4096)),
    ('4', nn.ReLU()),
    ('5', nn.Dropout(0.5)),
    ('6', nn.Linear(4096,133))
        ]))

# Change classifier in the model.
model_transfer.classifier = dog_breed_classifier


# Move model to GPU if CUDA is available.
if use_cuda:
    model_transfer = model_transfer.cuda()

# Define transformations for train, validation and test sets.
train_trans_tr = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

test_valid_trans_tr = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

# Paths to train, validation and test sets.
root_dir = '../data/dog_images/'
train_dir = root_dir + 'train/'
valid_dir = root_dir + 'valid/'
test_dir = root_dir + 'test/'


# Dataloaders with local paths.
train_data_tr = datasets.ImageFolder(train_dir, transform = train_trans_tr)
valid_data_tr = datasets.ImageFolder(valid_dir, transform = test_valid_trans_tr)
test_data_tr = datasets.ImageFolder(test_dir, transform = test_valid_trans_tr)

data_transfer = {'train':train_data_tr, 'valid':valid_data_tr, 'test':test_data_tr}

class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]

class DogAppCNN():

    def __init__(self):

        self._test = 1

        self.model = model_transfer.load_state_dict(torch.load('model_transfer.pt'))

    @staticmethod
    def transform_train_picture(picture):
        pass



    def predict_breed_transfer(img_path):
        # Load image with PIL package.
        im = Image.open(img_path).convert('RGB')

        # Define transforms needed for VGG16.
        transform_im = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        # Transform image and reshape tensor.
        im_t = transform_im(im)
        im_t = im_t.view(-1, 3, 224, 224)

        if use_cuda:
            im_t = im_t.cuda()

        # Pass image through VGG network.
        model_transfer.eval()
        with torch.no_grad():
            output = model_transfer(im_t)

            # Get the class with highest predicted probability.
            prob, pred = torch.topk(output, 1)

        # Exctract and return the predicted class.
        pred_class = pred.item()
        dog = class_names[pred_class]

        return pred_class, dog

app = DogAppCNN()

label, dog = app.predict_breed_transfer('/test_images/alaskan_malamute.jpg')

print(label)
print(dog)