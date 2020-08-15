"""
MODEL TRAINING SCRIPT
"""

# import datetime as dt
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import cv2
# import numpy as np
# import os
# import sys
# import random
# import warnings
# from sklearn.model_selection import train_test_split

import keras
import matplotlib.pyplot as plt

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.layers import Dense, Dropout, Activation

# from keras import backend as K
# from keras import regularizers
# from keras.models import Sequential
# from keras.models import Model
# from keras.layers import Dense, Dropout, Activation
# from keras.layers import Flatten, Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import BatchNormalization, Input
# from keras.layers import Dropout, GlobalAveragePooling2D
# from keras.callbacks import Callback, EarlyStopping
# from keras.callbacks import ReduceLROnPlateau
# from keras.callbacks import ModelCheckpoint
# import shutil
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator
#
# from keras.models import load_model
#
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input
# from keras.applications.resnet50 import decode_predictions
#
# from keras.applications import inception_v3
# from keras.applications.inception_v3 import InceptionV3
#
#
# from keras.applications.nasnet import NASNetMobile


from keras.preprocessing.image import ImageDataGenerator

# -------- #
# Settings #
# -------- #

# Image settings
image_size = 224
random_rotation = 15

# Training settings
batch_size = 100
learning_rate = 0.001
nr_epochs = 10

nr_classes = 133

# ------------- #
# Working Paths #
# ------------- #

root_dir = '/data/dog_images/'
train_dir = root_dir + 'train/'
valid_dir = root_dir + 'valid/'
test_dir = root_dir + 'test/'

# ------------------- #
# Image Preprocessing #
# ------------------- #

# Train image generator with image augmentation
train_image_generator = ImageDataGenerator(
    rotation_range=random_rotation,
    featurewise_std_normalization=True,
    featurewise_center=True,
    rescale=1. / 255,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and test  image generator
validation_and_test_image_generator = ImageDataGenerator(
    # featurewise_std_normalization=True,
    rescale=1. / 255,
    fill_mode='nearest'
)


#
def display_image(path):
    image = keras.preprocessing.image.load_img(
        path,
        color_mode="rgb",
        target_size=(image_size, image_size),
        interpolation="nearest"
    )

    plt.imshow(image)
    plt.show()
    return image


# loading from directory
train_iterator = train_image_generator.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    shuffle=True,
    class_mode="categorical")

validation_iterator = validation_and_test_image_generator.flow_from_directory(
    valid_dir,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    shuffle=True,
    class_mode="categorical")

test_iterator = validation_and_test_image_generator.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    target_size=(image_size, image_size),
    shuffle=True,
    class_mode="categorical")


# ------------------ #
# Model Architecture #
# ------------------ #

def defineTransferModel():

    # Pretrained model
    ResNet50_model = ResNet50(weights='imagenet',
                          include_top=False, pooling='avg',
                          input_shape=(image_size, image_size, 3))


    # Dense layers
    dense_layer1 = Dense(4096, activation="relu")
    dropout_layer1 = Dropout(0.5)

    dense_layer2 = Dense(512, activation="relu")
    dropout_layer2 = Dropout(0.5)

    output_layer = Dense(nr_classes, activation="softmax")


    # Freeze layers of pretrained model
    for layer in ResNet50_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        ResNet50_model,
        dense_layer1,
        dropout_layer1,
        dense_layer2,
        dropout_layer2,
        output_layer
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return (model)


transfer_model = defineTransferModel()

# -------------- #
# Model Training #
# -------------- #

transfer_model.fit_generator(
        train_iterator,
        steps_per_epoch=None,
        epochs=nr_epochs,
        validation_data=validation_iterator,
        validation_steps=None)
transfer_model.save_weights('transfer_model.h5')

#
#
#
#
# # Allow loading truncated images.
# from PIL import ImageFile
# #ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# import cv2
# import matplotlib.pyplot as plt
#
# # Check if CUDA is available.
# use_cuda = torch.cuda.is_available()
#
#
# # Use VGG16 for transfer learning.
# model_transfer = models.vgg16(pretrained=True)
#
# # Freeze parameters of the model.
# for param in model_transfer.parameters():
#     param.requires_grad = False
#
#
#
#
# # Define new classifier.
# dog_breed_classifier = nn.Sequential(OrderedDict([
#     ('0', nn.Linear(25088,4096)),
#     ('1', nn.ReLU()),
#     ('2', nn.Dropout(0.5)),
#     ('3', nn.Linear(4096,4096)),
#     ('4', nn.ReLU()),
#     ('5', nn.Dropout(0.5)),
#     ('6', nn.Linear(4096,133))
#         ]))
#
# # Change classifier in the model.
# model_transfer.classifier = dog_breed_classifier
#
#
# # Move model to GPU if CUDA is available.
# if use_cuda:
#     model_transfer = model_transfer.cuda()
#
# # Define transformations for train, validation and test sets.
# train_trans_tr = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])])
#
# test_valid_trans_tr = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])])
#
# # Paths to train, validation and test sets.
# root_dir = '../data/dog_images/'
# train_dir = root_dir + 'train/'
# valid_dir = root_dir + 'valid/'
# test_dir = root_dir + 'test/'
#
#
# # Dataloaders with local paths.
# train_data_tr = datasets.ImageFolder(train_dir, transform = train_trans_tr)
# valid_data_tr = datasets.ImageFolder(valid_dir, transform = test_valid_trans_tr)
# test_data_tr = datasets.ImageFolder(test_dir, transform = test_valid_trans_tr)
#
# data_transfer = {'train':train_data_tr, 'valid':valid_data_tr, 'test':test_data_tr}
#
# class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]

class DogAppCNN():

    def __init__(self):
        self._test = 1

        # self.model = model_transfer.load_state_dict(torch.load('model_transfer.pt'))

    @staticmethod
    def transform_train_picture(picture):
        pass

    @staticmethod
    def predict_breed_transfer(img_path):
        pass

        # Load image with PIL package.
        # im = Image.open(img_path).convert('RGB')
        #
        # # Define transforms needed for VGG16.
        # transform_im = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])
        #
        # # Transform image and reshape tensor.
        # im_t = transform_im(im)
        # im_t = im_t.view(-1, 3, 224, 224)
        #
        # if use_cuda:
        #     im_t = im_t.cuda()
        #
        # # Pass image through VGG network.
        # model_transfer.eval()
        # with torch.no_grad():
        #     output = model_transfer(im_t)
        #
        #     # Get the class with highest predicted probability.
        #     prob, pred = torch.topk(output, 1)
        #
        # # Exctract and return the predicted class.
        # pred_class = pred.item()
        # dog = class_names[pred_class]

        # return pred_class, dog


app = DogAppCNN()

# label, dog = app.predict_breed_transfer('/test_images/alaskan_malamute.jpg')
#
# print(label)
# print(dog)
