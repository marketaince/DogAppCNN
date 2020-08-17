"""
MODEL TRAINING SCRIPT
"""

# ------- #
# Imports #
# ------- #

import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import numpy as np
from os import path

from configs.config import IMAGE_SIZE, RANDOM_ROTATION

# ------------- #
# Working Paths #
# ------------- #

root_dir = 'data/dog_images/'
train_dir = root_dir + 'train/'
valid_dir = root_dir + 'valid/'
test_dir = root_dir + 'test/'

# ------------------- #
# Image Preprocessing #
# ------------------- #

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Display image
def display_image(img_path):
    image = keras.preprocessing.image.load_img(
        img_path,
        color_mode="rgb",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        interpolation="nearest"
    )

    plt.imshow(image)
    plt.show()
    return image


# Calculate mean and std for training set
def precalculate_train_mean_and_std():
    # Validation and test  image generator
    pre_generator = ImageDataGenerator(
        rescale=1. / 255,
        fill_mode='nearest'
    )

    pre_iterator = pre_generator.flow_from_directory(
        train_dir,
        batch_size=167,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=True,
        class_mode="categorical")

    # Lists to collect means and stds for all images.
    means = []
    stds = []

    # Calculate mean and std for each image
    for batch in range(len(pre_iterator)):
        image, label = next(pre_iterator)
        arr = np.array(image)
        mean = np.mean(arr, axis=(0, 1, 2))
        std = np.std(arr, axis=(0, 1, 2))
        means.append(mean)
        stds.append(std)

    # Get mean of means.
    train_mean = np.mean(means, axis=0)

    # Get mean of stds.
    train_std = np.mean(stds, axis=0)

    return train_mean, train_std


def stand_norm_data(x, channels_means, channels_stds):
    x = x - channels_means
    x = x / channels_stds
    return x


def stand_norm_generator(generator, channels_means, channels_stds):
    for x, y in generator:
        yield stand_norm_data(x, channels_means, channels_stds), y


if path.isfile('train_mean.csv') and path.isfile('train_std.csv'):
    TRAIN_MEAN = np.loadtxt('train_mean.csv', delimiter=';')
    TRAIN_STD = np.loadtxt('train_std.csv', delimiter=';')
else:
    TRAIN_MEAN, TRAIN_STD = precalculate_train_mean_and_std()
    np.savetxt('train_mean.csv', TRAIN_MEAN, delimiter=';')
    np.savetxt('train_std.csv', TRAIN_STD, delimiter=';')


# ------------- #
# Dog App Class #
# ------------- #

class DogAppCNN:

    def __init__(self, pretrained_model_dict):
        self._test = 1

        # Extract pretrained model properties
        self.pretrained_model_dict = pretrained_model_dict
        self.pretrained_model = pretrained_model_dict["model"]
        self.name = pretrained_model_dict["name"]

        # Train image generator with image augmentation
        self.train_image_generator = ImageDataGenerator(
            rotation_range=RANDOM_ROTATION,
            rescale=1. / 255,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Validation and test  image generator
        self.validation_and_test_image_generator = ImageDataGenerator(
            rescale=1. / 255,
            fill_mode='nearest'
        )

    @staticmethod
    def create_iterator(directory, generator, batch_size):
        iterator = stand_norm_generator(generator.flow_from_directory(
            directory,
            batch_size=batch_size,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            shuffle=True,
            class_mode="categorical"), TRAIN_MEAN, TRAIN_STD)

        return iterator

    # ------------------ #
    # Model Architecture #
    # ------------------ #

    # Define model architecture with custom dense layers
    def define_transfer_model(self, output_classes):
        # Dense layers
        dense_layer1 = Dense(1024, activation="relu")
        dropout_layer1 = Dropout(0.5)

        dense_layer2 = Dense(512, activation="relu")
        dropout_layer2 = Dropout(0.5)

        output_layer = Dense(output_classes, activation="softmax")

        # Freeze layers of pretrained model
        for layer in self.pretrained_model.layers:
            layer.trainable = False

        # Compose model
        final_model = keras.Sequential([
            self.pretrained_model,
            dense_layer1,
            dropout_layer1,
            dense_layer2,
            dropout_layer2,
            output_layer
        ])

        # Compile model
        final_model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        return final_model

    # -------------- #
    # Model Training #
    # -------------- #

    def train(self, train_model, epochs, lr, batches):
        train_iterator = self.create_iterator(train_dir, self.train_image_generator, batches)
        validation_iterator = self.create_iterator(valid_dir, self.validation_and_test_image_generator, batches)

        train_model.fit_generator(
            train_iterator,
            steps_per_epoch=None,
            epochs=epochs,
            validation_data=validation_iterator,
            validation_steps=None)

        train_model.save_weights(f'{self.name}_epochs_{epochs}_lr_{lr}_batches_{batches}.h5')

        return train_model

    def test(self, test_model, batches):
        test_iterator = self.create_iterator(test_dir, self.validation_and_test_image_generator, batches)

        test_model.predict(
            test_iterator
        )

    @staticmethod
    def transform_train_picture(picture):
        pass

    @staticmethod
    def predict_breed_transfer(img_path):
        pass
