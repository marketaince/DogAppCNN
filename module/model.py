"""
DOG APP - CLASS WITH CNN - FOR TRAINING, TESTING, PREDICTION
"""

# ------- #
# Imports #
# ------- #

import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras.models import Model
from keras.models import model_from_json, Model
# from tensorflow.keras.models import model_from_json
from PIL import ImageFile
import numpy as np
from os import path
import json

from configs.config import (
    IMAGE_SIZE,
    TRAIN_MEAN_DEFAULT,
    TRAIN_STD_DEFAULT,
    USE_DEFAULT_MEAN_STD,
    RANDOM_ROTATION
)

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
def display_image(img_path: str):
    """
    Function to display image.
    :param img_path: String with image path.
    :return: PIL image.
    """
    image = load_img(
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
    """
    Function to calculate mean and std of three color channels of images in training data.
    :return: Numpy arrays with means and std.
    """

    # Image generator
    pre_generator = ImageDataGenerator(
        rescale=1. / 255,
        fill_mode='nearest'
    )

    # Image iterator
    pre_iterator = pre_generator.flow_from_directory(
        train_dir,
        batch_size=167,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=False,
        class_mode="categorical")

    # Lists to collect means and stds for all images.
    means = []
    stds = []

    # Calculate mean and std for each image
    for batch in range(len(pre_iterator)):
        image, label = next(pre_iterator)
        arr = np.array(image)
        mean = np.mean(arr, axis=(1, 2))
        std = np.std(arr, axis=(1, 2))
        means.extend(mean)
        stds.extend(std)

    # Get mean of means.
    train_mean = np.mean(means, axis=0)

    # Get mean of stds.
    train_std = np.mean(stds, axis=0)

    return train_mean, train_std


# Standardize and Normalize images with precalculated means and averages - helper
def stand_norm_data(x, channels_means, channels_stds):
    """

    :param x: Image transformed to Numpy array.
    :param channels_means: Numpy array with means.
    :param channels_stds: Numpy array with stds.
    :return: Normalized and Standardized Numpy array
    """

    x = x - channels_means
    x = x / channels_stds
    return x


# wrapper function around generator to standardize and normalize data - helper
# Not a good solution combined with fit_generator() method
def stand_norm_generator(generator, channels_means, channels_stds):
    for x, y in generator:
        yield stand_norm_data(x, channels_means, channels_stds), y


# Get mean and std of training set
if USE_DEFAULT_MEAN_STD:
    TRAIN_MEAN = TRAIN_MEAN_DEFAULT
    TRAIN_STD = TRAIN_STD_DEFAULT
elif path.isfile('train_mean.csv') and path.isfile('train_std.csv'):
    TRAIN_MEAN = np.loadtxt('train_mean.csv', delimiter=';')
    TRAIN_STD = np.loadtxt('train_std.csv', delimiter=';')
else:
    TRAIN_MEAN, TRAIN_STD = precalculate_train_mean_and_std()
    np.savetxt('train_mean.csv', TRAIN_MEAN, delimiter=';')
    np.savetxt('train_std.csv', TRAIN_STD, delimiter=';')


# Standardize and Normalize one image - helper
def preprocess(one_image):
    one_image = one_image - TRAIN_MEAN
    one_image = one_image / TRAIN_STD
    return one_image


# ------------- #
# Dog App Class #
# ------------- #

class DogAppCNN:

    def __init__(self, pretrained_model_dict):

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
            fill_mode='nearest',
            preprocessing_function=self.preprocess_image
        )

        # Validation and test  image generator
        self.validation_and_test_image_generator = ImageDataGenerator(
            rescale=1. / 255,
            fill_mode='nearest',
            preprocessing_function=self.preprocess_image
        )

    #         self.indices = self.get_indices()

    # Image preprocessing passed to generators
    @staticmethod
    def preprocess_image(one_image):
        return stand_norm_data(one_image, TRAIN_MEAN, TRAIN_STD)

    # Image iterator
    @staticmethod
    def create_iterator(directory, generator, batch_size):
        iterator = generator.flow_from_directory(
            directory,
            batch_size=batch_size,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            shuffle=True,
            class_mode="categorical")

        return iterator

    # Get training indices
    def get_indices(self):

        train_iterator = self.create_iterator(train_dir, self.train_image_generator, 10)
        indices = train_iterator.class_indices

        return {str(label): ' '.join([i.capitalize() for i in name[4:].split('_')]) for name, label in indices.items()}

    # ------------------ #
    # Model Architecture #
    # ------------------ #

    # Define model architecture with custom dense layers
    def define_transfer_model(self, output_classes):

        # Dense layers
        dense_1 = Dense(1024)(self.pretrained_model.output)
        activ_1 = Activation('relu')(dense_1)
        dropout_1 = Dropout(0.5)(activ_1)

        dense_2 = Dense(512)(dropout_1)
        activ_2 = Activation('relu')(dense_2)
        dropout_2 = Dropout(0.5)(activ_2)

        model_output = Dense(output_classes, activation="softmax")(dropout_2)

        # Compose model
        final_model = Model(inputs=self.pretrained_model.input, output=model_output)

        # Freeze layers of pretrained model
        for layer in self.pretrained_model.layers:
            layer.trainable = False

        # Compile model
        final_model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        print('Compiled model summary:')
        final_model.summary()

        return final_model

    # ----------------------- #
    # Load Model Architecture #
    # ----------------------- #

    # Load if model exists
    def load_existing_transfer_model(self, epochs, lr, batches):

        # Construct model name
        model_name = f'{self.name}_epochs_{epochs}_lr_{lr}_batches_{batches}'

        # Verify that model and weights are in directory
        if path.isfile(f'{model_name}.json') and path.isfile(f'{model_name}.h5'):

            # Load model from json
            model_file = open(f'{model_name}.json', 'r')
            model_json = model_file.read()
            model_file.close()

            # get indices
            indices = json.loads(model_json)["config"]["indices"]
            print(indices)

            loaded_model = model_from_json(model_json)
            loaded_model.summary()

            # Load model weights
            loaded_model.load_weights(f'{model_name}.h5')

            # Compile model
            loaded_model.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])

            return loaded_model, indices

        else:
            print(f"Can't find model with name: {model_name}!")

    # -------------- #
    # Model Training #
    # -------------- #

    def save_train(self, train_model, epochs, lr, train_batches):

        train_model_json = train_model.to_json()

        # Convert to dict
        model_dict = json.loads(train_model_json)

        # Modify config with model name and indices
        model_dict["config"]["name"] = f'{self.name}_epochs_{epochs}_lr_{lr}_batches_{train_batches}'
        model_dict["config"]["indices"] = self.get_indices()

        # dump back to json
        train_model_json = json.dumps(model_dict)

        # Save model as json
        with open(f'{self.name}_epochs_{epochs}_lr_{lr}_batches_{train_batches}.json', 'w') as json_file:
            json_file.write(train_model_json)



    def train(self, train_model, epochs, lr, train_batches, valid_batches):

        # Create iterator for training and validation
        train_iterator = self.create_iterator(train_dir, self.train_image_generator, train_batches)
        validation_iterator = self.create_iterator(valid_dir, self.validation_and_test_image_generator, valid_batches)

        # Train model
        train_model.fit_generator(
            train_iterator,
            epochs=epochs,
            validation_data=validation_iterator)

        # Create json from model
        train_model_json = train_model.to_json()

        # Convert to dict
        model_dict = json.loads(train_model_json)

        # Modify config with model name and indices
        model_dict["config"]["name"] = f'{self.name}_epochs_{epochs}_lr_{lr}_batches_{train_batches}'
        model_dict["config"]["indices"] = self.get_indices()

        # dump back to json
        train_model_json = json.dumps(model_dict)

        # Save model as json
        with open(f'{self.name}_epochs_{epochs}_lr_{lr}_batches_{train_batches}.json', 'w') as json_file:
            json_file.write(train_model_json)

        # Save model weights
        train_model.save_weights(f'{self.name}_epochs_{epochs}_lr_{lr}_batches_{train_batches}.h5')

        return train_model

    # ------------- #
    # Model Testing #
    # ------------- #

    def test(self, test_model, batches):

        # Create iterator for test directory
        test_iterator = self.create_iterator(test_dir, self.validation_and_test_image_generator, batches)

        # Evaluate model
        evaluation = test_model.evaluate_generator(
            test_iterator
        )

        results = {f'test_{test_model.metrics_names[i]}': result for i, result in enumerate(evaluation)}

        print(f'test_loss: {round(results["test_loss"], 4)} - test_accuracy: {round(results["test_accuracy"], 4)}')

        return results

    @staticmethod
    def transform_train_picture(picture):
        pass

    def predict_image_array(self, image_array, nr_epochs, lr_rate, batches):
        model, indices = self.load_existing_transfer_model(nr_epochs, lr_rate, batches)
        predictions = model.predict(image_array)
        labels = np.fliplr(predictions.argsort(axis=1)[:, -3:])
        probability = np.fliplr(np.sort(predictions, axis=1)[:, -3:])

        get_name = lambda x: indices[str(x)]
        get_name_vectorize = np.vectorize(get_name)

        predictions = get_name_vectorize(labels)

        return labels, predictions, probability

    def predict_image_from_path(self, image_path, nr_epochs, lr_rate, batches):
        image_loaded = load_img(
            image_path,
            color_mode="rgb",
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            interpolation="nearest"
        )

        image_array = img_to_array(image_loaded)

        image_processed = preprocess(image_array / 255).reshape(1, 224, 224, 3)

        prediction_labels, prediction_array, prob_array = self.predict_image_array(image_processed, nr_epochs, lr_rate,
                                                                                   batches)

        return prediction_labels[0], prediction_array[0], (prob_array[0] * 100).round(1)

    @staticmethod
    def predict_breed_transfer(img_path):
        pass
