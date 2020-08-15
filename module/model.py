"""
MODEL TRAINING SCRIPT
"""

import keras
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

from configs.config import IMAGE_SIZE, RANDOM_ROTATION

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


#
def display_image(path):
    image = keras.preprocessing.image.load_img(
        path,
        color_mode="rgb",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        interpolation="nearest"
    )

    plt.imshow(image)
    plt.show()
    return image


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
            featurewise_std_normalization=True,
            featurewise_center=True,
            rescale=1. / 255,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Validation and test  image generator
        self.validation_and_test_image_generator = ImageDataGenerator(
            # featurewise_std_normalization=True,
            rescale=1. / 255,
            fill_mode='nearest'
        )

    @staticmethod
    def create_iterator(directory, generator, batch_size):
        iterator = generator.flow_from_directory(
            directory,
            batch_size=batch_size,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            shuffle=True,
            class_mode="categorical")

        return iterator

    # ------------------ #
    # Model Architecture #
    # ------------------ #

    # Define model architecture with custom dense layers
    def define_transfer_model(self, output_classes):
        # Dense layers
        dense_layer1 = Dense(4096, activation="relu")
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
