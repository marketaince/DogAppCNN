"""
DOG APP - MAIN TRAINING SCRIPT
"""

# ------- #
# Imports #
# ------- #

import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


from module.model import DogAppCNN
from configs.config import (
    IMAGE_SIZE,
    NR_EPOCHS,
    NR_CLASSES,
    LEARNING_RATE,
    TRAINING_BATCH_SIZE,
    VAL_TEST_BATCH_SIZE
)

# ----------------- #
# Pretrained models #
# ----------------- #

# 1) ResNet50
ResNet50_model = {
    "name": "ResNet50",
    "model": ResNet50(weights='imagenet',
                      include_top=False, pooling='avg',
                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
}

# 2) VGG16
VGG16_model = {
    "name": "VGG16",
    "model": VGG16(weights='imagenet',
                   include_top=False, pooling='avg',
                   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
}

# -------------- #
# Model Training #
# -------------- #

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_built_with_cuda())

    # Instantiate DogAppCnn class with pretrained VGG16
    DogApp = DogAppCNN(VGG16_model)

    # Create model (redefine classifier part)
    model = DogApp.define_transfer_model(NR_CLASSES)

    # Train model
    model = DogApp.train(model, NR_EPOCHS, LEARNING_RATE, TRAINING_BATCH_SIZE, VAL_TEST_BATCH_SIZE)

    # Test model
    results = DogApp.test(model, VAL_TEST_BATCH_SIZE)
