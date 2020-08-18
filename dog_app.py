"""
DOG APP - MAIN SCRIPT
"""

# ------- #
# Imports #
# ------- #

import tensorflow as tf
from keras.applications.resnet50 import ResNet50
# from keras import backend as K


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


# -------------- #
# Model Training #
# -------------- #

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_built_with_cuda())
    # K.tensorflow_backend._get_available_gpus()
    DogApp = DogAppCNN(ResNet50_model)
    model = DogApp.define_transfer_model(NR_CLASSES)
    model = DogApp.train(model, NR_EPOCHS, LEARNING_RATE, TRAINING_BATCH_SIZE, VAL_TEST_BATCH_SIZE)
    results = DogApp.test(model, VAL_TEST_BATCH_SIZE)
