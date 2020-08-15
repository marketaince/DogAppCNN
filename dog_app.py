# Image settings
from keras.applications.resnet50 import ResNet50

from module.model import DogAppCNN
from configs.config import (
    IMAGE_SIZE,
    NR_EPOCHS,
    NR_CLASSES,
    LEARNING_RATE,
    BATCH_SIZE
)



# Pretrained models
ResNet50_model = {
    "name": "ResNet50",
    "model": ResNet50(weights='imagenet',
                      include_top=False, pooling='avg',
                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
}

# Run training
if __name__ == '__main__':
    DogApp = DogAppCNN(ResNet50_model)
    model = DogApp.define_transfer_model(NR_CLASSES)
    model = DogApp.train(model, NR_EPOCHS, LEARNING_RATE, BATCH_SIZE)
