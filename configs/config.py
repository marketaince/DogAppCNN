"""
CONFIG FOR DOG APP CNN MODEL
"""

# ------- #
# Imports #
# ------- #

import numpy as np

# -------------- #
# Model Settings #
# -------------- #

NR_CLASSES = 133

# ----------------- #
# Training Settings #
# ----------------- #

TRAINING_BATCH_SIZE = 200
VAL_TEST_BATCH_SIZE = 100
LEARNING_RATE = 0.001
NR_EPOCHS = 10

# ----------------------------- #
# Image Transformation Settings #
# ----------------------------- #

IMAGE_SIZE = 224
RANDOM_ROTATION = 15

# Default mean and std values
TRAIN_MEAN_DEFAULT = np.array([0.485, 0.456, 0.406])
TRAIN_STD_DEFAULT = np.array([0.229, 0.224, 0.225])

# use default values
USE_DEFAULT_MEAN_STD = True
