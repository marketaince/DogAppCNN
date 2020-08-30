"""
Resource for Dog App Predictions
"""

# ------- #
# Imports #
# ------- #

import json
import numpy as np
from PIL import Image
from flask_restful import Resource, reqparse
from flask import request
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import model_from_json

TRAIN_MEAN = np.array([0.48703179, 0.46650660, 0.39718178])
TRAIN_STD = np.array([0.23467374, 0.23012057, 0.22998497])
IMAGE_SIZE = (224, 224)


# -------------------- #
# Function Definitions #
# -------------------- #

def stand_norm_image(one_image):
    # Move mean to 0
    one_image = one_image - TRAIN_MEAN

    # Divide by std
    one_image = one_image / TRAIN_STD

    return one_image


def process_pil_image(one_image):
    # Get image width and height
    width = one_image.size[0]
    height = one_image.size[1]

    # Choose size as the smaller number from width and height
    size = min(width, height)

    # Calculate pixels where image should be cropped
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop image
    one_image = one_image.crop((left, top, right, bottom))

    # Resize image
    img = one_image.resize(IMAGE_SIZE)

    # Convert image to numpy array
    img_array = img_to_array(img)

    # Process and reshape numpy array
    image_processed = stand_norm_image(img_array / 255).reshape(1, 224, 224, 3)

    return image_processed


def load_model(model_name):
    # Load model from json
    model_file = open(f'{model_name}.json', 'r')
    model_json = model_file.read()
    model_file.close()

    # Get indices
    indices = json.loads(model_json)["config"]["indices"]

    loaded_model = model_from_json(model_json)
    loaded_model.summary()

    # Load model weights
    loaded_model.load_weights(f'{model_name}.h5')

    # Compile model
    loaded_model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    return loaded_model, indices


def predict_dog_breeds(model, indices, image):
    # Get prediction for image
    predictions = model.predict(image)

    # Get three highest probabilities and coresponding labels
    labels = np.fliplr(predictions.argsort(axis=1)[:, -3:])
    probability = np.fliplr(np.sort(predictions, axis=1)[:, -3:])

    # Create label converter
    get_name = lambda x: indices[str(x)]
    get_name_vectorize = np.vectorize(get_name)

    # Get names corresponding to labels
    predictions = get_name_vectorize(labels)

    # Zip labels, predictions and probabilities to results
    results = zip(labels[0], predictions[0], (probability[0] * 100).round(1))

    return results


def predict_pipeline(img, model_name):
    # Process image
    img_processes = process_pil_image(img)

    # Load model
    model, indices = load_model(model_name)

    # Predict three most probable dog breeds
    results = predict_dog_breeds(model, indices, img_processes)

    # Format results
    results_list = [
        {'location': f"/img/dogs/{a}.png",
         'name': b,
         'probability': f"width: {str(c)}%;",
         'prob': str(c)
         } for a, b, c, in results
    ]

    return results_list


# Could be deployed to AWS lambda
def lambda_handler(event):  # , context
    # Get image from event
    img = Image.open(event.files['image'].stream)

    # Predict dog classes
    results_list = predict_pipeline(img, "model")

    return {
        'statusCode': 200,
        'message': json.dumps(results_list)
    }


# ----------------------------- #
# Create resource for Flask app #
# ----------------------------- #

class PredictImage(Resource):

    @classmethod
    def post(cls):
        # Get image from request
        img = Image.open(request.files['image'].stream)

        # Predict dog classes
        results_list = predict_pipeline(img, "model")

        # Dump results to json
        results_json = json.dumps(results_list)

        return results_json


# --------------------------- #
# Test prediction on an image #
# --------------------------- #

if __name__ == '__main__':
    # load test image
    test_img = load_img(
        'DogAppDeploy/test_image.jpg',
        color_mode="rgb",
        interpolation="nearest"
    )

    # Predict dog classes
    test_results = predict_pipeline(test_img, "model")

    # Print results to terminal
    print(test_results)
