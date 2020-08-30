import json
import numpy as np
from PIL import Image
from flask_restful import Resource, reqparse
from flask import request
import os

print(os.getcwd())
print(os.listdir())

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import model_from_json

TRAIN_MEAN = np.array([0.48703179, 0.46650660, 0.39718178])
TRAIN_STD = np.array([0.23467374, 0.23012057, 0.22998497])

# TRAIN_MEAN = [0.48703179, 0.46650660, 0.39718178]
# TRAIN_STD = [0.23467374, 0.23012057, 0.22998497]


# Standardize and Normalize one image - helper
def stand_norm_image(one_image):
    one_image = one_image - TRAIN_MEAN
    one_image = one_image / TRAIN_STD
    return one_image


def process_pil_image(one_image):

    width = one_image.size[0]
    height = one_image.size[1]

    print(width)
    print(height)
    size = min(width, height)

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    one_image = one_image.crop((left, top, right, bottom))
    newsize = (224, 224)
    img = one_image.resize(newsize)

    img_array = img_to_array(img)

    image_processed = stand_norm_image(img_array / 255).reshape(1, 224, 224, 3)

    return image_processed


def load_model(model_name):
    # Load model from json
    model_file = open(f'{model_name}.json', 'r')
    model_json = model_file.read()
    model_file.close()

    # get indices
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

    predictions = model.predict(image)
    labels = np.fliplr(predictions.argsort(axis=1)[:, -3:])
    probability = np.fliplr(np.sort(predictions, axis=1)[:, -3:])

    get_name = lambda x: indices[str(x)]
    get_name_vectorize = np.vectorize(get_name)

    predictions = get_name_vectorize(labels)

    results = zip(labels[0], predictions[0], (probability[0] * 100).round(1))

    return results


def lambda_handler(event, context):
    print(event)

    request_data = event.files['image']

    img = Image.open(event.files['image'].stream)

    img = process_pil_image(img)

    model, indices = load_model("model")

    results = predict_dog_breeds(model, indices, img)

    results_list = [
        {"location": f"/img/dogs/{a}.png",
         "name": b,
         "probability": f"width: {c}%;",
         "prob": c
         } for a, b, c, in results
    ]


    print(results_list)


    # TODO implement
    return {
        'statusCode': 200,
        'message': json.dumps('Hello from Lambda!')
    }



class PredictImage(Resource):
    # parser = reqparse.RequestParser()
    # parser.add_argument('message')

    @classmethod
    def post(cls):

        request_data = request.files['image']
        img = Image.open(request.files['image'].stream)

        img = process_pil_image(img)

        model, indices = load_model("model")

        results = predict_dog_breeds(model, indices, img)

        results_list = [
            {'location': f"/img/dogs/{a}.png",
             'name': b,
             'probability': f"width: {str(c)}%;",
             'prob': str(c)
             } for a, b, c, in results
        ]

        print(results_list)

        results_json = json.dumps(results_list)

        return results_json



if __name__ == '__main__':
    img = load_img(
        './Welsh_springer_spaniel_08203.jpg',
        color_mode="rgb",
        interpolation="nearest"
    )

    img = process_pil_image(img)

    model, indices = load_model("model")

    results = predict_dog_breeds(model, indices, img)

    results_list = [
        {"location": f"/img/dogs/{a}.png",
         "name": b,
         "probability": f"width: {c}%;",
         "prob": c
         } for a, b, c, in results
    ]

    print(results_list)