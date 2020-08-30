from flask_restful import Resource, reqparse
from flask import request
from PIL import Image
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array
import json

from configs.config import IMAGE_SIZE
from module.model import preprocess, DogAppCNN

VGG16_model = {
    "name": "VGG16",
    "model": VGG16(weights='imagenet',
                   include_top=False, pooling='avg',
                   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
}

DogApp = DogAppCNN(VGG16_model)


class PredictImage(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('message')

    @classmethod
    def post(cls):
        # print(request.get_data())
        # request_data = request.get_json()
        # request_data = cls.parser.parse_args()
        request_data = request.files['image']
        img = Image.open(request.files['image'].stream)

        width = img.size[0]
        height = img.size[1]

        print(width)
        print(height)
        size = min(width, height)

        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2

        img = img.crop((left, top, right, bottom))
        newsize = (224, 224)
        img = img.resize(newsize)

        img_array = img_to_array(img)

        image_processed = preprocess(img_array / 255).reshape(1, 224, 224, 3)

        print(image_processed)

        prediction_labels, prediction_array, prob_array = DogApp.predict_image_array(image_processed, 10, 0.001, 50)

        # prediction_labels[0], prediction_array[0], (prob_array[0] * 100).round(1)

        print(prediction_labels[0])

        print(prediction_array[0])

        print((prob_array[0] * 100).round(1))

        results_zip = zip(prediction_labels[0], prediction_array[0], (prob_array[0] * 100).round(1))

        results_list = [
            {"location": f"/img/dogs/{a}.png",
             "name": b,
             "probability": f"width: {c}%;",
             "prob": c
             } for a, b, c, in results_zip
        ]

        results_json = json.dumps({"results": results_list})

        return results_json
