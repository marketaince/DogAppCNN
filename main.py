# import os

from flask import Flask
from flask_restful import Api

from resources.prediction import PredictImage
from flask_cors import CORS, logging

app = Flask(__name__)

app.config["PROPAGATE_EXCEPTIONS"] = True
cors = CORS(app)
logging.getLogger('flask_cors').level = logging.DEBUG
api = Api(app)


api.add_resource(PredictImage, '/predict')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
