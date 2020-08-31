"""
Flask App For Dog App Predictions
"""

# ------- #
# Imports #
# ------- #

from flask import Flask
from flask_restful import Api
from flask_cors import CORS, logging

from predict_image_resource import PredictImage

# -------------- #
# App Definition #
# -------------- #

# Create new Flask instance
app = Flask(__name__)

# Settings
app.config["PROPAGATE_EXCEPTIONS"] = True
cors = CORS(app)
logging.getLogger('flask_cors').level = logging.DEBUG
api = Api(app)

# Adding Resources
api.add_resource(PredictImage, '/predict')


# ------- #
# Run App #
# ------- #

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
