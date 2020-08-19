from flask_restful import Resource, reqparse
from flask import request

class PredictImage(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('message')


    @classmethod
    def post(cls):

        request_data = request.get_json()
        # request_data = cls.parser.parse_args()

        print(request_data)

        return {"message": "Message back"}