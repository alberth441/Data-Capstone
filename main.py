from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2
import numpy as np
from tensorflow import keras
import os


from api_model.Prediction import Prediction

app = Flask(__name__)
api=Api(app)

api.add_resources(Prediction,"/prediction")


if __name__ == ' __main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)