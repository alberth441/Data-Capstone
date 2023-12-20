from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2
import numpy as np
from tensorflow import keras
import os


from Prediction import Prediction
from Opening import Opening

app = Flask(__name__)
api=Api(app)

api.add_resource(Prediction,"/prediction",methods=['POST'])


PORT = 5000
if __name__ == '__main__':
    app.run(debug = False, host='0.0.0.0', port=PORT)