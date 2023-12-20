#Importing Packages
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import Basuwara

class Prediction(Resource):
    #Defining MOdel and label 
    def __init__(self):
        self.model=Basuwara.MODEL
        self.classes=Basuwara.CLASS_NAMES
    #Detecting Characters in a RGB Photo
    def img_modifier(self,im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(im, (3,3), 0)
        im = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)[1]
        im = cv2.Canny(im, 150, 255, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        im = cv2.dilate(im, kernel, iterations=5)
        return im
    def get_bounding(self,im_src, im, min_area=5000):
        img_obtained = []   # array to store all the bounded images
        cnts = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = list(cnts)
        cnts.sort(key=lambda x:cv2.boundingRect(x)[0])  # sort from the leftmost x coordinate

        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area:
                x,y,w,h = cv2.boundingRect(c)
                bound = cv2.rectangle(im_src, (x, y), (x + w, y + h), (36,255,12), 2)
                img_part = im_src[y:y+h, x:x+w]
                img_obtained.append(img_part)

        return img_obtained
    #Predicting 
    def get_char(self,pred_arr):
        #Loading label
        classes=self.classes
        predicted_char = classes[np.argmax(pred_arr)]
        predicted_char = predicted_char.split('_')
        predicted_char = predicted_char[1]

        return predicted_char

    def predict_single_image(self,im):
        pred_model=self.model
        print(f'Default image: {im.shape}')
        im = tf.image.resize(im,(150,150))
        print(f'Image before expnad dims predicted shape: {im.shape}')
        im = np.expand_dims(im,axis=0)
        print(f'Image before predicted shape: {im.shape}')
        pred = pred_model.predict(im)
        predicted_char = self.get_char(pred)
        return predicted_char

    def prediction_img(self,im):
        img_modified = self.img_modifier(im)
        print(f'Image mod shape: {img_modified.shape}')
        bounded = self.get_bounding(im, img_modified)
        pred_text = []
        for images in bounded:
            char = self.predict_single_image(images)
            pred_text.append(char)

        predicted = ''.join(pred_text)

        return predicted
    
    def get(self):
        return {'error': 'GET method not allowed for this endpoint'}, 405

    def post(self):
        try:
            if 'image' not in request.files:
                response = {'error': 'no image found'}
                return response
            file = request.files['image']
            im=cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            print(f'Image shape: {im.shape}')
            predict = self.prediction_img(im)
            predict_output = predict
            return {"result": predict_output}
        except Exception as error:
            return {'error': str(error)}

