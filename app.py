'''
created on: 13/08/2021
@author: Rohit Sharma
'''
from __future__ import division, print_function
import numpy as np
import sys
import os
import glob
import re
# keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# define a flask app
app = Flask(__name__)
# model save with keras model.save()
model_path = 'vgg19_model.h5'
# Load model
model = load_model(model_path)
model.make_predict_function()  #necessary

# preprocessing function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size = (224,224))

    # preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)
    
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # get the file from the post
        f = request.files['file'] #from html
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path) # save images into upload folder
        
        # make predictions
        pred = model_predict(file_path, model)
        # using decode convert that index into a class name
        pred_class = decode_predictions(pred, top=1)  #ImageNet Decode
        result = str(pred_class[0][0][1])  #convert to string [0][0][1] is the location of class
        return result
    return None 


if __name__ == '__main__':
    app.run(debug=True)