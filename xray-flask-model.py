import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from matplotlib.image import imread
from tensorflow.keras.models import load_model

# App section

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

# X-Ray Functions

image_size=180
from skimage import color
def resize_image(image_path):
    image = imread(image_path)
    image_shape = image.shape
    if (len(image.shape)==3):
        image = color.rgb2gray(image)
    image_adjusted = tf.image.resize(image.reshape(image_shape[0],image_shape[1],1),(image_size,image_size))
    return image_adjusted

model = load_model('xray_pneumonia_imaging_version4_92percent.h5')

def model_predict(image_data):
    labels = ['PNEUMONIA','NORMAL']
    image_data = np.array(image_data)/255
    image_data = image_data.reshape(-1,180,180,1)
    return labels[int(int(model.predict(image_data))>0.5)]

# Configuration section

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('xray-home.html')

@app.route('/',methods=['POST'])
def upload_image():

    labels = ['PNEUMONIA','NORMAL']

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        
        image_adjusted = resize_image(filename)
        print("\n=======\n")
        result = model_predict(image_adjusted)
        print(result)
        print("\n=======\n")
        
        #print('upload_image filename: ' + filename)
        flash('Image successfuly uploaded and displayed below:')

        return render_template('xray-home.html',filename=filename,result=result)

    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static',filename='uploads/' + filename),code=301)

@app.route('/about')
def about_page():
    return render_template('xray-about.html')

if __name__ == "__main__":
    app.run(debug=True)