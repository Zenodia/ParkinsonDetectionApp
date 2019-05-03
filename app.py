from __future__ import print_function
import os
import sys
from IPython.display import display, Image
from scipy import ndimage
import scipy.misc
import numpy as np
import flask
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from flask import Flask, request, render_template, send_from_directory
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import os, cv2
import argparse



### modified from :https://github.com/ibrahimokdadov/upload_file_python/tree/master/src
##### model loading methods
    
    
def load_model():
    #make model global
    global model
    ######## build the model and load the pre-trained weights  
    ### building the network
    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    from keras.models import load_model
    model=load_model('CNN_parkinson.h5') #load the model locally
            
    return model

def load_model_here():
    global model
    #make model global
    #path = Model.get_model_path('visualsearch.h5')
    #model=load_model(path)
    try:
        model=load_model()
        #define graf othervise does not work with flask 
        #graph = tf.get_default_graph()
    except:
        del model
        keras.backend.clear_session()
        model=load_model()
        #graph = tf.get_default_graph()
    return model

def data_preprocessing(img):
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img= img.reshape(224,224,1)
    return np.expand_dims(img,axis=0)
####### end of model loading method


app = Flask(__name__)
#app = Flask(__name__, static_folder="images")

model=load_model_here()
graph = tf.get_default_graph()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")




@app.route("/upload", methods=["POST"])
def upload():
    global graph
    #model=load_model_here()
    target = os.path.join(APP_ROOT, 'images/')
    #target = os.path.join(APP_ROOT, 'static/')
    print("taget",target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        forsave=ndimage.imread(destination)
        im=data_preprocessing(forsave)
        with graph.as_default():        
            ye_test = model.predict(im)            
        title='probability:{}% having parkinsons'.format(str(round(ye_test[0][0]*100,2)))
        
        YN=1 if ye_test[0][0] >0.5 else 0
        print("YN",YN)
        have={1:'parkinsons',0:'healthy'}
        print(have.items())
        print(title)
        img=im[0]
        img=np.squeeze(img)
        name=filename.split('.')[0]
        plt.imsave(target+'/pred_{}.jpg'.format(have[YN]),img, vmin=0, vmax=255,cmap="gray")     
        print("destination",destination)
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete_display_image.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)

if __name__ == "__main__":
    app.run(port=1235711 ,debug=True)