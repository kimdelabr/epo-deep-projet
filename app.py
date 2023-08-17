from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf

#from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']
    model_file = "model_cnn.h5"
    loaded_model = tf.keras.models.load_model(model_file)

    if 'file' not in request.files:
        flash('Pas de fichier')
        return redirect(request.url)
    if file.filename == '':
        flash('Aucune image sélectionnée pour le téléchargement')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("nom de fichier de l'image téléchargée : " + filename)
        flash("L'image a été téléchargée avec succès et s'affiche ci-dessous")

        image_path = "./static/uploads/" + filename
        #imagefile.save(image_path)

        img_file = image_path

        test_image = load_img(img_file, target_size = (64, 64))
        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        result = loaded_model.predict(test_image)
        if result[0][0]  <= 0.85:
            classif = 'chien'
        else:
            classif = 'chat'

        return render_template('index.html', filename=filename, prediction=classif)
    else:
        flash("Les types d'images autorisés sont : png, jpg, jpeg, gif.")
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    print("nom du fichier de l'image affichée : " + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()