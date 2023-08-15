from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

#from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)


@app.route("/")


def hello_world():
    img1 = "static/cat_or_dog_1.jpg"
    img2 = "static/cat_or_dog_2.jpg"
    img3 = "static/cat_or_dog_3.jpg"

    file = "mobNet_model_tf.tf"
    # Load the entire model (architecture and weights)
    loaded_model = tf.keras.models.load_model(file)
    

    img_file = img1

    test_image = load_img(img_file, target_size = (224, 224))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    result = loaded_model.predict(test_image)

    if result[0][0]  <= 0.85:
        classif = 'chien'
    else:
        classif = 'chat'

    print(classif)



app.debug = True


if __name__ == '__main__':
    app.run()




    



