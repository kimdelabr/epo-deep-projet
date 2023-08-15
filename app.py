from flask import Flask, render_template, request
import numpy as np
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

    model_final = load_model("model_cnn.h5")

    img_file = img3
    #test_image = load_img(img_file, target_size = (64, 64))
    test_image = Image.open(img_file).resize((64, 64))

    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model_final.predict(test_image)
    #training_set.class_indices
    if result[0][0] == 1:
        return 'chien'
    else:
        return 'chat'

app.debug = True


if __name__ == '__main__':
    app.run()




    



