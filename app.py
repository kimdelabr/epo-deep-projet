from flask import Flask, render_template, request
import numpy as np
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image


import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub

mobilenet ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
feature_extractor = hub.KerasLayer(mobilenet,input_shape=(224,224,3))


# Freeze the variables in the feature extractor layer, so that the training only modifies the final classifier layer.
feature_extractor.trainable = False

model = tf.keras.Sequential()

model.add(feature_extractor),
model.add(Dense(1, activation='sigmoid'))

# Model Compile
model.compile(loss='binary_crossentropy',
optimizer="adam",
metrics=['accuracy'])

# Train
model.fit(training_set, epochs=20, verbose=1, validation_data=test_set)



app = Flask(__name__)

@app.route("/")
def hello_world():
    img1 = "static/cat_or_dog_1.jpg"
    img2 = "static/cat_or_dog_2.jpg"
    img3 = "static/cat_or_dog_3.jpg"


    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                    shear_range = 0.2, 
                                    zoom_range = 0.2, 
                                    horizontal_flip = True)


    test_datagen = ImageDataGenerator(rescale = 1./255)


    dataset_training_set = "dataset/training_set"


    training_set = train_datagen.flow_from_directory(dataset_training_set, 
                                                    target_size = (224, 224), batch_size = 32,
                                                    class_mode = 'binary')

    dataset_test_set = "dataset/test_set"


    test_set = test_datagen.flow_from_directory(dataset_test_set,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

    

    img_file = img3
  
    test_image = load_img(img_file, target_size = (224, 224))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    result = model.predict(test_image)
    #training_set.class_indices
    if result[0][0] <= 0.85:
        return 'chien'
    else:
        return 'chat'


if __name__ == '__main__':
    app.run()
