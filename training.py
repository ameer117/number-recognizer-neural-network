import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import itertools
import numpy as np
import matplotlib.pyplot as plt
#Ameer Hussain
#73574836
#followed techwithtim youtube channel's tutorial for help with code
data = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()


train_images = train_images/255.0
train_images = train_images[:20000]
test_images = test_images/255.0
test_images = test_images



model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(2800, activation="relu"),
    keras.layers.Dense(140, activation="relu"),
   keras.layers.Dense(10, activation="softmax")
    ])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images[:20000], train_labels[:20000], epochs=5)


model.save("model.h5")