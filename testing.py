import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys

data = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

model = keras.models.load_model(sys.argv[1])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

test_loss, test_acc = model.evaluate(test_images, test_labels)


print('\n')
print('accuracy:', test_acc)