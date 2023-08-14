# 1. Thêm các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
import keras
import tensorflow as tf

print(keras.__version__)
# 2. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000,:], y_train[:50000]
print(X_train.shape)