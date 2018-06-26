#!/usr/local/bin/python
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as ply
#%matplotlib inline

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

#print('Training data shape : ', train_X.shape, train_Y.shape)
#print('Testing data shape : '), test_X.shape, test_Y.shape)
