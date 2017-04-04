"""BSDS500 Dataset


"""

from __future__ import absolute_import, print_function


import tensorflow as tf
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

width = 100
height = 100

def load_data(dirname="BSR/BSDS500/data/images"):

    print("Start Data Loading")
    fpath = os.path.join(dirname, 'train')
    X_train, Y_train = jpg_to_tensor(fpath)
    print("Finish Train Data Loading")
    print("Train Data: ", len(X_train))
    
    fpath = os.path.join(dirname, 'test')
    X_test, Y_test = jpg_to_tensor(fpath)
    print("Finish Test Data Loading")
    print("Test Data: ", len(X_test))
    
    fpath = os.path.join(dirname, 'val')
    X_val, Y_val = jpg_to_tensor(fpath)
    print("Finish Validation Data Loading")
    print("Validation Data: ", len(X_val))
    
    return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)
    

def jpg_to_tensor(dirname):
    fpath = os.path.join(dirname,'*.jpg')
    file_list = glob.glob(fpath)
    X = []
    Y = []
    for i in range(len(file_list)):
        jpeg_r = tf.read_file(file_list[i])
        image = tf.image.decode_jpeg(jpeg_r, channels=3)
        resized_image = tf.image.resize_image_with_crop_or_pad(image, width, height)
        gaussian_image = Gaussian_noise_layer(resized_image, 10)
        gaussian_image = tf.cast(gaussian_image, tf.float16)
        resized_image = tf.cast(resized_image, tf.float16)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        y = sess.run(resized_image)
        x = sess.run(gaussian_image)
        """
        y = y.astype(np.uint8)
        x =  x.astype(np.uint8)
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        a.set_title('Noise Image(Input)')
        plt.imshow(y)
        b = fig.add_subplot(1,2,2)
        b.set_title('Denoise Image(Output)')
        plt.imshow(x)
        plt.show()
        """
        if i == 0:
            X = x
            Y = y
        elif i == 1:
            X = np.stack((X,x))
            Y = np.stack((Y,y))
        else:
            X = np.concatenate((X, x[None, ...]))
            Y = np.concatenate((Y, y[None, ...]))
        if (i+1) % 20 == 0:
            print(i+1, " / ", len(file_list), "Loading...")

    return X, Y

def Gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape = input_layer.get_shape(), mean = 0.0, stddev = std, dtype = tf.float32)
    input_layer = tf.to_float(input_layer)
    return input_layer + noise
