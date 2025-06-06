# -*- coding: utf-8 -*-
# +
import json, os

import tensorflow as tf

from tensorflow.keras.layers import Reshape, UpSampling2D, MaxPooling2D, add, SeparableConv2D, Dense
from tensorflow.keras.layers import ReLU, Dropout, BatchNormalization, Flatten
from tensorflow.keras.activations import tanh
# -

with open(os.path.dirname(os.path.abspath(__file__)) + '/model_config.json', 'r') as f:
    config = json.load(f)


# Sampling function for VAEs
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], config["latent_dim"]), mean=0., stddev=0.1)
    return z_mean + tf.math.exp(z_log_sigma) * epsilon


def create_encoder(input_img, vae=False):
    x = input_img

    # Downsize layers
    for i in range(len(config["filters_encoder"])):
        if i != 0:
            x = MaxPooling2D(3, strides=2, padding="same")(x)
        x = create_res_block(x, config["filters_encoder"][i], kernel_size=3)

    # Latent dimension
    x = Flatten()(x)
    
    if vae==True:
        # For the vae, encode into two vectors
        y = Dense(config["latent_dim"])(x)
        y = BatchNormalization()(y)
        y = tanh(y)
        z_mean = Dropout(0.3)(y)
        
        y = Dense(config["latent_dim"])(x)
        y = BatchNormalization()(y)
        y = tanh(y)
        z_log_sigma = Dropout(0.3)(y)
        
        return z_mean, z_log_sigma
        
    else:
        # Latent dimension      
        x = Dense(config["latent_dim"])(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3, name='latent_layer')(x)
        
        return x


def create_decoder(latent_space):
    # From latent space to image
    lat_width = int(config["input_dim"][0]/(pow(2, len(config["filters_encoder"])-1)))
    lat_height = int(config["input_dim"][1]/(pow(2, len(config["filters_encoder"])-1)))
    lat_channels = int(config["filters_decoder"][0])

    x = Dense(lat_height*lat_width*lat_channels)(latent_space)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    x = Reshape((lat_width, lat_height, lat_channels))(x)

    # Upsampling layers
    for i in range(len(config["filters_decoder"])):
        if i != 0:
            x = UpSampling2D(2)(x)
        x = create_res_block(x, config["filters_decoder"][i], kernel_size=3)

    # Output image generation
    x = SeparableConv2D(config["input_dim"][2], kernel_size=1, strides=1, padding="same",
                        activation='tanh', name='rec')(x)
    return x


def create_classifier(latent_dim):
    # Perceptron layers
    x = Flatten()(latent_dim)

    for n_neurons in config["classifier_perceptron"]:
        x = Dense(n_neurons)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)

    x = Dense(config["n_classes"], activation='softmax', name='class')(x)
    return x


def create_res_block(input_layer, filters, kernel_size):
    x = input_layer
    for i in range(2):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = SeparableConv2D(filters, kernel_size=kernel_size, strides=1, padding="same")(x)

    # Match num of filters
    y = SeparableConv2D(filters, kernel_size=(1,1), strides=1, padding="same")(input_layer)

    x = add([x, y])
    return x