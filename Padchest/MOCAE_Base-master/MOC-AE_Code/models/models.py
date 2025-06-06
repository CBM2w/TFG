# -*- coding: utf-8 -*-
# +
import json, os

import tensorflow as tf

from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model

import sys
sys.path.append("..")
from models.dl_blocks import *
# -

class DL_Model(object):
    def __init__(self):
        with open(os.path.dirname(os.path.abspath(__file__)) + '/model_config.json', 'r') as f:
            config = json.load(f)
        
        # Models
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.classifier = None
        self.mocae = None

        # Networks configuration from config file
        self.latent_dim = config['latent_dim']

        # Image dimensions from config file
        self.img_width = config["input_dim"][0]
        self.img_height =  config["input_dim"][1]
        self.img_channels = config["input_dim"][2]
    
    def create_ae(self):
        input_img = Input(shape=(self.img_width,
                                 self.img_height,
                                 self.img_channels))

        # Encoder generation
        latent_space = create_encoder(input_img)
        self.encoder = Model(input_img, latent_space)

        # Decoder generation
        z = Input(shape=(self.latent_dim,))
        output_img = create_decoder(z)
        self.decoder = Model(z, output_img, name='rec')
        
        reconstruction = self.decoder(self.encoder(input_img))
        
        # Autoencoder generation
        self.autoencoder = Model(input_img, reconstruction)
        
        # Full autoencoder (duplicity for simplicity purposes)
        self.mocae = Model(input_img, reconstruction, name='rec')
        
    def create_classifier(self):
        input_img = Input(shape=(self.img_width,
                                 self.img_height,
                                 self.img_channels))

        # Encoder generation
        latent_space = create_encoder(input_img)
        self.encoder = Model(input_img, latent_space)

        # Classifier generation
        z = Input(shape=(self.latent_dim,))
        classification = create_classifier(z)
        self.classifier = Model(z, classification, name='class')
        
        classification = self.classifier(self.encoder(input_img))
        
        # Full classifier
        self.mocae = Model(input_img, classification, name='class')
        
    def create_mocae(self):
        input_img = Input(shape=(self.img_width,
                                 self.img_height,
                                 self.img_channels))

        # Encoder generation
        latent_space = create_encoder(input_img)
        self.encoder = Model(input_img, latent_space)
        
        # Decoder generation
        z_dec = Input(shape=(self.latent_dim,))
        output_img = create_decoder(z_dec)
        self.decoder = Model(z_dec, output_img, name='rec')
        
        reconstruction = self.decoder(self.encoder(input_img))
        
        # Autoencoder generation
        self.autoencoder = Model(input_img, reconstruction)
        
        # Classifier generation
        z_clf = Input(shape=(self.latent_dim,))
        output_class = create_classifier(z_clf)
        self.classifier = Model(z_clf, output_class, name='class')
        
        classification = self.classifier(self.encoder(input_img))

        # MOCAE generation
        self.mocae = Model(input_img, [reconstruction, classification])
        
    def create_mocvae(self):
        input_img = Input(shape=(self.img_width,
                                 self.img_height,
                                 self.img_channels))

        # Encoder generation
        z_mean, z_log_sigma = create_encoder(input_img, vae=True)
        z = Lambda(sampling)([z_mean, z_log_sigma])
        
        self.encoder = Model(input_img, [z_mean, z_log_sigma, z])

        # Decoder generation
        z_dec = Input(shape=(self.latent_dim,))
        output_img = create_decoder(z_dec)
        self.decoder = Model(z_dec, output_img, name='rec')
        
        reconstruction = self.decoder(self.encoder(input_img)[2])
        
        # Autoencoder generation
        self.autoencoder = Model(input_img, reconstruction)

        # Classifier generation
        z_clf = Input(shape=(self.latent_dim,))
        output_class = create_classifier(z_clf)
        self.classifier = Model(z_clf, output_class, name='class')
        
        # Use z as the classifier input
        classification = self.classifier(self.encoder(input_img)[2])
        
        # MOCAE generation
        self.mocae = Model(input_img, [reconstruction, classification])
        
        # KLDivergence loss
        kl_loss = self.kl_loss(z_mean, z_log_sigma)
        self.mocae.add_loss(kl_loss)

    def kl_loss(self, z_mean, z_log_sigma):
        return -tf.reduce_mean(z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma) + 1) / 2