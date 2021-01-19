#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.layers     import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras            import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses     import MeanSquaredError

import pickle
import numpy as np

import sys
import os
# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import *
from Audex.utils.utils_audex  import *

class Autoencoder:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture
    with mirrored encoder and decoder components.
    """
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape      = input_shape      # (28, 28, 1)
        self.conv_filters     = conv_filters     # (2, 4, 8)-tuple
        self.conv_kernels     = conv_kernels     # (3, 5, 3)-tuple
        self.conv_strides     = conv_strides     # (1, 2, 2)-tuple
        self.latent_space_dim = latent_space_dim # 2

        self.model_enc = None
        self.model_dec = None
        self.model_ae  = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input             = None

        self.MODEL_FULLPATH   = os.path.join(Aimx.Paths.GEN_SAVED_MODELS, "model_ae_trainid")
        self.PARAMS_FULLPATH  = os.path.join(self.MODEL_FULLPATH, "parameters.pkl")
        self.WEIGHTS_FULLPATH = os.path.join(self.MODEL_FULLPATH, "weights.h5")

        self._build()

    def summary(self):
        self.model_enc.summary()
        self.model_dec.summary()
        self.model_ae.summary()

    def compile(self, learning_rate=0.0001):
        opt = Adam(learning_rate=learning_rate)
        mse = MeanSquaredError()
        self.model_ae.compile(optimizer=opt, loss=mse)

    def train(self, x_train, batch_size, epochs):
        # Passing x_train as target data is essentially the trick to make this NN generative
        # For NC, pass noisy audio as x (input data) and clean audio as y (target data)
        self.model_ae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    def save(self):
        self._save_parameters()
        self._save_weights()

    def load_weights(self, weights_path):
        self.model_ae.load_weights(weights_path)

    def reconstruct(self, images):
        latent = self.model_enc.predict(images)
        genim  = self.model_dec.predict(latent)
        return genim, latent

    @classmethod
    def load(cls):
        MODEL_FULLPATH   = os.path.join(Aimx.Paths.GEN_SAVED_MODELS, "model_ae_trainid")
        PARAMS_FULLPATH  = os.path.join(MODEL_FULLPATH, "parameters.pkl")
        WEIGHTS_FULLPATH = os.path.join(MODEL_FULLPATH, "weights.h5")
        with open(PARAMS_FULLPATH, "rb") as f:
            params = pickle.load(f)
        autoencoder  = Autoencoder(*params)
        weights_path = os.path.join(MODEL_FULLPATH, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _save_parameters(self):
        params = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        Path(self.MODEL_FULLPATH).mkdir(parents=True, exist_ok=True)
        with open(self.PARAMS_FULLPATH, "wb") as f:
            pickle.dump(params, f)

    def _save_weights(self):
        self.model_ae.save_weights(self.WEIGHTS_FULLPATH)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input     = self._add_encoder_input()
        conv_layers       = self._add_conv_layers(encoder_input)
        bottleneck        = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.model_enc    = tf.keras.Model(encoder_input, bottleneck, name="ENCODER")

    def _build_decoder(self):
        decoder_input         = self._add_decoder_input()
        dense_layer           = self._add_dense_layer(decoder_input)
        reshape_layer         = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output        = self._add_decoder_output(conv_transpose_layers)
        self.model_dec        = tf.keras.Model(decoder_input, decoder_output, name="DECODER")

    def _build_autoencoder(self):
        model_input   = self._model_input
        model_output  = self.model_dec(self.model_enc(model_input))
        self.model_ae = tf.keras.Model(model_input, model_output, name="AUTOENCODER")

    def _add_encoder_input(self):
        return tf.keras.layers.Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """ Create all convolutional blocks in encoder. """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """ Add a convolutional block to a graph of layers, consisting of conv 2d + ReLU + batch normalization. """
        layer_number = layer_index + 1
        conv_layer = tf.keras.layers.Conv2D(
            filters     = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides     = self.conv_strides[layer_index],
            padding     = "same",
            name        = f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = tf.keras.layers.ReLU(              name = f"encoder_relu_{layer_number}")(x)
        x = tf.keras.layers.BatchNormalization(name = f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """ Flatten data and add bottleneck (Dense layer). """
        # Save the shape just before flattening
        # to be able to restore it later in the decoder
        self._shape_before_bottleneck = K.int_shape(x)[1:] # first dimension is batch size, ignore it and take only width, height and number of channels
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.latent_space_dim, name="encoder_output")(x)
        return x

    def _add_decoder_input(self):
        return tf.keras.layers.Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        # Here we want the same number of neurons as there are in the layer before the bottleneck,
        # but flattened. So for a shape of (1, 2, 4) we need 8 (which is the product of dimensions).
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = tf.keras.layers.Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return tf.keras.layers.Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """ Add conv transpose blocks. """
        # loop through all the conv layers in reverse order and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters          = self.conv_filters[layer_index],
            kernel_size      = self.conv_kernels[layer_index],
            strides          = self.conv_strides[layer_index],
            padding          = "same",
            name             = f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = tf.keras.layers.ReLU(              name = f"decoder_relu_{layer_num}")(x)
        x = tf.keras.layers.BatchNormalization(name = f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters          = 1,
            kernel_size      = self.conv_kernels[0],
            strides          = self.conv_strides[0],
            padding          = "same",
            name             = f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = tf.keras.layers.Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

#if __name__ == "__main__":
#    autoencoder = Autoencoder(
#        input_shape      = (28, 28, 1),
#        conv_filters     = (32, 64, 64, 64), # 4 conv layers each with the corresponding number of filters
#        # len() of tuples below must be at least that of the above, like here they are both of len() 4. Otherwise you'll get an error.
#        conv_kernels     = (3, 3, 3, 3),
#        conv_strides     = (1, 2, 2, 1),     # stride 2 in conv layers means downsampling (halving) at that point
#        latent_space_dim = 10
#    )
#    autoencoder.summary()