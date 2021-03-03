#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.layers     import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses     import MeanSquaredError

import pickle
import numpy as np
import argparse

import sys
import os
# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import *
from Audex.utils.utils_audex  import *

def process_clargs():
    parser = argparse.ArgumentParser(description = 'This scrip launches an ASR client.')
    
    parser.add_argument("-sample_arch", action  ='store_true', help='Show an example model architecture summary.')
    
    args = parser.parse_args()
    return args

class Autoencoder:
    FILENAME_WEIGHTS  = "weights.h5"
    FILENAME_HYPARAMS = "hyparams.pkl"
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture
    with mirrored encoder and decoder components.
    """
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, dim_latent):
        self.input_shape  = input_shape  # (28, 28, 1)
        self.conv_filters = conv_filters # (2, 4, 8)-tuple
        self.conv_kernels = conv_kernels # (3, 5, 3)-tuple
        self.conv_strides = conv_strides # (1, 2, 2)-tuple
        self.dim_latent   = dim_latent   # 2

        self.model_enc = None
        self.model_dec = None
        self.model_ae  = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input             = None

        self._build()

    def summary(self):
        self.model_enc.summary()
        self.model_dec.summary()
        self.model_ae.summary()

    def compile(self, learning_rate=0.0001):
        opt = Adam(learning_rate=learning_rate)
        mse = MeanSquaredError()
        self.model_ae.compile(optimizer=opt, loss=mse)

    def train(self, inputs, targets, batch_size, epochs, callbacks):
        # Example inputs.shape == (1660, 44, 16, 1)
        # Passing inputs as targets data is essentially the trick to make this NN generative
        # For NC, pass noisy audio as input data and clean audio as target data

        # x = np.full_like(inputs, 0.3) # experimentation with a fixed target
        
        # Fixed audio signum target
        #x = inputs[40]
        #x = np.repeat(x[np.newaxis, :, :, :], inputs.shape[0], axis=0) # repeated array

        print_info("Training AE on data of shapes:")
        print_info("Inputs: ", inputs.shape)
        print_info("Targets:", targets.shape)
        history = self.model_ae.fit(inputs, targets, batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=callbacks)
        return history

    def save_model(self, trainid):
        MODEL_FULLPATH        = os.path.join(Aimx.Paths.GEN_SAVED_MODELS, "model_" + trainid)
        MODEL_PARAMS_FULLPATH = os.path.join(MODEL_FULLPATH, self.FILENAME_HYPARAMS)
        MODEL_ASSETS_FULLPATH = os.path.join(MODEL_FULLPATH, "assets")
        Path(MODEL_FULLPATH).mkdir(parents=True, exist_ok=True)
        params = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.dim_latent
        ]
        # Save parameters
        print_info("|||||| Saving model", quote_path(MODEL_FULLPATH), "... ", end="")
        with open(MODEL_PARAMS_FULLPATH, "wb") as f:
            pickle.dump(params, f)
        
        # Save weights
        self.model_ae.save_weights(os.path.join(MODEL_FULLPATH, self.FILENAME_WEIGHTS))
        print_info("[DONE]")

        # Save model quicksetup
        MODEL_QUICKSETUP_FULLPATH = os.path.join(WORKDIR, trainid + (".bat" if windows() else ".sh"))
        with open(MODEL_QUICKSETUP_FULLPATH, "w") as f:
            print_info("|||||| Writing file", quote_path(MODEL_QUICKSETUP_FULLPATH), "... ", end="")
            MODEL_METAS_FULLPATH = os.path.join(MODEL_ASSETS_FULLPATH, "*.json")
            if windows():
                content = "xcopy " + dquote(MODEL_METAS_FULLPATH) + "^\n\t  " + dquote(WORKDIR) + "^\n\t" + "/K /H /Y"
            else: # Linux
                content = "cp " + MODEL_METAS_FULLPATH + " \\\n   " + WORKDIR
            f.write(content)
            print_info("[DONE]")
        
        # Save assets
        Path(MODEL_ASSETS_FULLPATH).mkdir(parents=True, exist_ok=True) # create model assets directory (similar to TF2.x default's)
        print_info("|||||| Copying file", quote_path(Aimx.Dataprep.RESULT_METADATA_FULLPATH), "into model assets... ", end="")
        copy2(Aimx.Dataprep.RESULT_METADATA_FULLPATH, MODEL_ASSETS_FULLPATH)
        print_info("[DONE]")
        print_info("|||||| Copying file", quote_path(Aimx.Training.RESULT_METADATA_FULLPATH), "into model assets... ", end="")
        copy2(Aimx.Training.RESULT_METADATA_FULLPATH, MODEL_ASSETS_FULLPATH)
        print_info("[DONE]")
        print_info("|||||| Copying file", quote_path(MODEL_QUICKSETUP_FULLPATH), "into model assets... ", end="")
        copy2(MODEL_QUICKSETUP_FULLPATH, MODEL_ASSETS_FULLPATH)
        print_info("[DONE]")

    def load_weights(self, weights_path):
        self.model_ae.load_weights(weights_path)

    def regen(self, images):
        vencs  = self.model_enc.predict(images) # encode into latent space
        genums = self.model_dec.predict(vencs)  # decode from latent space into genum
        return vencs, genums

    def gen_random(self, n):
        vencs  = np.random.rand(n, self.dim_latent)  # n 1d arrays of size dim_latent
        genums = self.model_dec.predict(vencs)
        return vencs, genums

    @classmethod
    def load_model(cls, model_path):
        PARAMS_FULLPATH  = os.path.join(model_path, cls.FILENAME_HYPARAMS)
        WEIGHTS_FULLPATH = os.path.join(model_path, cls.FILENAME_WEIGHTS)
        try:
            print_info("|||||| Loading model " + quote_path(model_path) + "... ", end="")
            with open(PARAMS_FULLPATH, "rb") as p:
                params = pickle.load(p)
            ae = Autoencoder(*params)
            ae.load_weights(WEIGHTS_FULLPATH)
            print_info("[DONE (model loaded)]", quote_path(model_path))
        except Exception as e:
            print(pinkred("\nException caught while trying to load the model: " + quote_path(model_path)))
            print(pinkred("Exception message: ") + red(str(e)))
        return ae

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
        self._shape_before_bottleneck = tf.keras.backend.int_shape(x)[1:] # first dimension is batch size, ignore it and take only width, height and number of channels
        
        x = tf.keras.layers.Flatten()(x)
        #n = np.prod(self._shape_before_bottleneck)
        #x = tf.keras.layers.Dense(n/2)(x)
        #x = tf.keras.layers.Dense(n/4)(x)
        #x = tf.keras.layers.Dense(n/8)(x)
        x = tf.keras.layers.Dense(self.dim_latent, name="encoder_output")(x)
        #x = tf.keras.layers.Dense(self.dim_latent, name="encoder_output2")(x)
        #x = tf.keras.layers.Dense(self.dim_latent, name="encoder_output3")(x)
        #x = tf.keras.layers.Dense(self.dim_latent, name="encoder_output4")(x)
        #x = tf.keras.layers.Dense(self.dim_latent, name="encoder_output5")(x)
        return x

    def _add_decoder_input(self):
        return tf.keras.layers.Input(shape=self.dim_latent, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        # Here we want the same number of neurons as there are in the layer before the bottleneck,
        # but flattened. So for a shape of (1, 2, 4) we need 8 (which is the product of dimensions).
        num_neurons = np.prod(self._shape_before_bottleneck)
        #n = np.prod(self._shape_before_bottleneck)
        #dense_layer = tf.keras.layers.Dense(n/8)(decoder_input)
        #dense_layer = tf.keras.layers.Dense(n/4)(dense_layer)
        #dense_layer = tf.keras.layers.Dense(n/2)(dense_layer)
        #dense_layer = tf.keras.layers.Dense(num_neurons, name="decoder_dense")(dense_layer)
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

if __name__ == "__main__":
    args = process_clargs()

    if args.sample_arch:
        autoencoder = Autoencoder(
            input_shape      = (28, 28, 1),
            conv_filters     = (32, 64, 64, 64), # 4 conv layers each with the corresponding number of filters
            # len() of tuples below must be at least that of the above, like here they are both of len() 4. Otherwise you'll get an error.
            conv_kernels     = (3, 3, 3, 3),
            conv_strides     = (1, 2, 2, 1),     # stride 2 in conv layers means downsampling (halving) at that point
            dim_latent = 10
        )
        autoencoder.summary()