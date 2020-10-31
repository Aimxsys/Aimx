﻿from pathlib import Path
from pathlib import PurePath
import argparse
import json
import os
import math
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

from preprocess_utils import *

# Calling with "-data_path /to/file" will expect to find the file in ./to directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' audio files and their various spectrograms.')

parser.add_argument("-data_path", type = Path, help = 'Path to the data file to be fed to the NN.')

args = parser.parse_args()

############################## Command Argument Verification ##############################

if provided(args.data_path) and not args.data_path.exists():
    raise FileNotFoundError("Provided file " + quote(str(args.data_path)) + " not found.")

###########################################################################################

# path to json file that stores MFCCs and genre labels for each processed segment
PAR_DATA_PATH  = args.data_path if provided(args.data_path) else ""

def load_data(dataset_path):
    """
    Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    try:
        with open(dataset_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Provided dataset file " + quote(dataset_path) + " not found.")
        exit()

    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])
    print("Data loaded succesfully.")
    return  inputs, labels

if __name__ == "__main__":

    inputs, labels = load_data(PAR_DATA_PATH)

    # create train/test split
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size = 0.3)

    # build network topology
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation = 'relu'),
        keras.layers.Dense(256, activation = 'relu'),
        keras.layers.Dense( 64, activation = 'relu'),
        keras.layers.Dense( 10, activation = 'softmax')
    ])

    # compile model
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
                  loss      = 'sparse_categorical_crossentropy',
                  metrics   = ['accuracy'])

    model.summary()

    # train model
    history = model.fit(inputs_train, labels_train, validation_data = (inputs_test, labels_test), batch_size = 32, epochs = 50)