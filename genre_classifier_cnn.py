from pathlib import PurePath
from pathlib import Path
import argparse
import librosa
import numpy as np
import time
import json
import math
import os

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from termcolor import colored

from genre_classifier import load_data
from genre_utils      import *

# Calling with "-data_path /to/file" will expect to find the file in ./to directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' audio files and their various spectrograms.')

parser.add_argument("-data_path", type = Path, help = 'Path to the data file to be fed to the NN. Or use "most_recent_output", which'
                                                      ' by design is the output of the previous step of dataset preprocessing.')
parser.add_argument("-batch_size", default = 32, type=int, help = 'Batch size.')
parser.add_argument("-epochs",     default = 50, type=int, help = 'Number of epochs to train.')
parser.add_argument("-verbose",    default =  1, type=int, help = 'Verbosity modes: 0 (silent), 1 (will show progress bar),'
                                                                  ' or 2 (one line per epoch). Default is 1.')
parser.add_argument("-noplot",     action ='store_true',   help = 'Will not show any plots (useful for certain test automation).')

args = parser.parse_args()

############################## Command Argument Verification ##############################

if provided(args.data_path) and not args.data_path.exists():
    if str(args.data_path) is not "most_recent_output":
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.data_path)))

###########################################################################################

# path to json file that stores MFCCs and genre labels for each processed segment
ARG_DATA_PATH = args.data_path if provided(args.data_path) else ""
if str(ARG_DATA_PATH) == "most_recent_output":
    ARG_DATA_PATH = get_preprocess_result_meta()["most_recent_output"]

def prepare_datasets(test_size, valid_size):
    """
    Loads data and splits it into train, validation and test sets.
    Params:
         test_size (float): Value in [0, 1] indicating percentage of dataset to allocate to test       split
        valid_size (float): Value in [0, 1] indicating percentage of dataset to allocate to validation split
    Returns:
        x_train (ndarray): Input training set
        x_valid (ndarray): Input valid set
        x_test  (ndarray): Input test set
        y_train (ndarray): Target training set
        y_valid (ndarray): Target valid set
        y_test  (ndarray): Target test set
    """
    x, y = load_data(ARG_DATA_PATH)

    # create train, validation and test split
    x_train, x_test,  y_train, y_test  = train_test_split(x,       y,       test_size = test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = valid_size)

    # add an axis to input sets; example resulting shape: (89, 259, 13, 1)
    x_train = x_train[..., np.newaxis]
    x_valid = x_valid[..., np.newaxis]
    x_test  =  x_test[..., np.newaxis]

    print_info("Extended x_train (input) shape: " + str(x_train.shape))
    print_info("Extended x_valid (input) shape: " + str(x_valid.shape))
    print_info("Extended x_test  (input) shape: " + str(x_test.shape))

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def build_model(input_shape):
    """
    Generates CNN model
    Param:
        input_shape (tuple): Shape of input set
    Returns:
        model: CNN model
    """
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, x, y):
    """
    Predict a single sample using the trained model
    Params:
        model: Trained classifier
        x: Input data
        y (int): Target
    """
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    x = x[np.newaxis, ...] # change array shape from (130, 13, 1) to (1, 130, 13, 1)

    prediction = model.predict(x)
    predicted_index = np.argmax(prediction, axis=1) # index with max value

    print_info("Prediction: ", prediction)
    print_info("Target: {} = {}, Predicted label: {} = {}".format(y, to_genre_name(y), predicted_index[0], to_genre_name(predicted_index[0])))

if __name__ == "__main__":

    # get train, validation, test splits
    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_datasets(test_size = 0.25, valid_size = 0.2)

    # create network
    model = build_model(input_shape = (x_train.shape[1], x_train.shape[2], 1))

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
                  loss      = 'sparse_categorical_crossentropy',
                  metrics   = ['accuracy'])

    model.summary()

    # train model
    history = model.fit(x_train, y_train, validation_data = (x_valid, y_valid), batch_size=args.batch_size, epochs=args.epochs)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 1)
    print_info('\nTest accuracy:', test_acc)
        
    # pick a sample to predict from the test set
    x_to_predict = x_test[30]
    y_to_predict = y_test[30] # target

    # predict sample
    predict(model, x_to_predict, y_to_predict)

    # plot accuracy/error for training and validation
    if not args.noplot:
        plot_history(history)