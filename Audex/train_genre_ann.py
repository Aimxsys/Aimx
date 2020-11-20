from pathlib import PurePath
from pathlib import Path
import argparse
import numpy as np
import time
import json
import math
import os

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from termcolor import colored

from common_utils import *
from  genre_utils import *

# Calling with "-data_path /to/file" will expect to find the file in ./to directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' audio files and their various spectrograms.')

parser.add_argument("-data_path", type = Path, help = 'Path to the data file to be fed to the NN. Or use "most_recent_output", which'
                                                      ' by design is the output of the previous step of dataset preprocessing.')
parser.add_argument("-batch_size", default = 32, type=int, help = 'Batch size.')
parser.add_argument("-epochs",     default = 50, type=int, help = 'Number of epochs to train.')
parser.add_argument("-verbose",    default =  1, type=int, help = 'Verbosity modes: 0 (silent), 1 (will show progress bar),'
                                                                  ' or 2 (one line per epoch). Default is 1.')
parser.add_argument("-showplot",   action ='store_true',   help = 'Will not show any plots (useful for certain test automation).')

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

def load_data(data_path):
    """
    Loads training data from json file and reads them into arrays for NN processing.
        :param data_path (str): Path to json file containing data
        :return inputs (ndarray: the "mfcc"   section in the json data) 
        :return labels (ndarray: the "labels" section in the json data, one label per segment)
    """
    try:
        with open(data_path, "r") as file:
            timestamp = str(time.ctime(os.path.getmtime(data_path)))
            m = "most recent [" + timestamp + "] " if str(args.data_path) == "most_recent_output" else ""
            print_info("\n|||||| Loading " + m + "data file " + quote(cyansky(data_path)) + "...", end="")
            data = json.load(file)
            print_info(" [DONE]")
    except FileNotFoundError:
        print_info("Data file " + quote(data_path) + " not provided or not found. Exiting...")
        exit() # cannot proceed without data file

    # convert lists to numpy arrays
    print_info("Reading data...", end="")
    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])
    print_info(" [DONE]\n")
    return inputs, labels

if __name__ == "__main__":

    inputs, labels = load_data(ARG_DATA_PATH)

    # create train/test split
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size = 0.3)

    # build network topology
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation = 'relu'),#, kernel_regularizer = keras.regularizers.l2(0.001)),
        #keras.layers.BatchNormalization(axis = 1),
        #keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation = 'relu'),#, kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(axis = 1),
        keras.layers.Dropout(0.3),
        keras.layers.Dense( 64, activation = 'relu'),#, kernel_regularizer = keras.regularizers.l2(0.001)),
        #keras.layers.BatchNormalization(axis = 1),
        #keras.layers.Dropout(0.3),
        keras.layers.Dense( 10, activation = 'softmax')
    ])

    # compile model
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
                  loss      = 'sparse_categorical_crossentropy',
                  metrics   = ['accuracy'])

    model.summary()

    # train model
    history = model.fit(inputs_train, labels_train, validation_data = (inputs_test, labels_test),
                        batch_size = args.batch_size,
                        epochs     = args.epochs,
                        verbose    = args.verbose)
    
    trainid = "ann_e" + str(args.epochs) + "_"

    save_current_model(model, trainid + extract_filename(os.path.basename(__file__)))

    plot_history(history, trainid, args.showplot) # accuracy and error as a function of epochs