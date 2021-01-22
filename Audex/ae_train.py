#!/usr/bin/env python
  
from tensorflow.keras.datasets import mnist
from ae import Autoencoder

from pathlib  import Path
from datetime import timedelta
import time
import argparse
import numpy as np
import cmd
import sys
import os

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_audex import *

def process_clargs():
    # Calling with "-traindata_path /to/file" will expect to find the file in ./to directory.
    parser = argparse.ArgumentParser(description = '[TODO: Script description].')

    parser.add_argument("-traindata_path", default=Aimx.MOST_RECENT_OUTPUT, type = Path,
                        help = 'Path to the data file to be fed to the NN. Or use ' + Aimx.MOST_RECENT_OUTPUT +
                               ', which by design is the output of the previous step of dataset preprocessing.')

    parser.add_argument("-ann_type",   default = "aen", type=str, help = 'ANN type. Default is aen (autoencoder network).')
    parser.add_argument("-batch_size", default = 32,    type=int, help = 'Batch size.')
    parser.add_argument("-epochs",     default =  1,    type=int, help = 'Number of epochs to train.')
    parser.add_argument("-patience",   default = 10,    type=int, help = 'Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument("-mnist_size", default = 10,    type=int, help = 'Number of images to train on from the MNIST dataset.')
    parser.add_argument("-verbose",    default =  1,    type=int, help = 'Verbosity modes: 0 (silent), 1 (will show progress bar),'
                                                                         ' or 2 (one line per epoch). Default is 1.')

    parser.add_argument("-dim_latent", default = 10, type=int, help = 'Dimension of the latent space.')
    parser.add_argument("-showplot",   action ='store_true',   help = 'At the end, will show an interactive plot of the training history.')
    parser.add_argument("-savemodel",  action ='store_true',   help = 'Save a trained model in directory ' + quote(Aimx.Paths.GEN_SAVED_MODELS))
    parser.add_argument("-example",    action ='store_true',   help = 'Show a working example on how to call the script.')

    args = parser.parse_args()

    ########################## Command Argument Handling & Verification #######################

    if args.example:
        print_info(nameofthis(__file__) + " -epochs 5")
        exit()

    if provided(args.traindata_path) and not args.traindata_path.exists():
        if str(args.traindata_path) is not Aimx.MOST_RECENT_OUTPUT:
            raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.traindata_path)))

    # path to the traindata file that stores MFCCs and genre labels for each processed segment
    args.traindata_path = get_actual_traindata_path(args.traindata_path)

    if not args.savemodel and os.path.getsize(args.traindata_path) > 50_000_000: # > 50 Mb
        args.savemodel = prompt_user_warning("Attempting to train on a large >50Mb traindata without '-savemodel',"
                                             " would you rather save the final model? [yes / no] ")
        print_info("As requested, proceeding with -savemodel =", args.savemodel)
    
    ###########################################################################################
    
    print_script_start_preamble(nameofthis(__file__), vars(args))

    return args

LEARNING_RATE = 0.0005

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test  = x_test.astype("float32") / 255
    x_test  = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    args = process_clargs()

    inputshape = (28, 28, 1)
    autoencoder = Autoencoder(
        input_shape      = inputshape,
        conv_filters     = (32, 64, 64, 64), # 4 conv layers each with the corresponding number of filters
        # len() of tuples below must be at least that of the above, like here they are both of len() 4. Otherwise you'll get an error.
        conv_kernels     = (3, 3, 3, 3),
        conv_strides     = (1, 2, 2, 1),     # stride 2 in conv layers means downsampling (halving) at that point
        latent_space_dim = args.dim_latent
    )
    autoencoder.summary()
    autoencoder.compile(LEARNING_RATE)

    start_time = time.time()

    x_train, _, _, _ = load_mnist()

    history = autoencoder.train(x_train[:args.mnist_size], args.batch_size, args.epochs)

    training_duration = timedelta(seconds = round(time.time() - start_time))
    timestamp = timestamp_now()

    print_info("Finished {} at {} with wall clock time: {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp),
                                                                    lightyellow(training_duration)))

    trainid = args.ann_type + "_e" + str(args.epochs) + "_" + str(args.mnist_size) + "_mnist"

    # save as most recent training result metadata
    save_training_result_meta_ae(history, trainid, timestamp, str(training_duration), inputshape, args.savemodel)

    if (args.savemodel):
        save_model(autoencoder.model_ae, trainid)
        autoencoder.save()

    autoencoder2 = Autoencoder.load_custom()
    autoencoder2.summary()