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
    parser = argparse.ArgumentParser(description = '[TODO: Script description].')

    parser.add_argument("-ann_type",      default = "aen",  type=str,   help = 'ANN type. Default is aen (autoencoder network).')
    parser.add_argument("-batch_size",    default = 32,     type=int,   help = 'Batch size.')
    parser.add_argument("-epochs",        default =  1,     type=int,   help = 'Number of epochs to train.')
    parser.add_argument("-learning_rate", default = 0.0005, type=float, help = 'Number of epochs to train.')
    parser.add_argument("-patience",      default = 10,     type=int,   help = 'Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument("-mnist_size",    default = 10,     type=int,   help = 'Number of images to train on from the MNIST dataset.')
    parser.add_argument("-verbose",       default =  1,     type=int,   help = 'Verbosity modes: 0 (silent), 1 (will show progress bar), or 2 (one line per epoch). Default is 1.')
    parser.add_argument("-note",          default = "",     type=str,   help = 'Short note to appear inside trainid.')

    parser.add_argument("-dim_latent",  default =  10, type=int, help = 'Dimension of the latent space.')
    parser.add_argument("-downscale",   default = 255, type=int, help = 'Factor by which to scale down the data.')
    parser.add_argument("-fixtarget",   action ='store_true',   help = 'Will train on a fixed target.')
    parser.add_argument("-showplot",    action ='store_true',   help = 'At the end, will show an interactive plot of the training history.')
    parser.add_argument("-savemodel",   action ='store_true',   help = 'Save a trained model in directory ' + quote(Aimx.Paths.GEN_SAVED_MODELS))
    parser.add_argument("-noquestions", action ='store_true',   help = 'Don\'t ask any questions.')
    parser.add_argument("-example",     action ='store_true',   help = 'Show a working example on how to call the script.')

    args = parser.parse_args()

    ########################## Command Argument Handling & Verification #######################

    if args.example:
        print_info(nameofthis(__file__) + " -mnist_size 5000 -dim_latent 10 -epochs 5")
        exit()

    if not args.noquestions:
        if not args.savemodel and args.mnist_size * args.epochs > 10_000:
            args.savemodel = prompt_user_warning("Attempting a likely long training session without '-savemodel',"
                                                 " would you rather save the final model? [yes / no] ")
            print_info("As requested, proceeding with -savemodel =", args.savemodel)
    
    ###########################################################################################
    
    print_script_start_preamble(nameofthis(__file__), vars(args))

    return args

def downscale_traindata_pixels(x_train, y_train, x_test, y_test, downscale):
    x_train = x_train.astype("float32") / downscale
    x_test  =  x_test.astype("float32") / downscale
    return x_train, y_train, x_test, y_test

def reshape_traindata(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test  =  x_test.reshape(x_test.shape  + (1,))
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    args = process_clargs()

    inputshape = (28, 28, 1)
    model = Autoencoder(
        input_shape  = inputshape,
        conv_filters = (32, 64, 64, 64), # 4 conv layers each with the corresponding number of filters
        # len() of tuples below must be at least that of the above, like here they are both of len() 4. Otherwise you'll get an error.
        conv_kernels = (3, 3, 3, 3),
        conv_strides = (1, 2, 2, 1),     # stride 2 in conv layers means downsampling (halving) at that point
        dim_latent   = args.dim_latent
    )
    model.summary()
    model.compile(args.learning_rate)

    earlystop_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.001, patience=args.patience)

    start_time = time.time()

    # load, reshape, downscale
    (x_inputs, _), (_, _) = mnist.load_data() # traindata                x_inputs.shape == (60000, 28, 28)
    x_inputs,  _,   _, _  = reshape_traindata(x_inputs, _, _, _) #       x_inputs.shape == (60000, 28, 28, 1)
    x_inputs,  _,   _, _  = downscale_traindata_pixels(x_inputs, _, _, _, args.downscale)

    x_inputs = x_inputs[:args.mnist_size]
    if args.fixtarget:
        x_targets = x_inputs[0]
        x_targets = x_targets[np.newaxis, ...]                      # turn into      (1, 28, 28, 1)
        x_targets = np.repeat(x_targets, x_inputs.shape[0], axis=0) # replicate into (n, 28, 28, 1) to match the x_inputs stack
    else:
        x_targets = x_inputs

    #plot_matrices_single_chart([x_inputs.squeeze()[0], x_targets.squeeze()[0]], ["input", "target"], "mnist")

    history = model.train(x_inputs, x_targets, args.batch_size, args.epochs, [earlystop_callback])

    training_duration = timedelta(seconds = round(time.time() - start_time))
    timestamp = timestamp_now()

    print_info("Finished {} at {} with wall clock time: {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp),
                                                                    lightyellow(training_duration)))

    trainid = args.ann_type + "_x" + str(args.dim_latent) + "_e" + str(args.epochs) + "_" + args.note + "_" + str(args.mnist_size) + "d_" + "mnist"

    # save as most recent training result metadata
    save_training_result_meta_ae(history, trainid, timestamp, str(training_duration), inputshape, args.dim_latent, args.downscale,
                                 cmdline   = nameofthis(__file__) + " " + " ".join(sys.argv[1:]),
                                 savemodel = args.savemodel)
    if (args.savemodel):
        model.save_model(trainid)

    plot_history_ae(history, trainid, args.showplot) # loss as a function of epochs