#!/usr/bin/env python
  
from tensorflow.keras.datasets import mnist
from ae       import Autoencoder
from pathlib  import Path
from datetime import timedelta

import librosa
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
from asr_train import prepare_traindata

def process_clargs():
    # Calling with "-traindata_path /to/file" will expect to find the file in ./to directory.
    parser = argparse.ArgumentParser(description = '[TODO: Script description].')

    parser.add_argument("-traindata_path", default=Aimx.MOST_RECENT_OUTPUT, type = Path,
                        help = 'Path to the data file to be fed to the NN. Or use ' + Aimx.MOST_RECENT_OUTPUT +
                               ', which by design is the output of the previous step of dataset preprocessing.')

    parser.add_argument("-ann_type",      default = "aen",   type=str,   help = 'ANN type. Default is aen (autoencoder network).')
    parser.add_argument("-batch_size",    default = 32,      type=int,   help = 'Batch size.')
    parser.add_argument("-epochs",        default =  1,      type=int,   help = 'Number of epochs to train.')
    parser.add_argument("-learning_rate", default =  0.0005, type=float, help = 'Number of epochs to train.')
    parser.add_argument("-patience",      default = 10,      type=int,   help = 'Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument("-verbose",       default =  1,      type=int,   help = 'Verbosity modes: 0 (silent), 1 (will show progress bar), or 2 (one line per epoch). Default is 1.')
    parser.add_argument("-note",          default = "",      type=str,   help = 'Short note to appear inside trainid.')

    parser.add_argument("-dim_latent",  default = 10, type=int, help = 'Dimension of the latent space.')
    parser.add_argument("-showplot",    action ='store_true',   help = 'At the end, will show an interactive plot of the training history.')
    parser.add_argument("-savemodel",   action ='store_true',   help = 'Save a trained model in directory ' + quote(Aimx.Paths.GEN_SAVED_MODELS))
    parser.add_argument("-noquestions", action ='store_true',   help = 'Don\'t ask any questions.')

    parser.add_argument("-example",     action ='store_true',   help = 'Show a working example on how to call the script.')

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

    if not args.noquestions:
        if not args.savemodel:
            if os.path.getsize(args.traindata_path) > 50_000_000: # > 50 Mb
                 args.savemodel = prompt_user_warning("Attempting to train on a large >50Mb traindata without '-savemodel',"
                                                      " would you rather save the final model? [yes / no] ")

            print_info("As requested, proceeding with -savemodel =", args.savemodel)
    
    ###########################################################################################
    
    print_script_start_preamble(nameofthis(__file__), vars(args))

    return args

def prepare_traindata(traindata_path):
    x_inputs, _ = load_traindata(traindata_path) # x = inputs, y = labels (here ignored)
    x_inputs = x_inputs[..., np.newaxis] # example resulting shape: (89, 259, 13, 1) for (signals, mfccvectors, mfccs, depth)
    print_info("Final prepared traindata inputs shape: " + str(x_inputs.shape))
    return x_inputs

if __name__ == "__main__":
    args = process_clargs()

    # get train, validation, test splits
    x_inputs = prepare_traindata(args.traindata_path)

    x_targets = prepare_traindata("../workdir/gen_traindata/1v_1d_one_2048w_512h_1i_22050r_1s.json")
    x_targets = np.repeat(x_targets, x_inputs.shape[0], axis=0) # repeated array
    #x_targets = librosa.util.normalize(x_targets)
    deprint(x_targets.shape, "x_targets repeated & final shape")

    inputshape = (x_inputs.shape[1], x_inputs.shape[2], 1) # x_train.shape == (11, 44, 128, 1) for (signals, mfccvectors, mfccs, depth)

    model = Autoencoder(
        input_shape  = inputshape,       # (44, 128, 1)
        conv_filters = (32, 64, 64, 64), # 4 conv layers each with the corresponding number of filters
        # len() of tuples below must be at least that of the above, like here they are both of len() 4. Otherwise you'll get an error.
        conv_kernels = (3, 3, 3, 3),
        conv_strides = (1, 2, 2, 1),     # stride 2 in conv layers means downsampling (halving) at that point
        dim_latent   = args.dim_latent
    )
    model.summary()
    model.compile(args.learning_rate)

    earlystop_callback = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.001, patience=args.patience)

    x_inputs = librosa.util.normalize(x_inputs)

    start_time = time.time()

    # Train
    history = model.train(x_inputs, x_targets, args.batch_size, args.epochs, [earlystop_callback])

    training_duration = timedelta(seconds = round(time.time() - start_time))
    timestamp = timestamp_now()

    print_info("Finished {} at {} with wall clock time: {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp),
                                                                    lightyellow(training_duration)))

    trainid = args.ann_type + "_x" + str(args.dim_latent) + "_e" + str(args.epochs) + "_" + args.note + "_" + extract_filename(args.traindata_path)

    # save as most recent training result metadata
    save_training_result_meta_ae(history, trainid, timestamp, str(training_duration), inputshape, args.dim_latent,
                                 cmdline   = nameofthis(__file__) + " " + " ".join(sys.argv[1:]),
                                 savemodel = args.savemodel)
    if (args.savemodel):
        model.save_model(trainid)

    plot_history_ae(history, trainid, args.showplot) # loss as a function of epochs