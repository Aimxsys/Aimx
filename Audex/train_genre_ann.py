from pathlib import Path
from datetime import timedelta
import time
import argparse
import cmd
import os

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

from common_utils import *
from  audex_utils import *

# Calling with "-traindata_path /to/file" will expect to find the file in ./to directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' audio files and their various spectrograms.')

parser.add_argument("-traindata_path", type = Path, help = 'Path to the data file to be fed to the NN. Or use ' + Aimx.MOST_RECENT_OUTPUT +
                                                           ', which by design is the output of the previous step of dataset preprocessing.')

parser.add_argument("-batch_size", default = 32, type=int, help = 'Batch size.')
parser.add_argument("-epochs",     default = 50, type=int, help = 'Number of epochs to train.')
parser.add_argument("-patience",   default =  5, type=int, help = 'Number of epochs with no improvement after which training will be stopped.')
parser.add_argument("-verbose",    default =  1, type=int, help = 'Verbosity modes: 0 (silent), 1 (will show progress bar),'
                                                                  ' or 2 (one line per epoch). Default is 1.')
parser.add_argument("-showplot",   action ='store_true',   help = 'At the end, will show an interactive plot of the training history.')
parser.add_argument("-savemodel",  action ='store_true',   help = 'Will save a trained model in directory ' + quote(Aimx.Paths.GEN_SAVED_MODELS))
parser.add_argument("-example",    action ='store_true',   help = 'Will show a working example on how to call the script.')

args = parser.parse_args()

########################## Command Argument Handling & Verification #######################

if args.example:
    print_info(nameofthis(__file__) + " -traindata_path most_recent_output -epochs 5")
    exit()

if provided(args.traindata_path) and not args.traindata_path.exists():
    if str(args.traindata_path) is not Aimx.MOST_RECENT_OUTPUT:
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.traindata_path)))

###########################################################################################

print_script_start_preamble(nameofthis(__file__), vars(args))

# path to the traindata file that stores MFCCs and genre labels for each processed segment
ARG_TRAINDATA_PATH = get_actual_traindata_path(args.traindata_path)

if not args.savemodel and os.path.getsize(ARG_TRAINDATA_PATH) > 100_000_000: # > 100 Mb
    args.savemodel = prompt_user_warning("Attempting to train on a large >100Mb traindata without '-savemodel',"
                                         " would you rather save the final model? [yes / no] ")
    print_info("As requested, proceeding with -savemodel =", args.savemodel)

if __name__ == "__main__":

    print_info("Dataset view (labels) from dataprep result meta:")
    cmd.Cmd().columnize(get_dataprep_result_meta()[Aimx.Dataprep.DATASET_VIEW], displaywidth=100)

    inputs, labels = load_traindata(args.traindata_path)

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

    earlystop_callback = keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=args.patience)

    start_time = time.time()

    # train model
    history = model.fit(inputs_train, labels_train, validation_data = (inputs_test, labels_test),
                        batch_size = args.batch_size,
                        epochs     = args.epochs,
                        verbose    = args.verbose,
                        callbacks  = [earlystop_callback])

    training_duration = timedelta(seconds = round(time.time() - start_time))
    timestamp = timestamp_now()

    print_info("Finished {} at {} with wall clock time: {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp),
                                                                    lightyellow(training_duration)))

    trainid = "ann_e" + str(args.epochs) + "_" + extract_filename(ARG_TRAINDATA_PATH)

    # save as most recent training result metadata
    save_training_result_meta(trainid, timestamp, str(training_duration), args.savemodel)

    if (args.savemodel):
        save_model(model, trainid)
    
    plot_history(history, trainid, args.showplot) # accuracy and error as a function of epochs