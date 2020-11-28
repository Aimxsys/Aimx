from pathlib import Path

import tensorflow as tf
import numpy as np
import time
import json
import os

import matplotlib.pyplot as pt

from common_utils import *

WORKDIR = os.path.join(Path().resolve().parent, "workdir")

class Aimx:
    class Paths:
        GEN_PLOTS        = os.path.join(WORKDIR, "gen_plots")
        GEN_SAVED_MODELS = os.path.join(WORKDIR, "gen_models")
        GEN_TRAINDATA    = os.path.join(WORKDIR, "gen_traindata")
    
    class Dataprep:
        RESULT_METADATA_FULLPATH = os.path.join(WORKDIR, "dataprep_result_meta.json")
        MOST_RECENT_OUTPUT = "most_recent_output"
        DATASET_VIEW       = "dataset_view"
        DURATION           = "duration"
        ALL_DIR_LABELS     = "alldirlabs"

    class TrainData:
        DURATION = "duration"
        MAPPING  = "mapping"
        LABELS   = "labels"
        FILES    = "files"
        MFCC     = "mfcc"

    TIMESTAMP = "timestamp"

def to_genre_name(label_id):
    return [
        'blues'    ,
        'classical',
        'country'  ,
        'disco'    ,
        'hiphop'   ,
        'jazz'     ,
        'metal'    ,
        'pop'      ,
        'reggae'   ,
        'rock'      
    ][label_id]

def get_preprocess_result_meta():
    if not hasattr(get_preprocess_result_meta, "cached"):
        with open(Aimx.Dataprep.RESULT_METADATA_FULLPATH, "r") as file:
            print_info("\n|||||| Loading file " + quote(cyansky(Aimx.Dataprep.RESULT_METADATA_FULLPATH)) + "... ", end="")
            preprocess_result_meta = json.load(file)
            print_info("[DONE]")
        get_preprocess_result_meta.cached = preprocess_result_meta
    return get_preprocess_result_meta.cached

def save_dataprep_result_meta(traindata_filename, dataset_view, dataprep_duration):
    prep_result_meta = {
        Aimx.Dataprep.MOST_RECENT_OUTPUT: {},
        Aimx.Dataprep.DATASET_VIEW:       {},
        Aimx.TIMESTAMP:                   {},
        Aimx.Dataprep.DURATION:           {}
    }
    prep_result_meta[Aimx.Dataprep.MOST_RECENT_OUTPUT] = os.path.join(Aimx.Paths.GEN_TRAINDATA, traindata_filename)
    prep_result_meta[Aimx.Dataprep.DATASET_VIEW] = dataset_view
    prep_result_meta[Aimx.TIMESTAMP]             = timestamp_now()
    prep_result_meta[Aimx.Dataprep.DURATION]     = dataprep_duration
    with open(Aimx.Dataprep.RESULT_METADATA_FULLPATH, 'w') as fp: 
        print_info("\n|||||| Writing file", quote(cyansky(Aimx.Dataprep.RESULT_METADATA_FULLPATH)), "... ", end="")
        json.dump(prep_result_meta, fp, indent=4)
        print_info("[DONE]")

def get_actual_traindata_path(arg_traindata_path):
    # Handle any special requests (most recent, largest, smallest, etc.)
    if str(arg_traindata_path) == Aimx.Dataprep.MOST_RECENT_OUTPUT:
        return get_preprocess_result_meta()[Aimx.Dataprep.MOST_RECENT_OUTPUT]
    return arg_traindata_path # no special requests, return pristine

def load_traindata(arg_traindata_path):
    """
    Loads training data from json file and reads them into arrays for NN processing.
        :param data_path (str): Path to json file containing traindata
        :return inputs (ndarray: the "mfcc"   section in the json traindata) 
        :return labels (ndarray: the "labels" section in the json traindata, one label per segment)
    """
    actual_traindata_path = get_actual_traindata_path(arg_traindata_path)
    try:
        with open(actual_traindata_path, "r") as file:
            timestamp = str(time.ctime(os.path.getmtime(actual_traindata_path)))
            m = "most recent [" + timestamp + "] " if str(arg_traindata_path) == Aimx.Dataprep.MOST_RECENT_OUTPUT else ""
            print_info("\n|||||| Loading " + m + "file " + quote(cyansky(actual_traindata_path)) + "... ", end="")
            traindata = json.load(file)
            print_info("[DONE]")            
    except FileNotFoundError:
        print_info("Data file " + quote(actual_traindata_path) + " not provided or not found. Exiting...")
        exit() # cannot proceed without traindata file
    
    print_info("Reading traindata... ", end="")
    inputs = np.array(traindata["mfcc"])   # convert the list to numpy array (MFCCs  turn into a 2d array)
    labels = np.array(traindata["labels"]) # convert the list to numpy array (labels turn into a 1d array)
    print_info("[DONE]\n")

    return inputs, labels

def predict(model, x, y):
    """
    Predict a single sample using the trained model
    Params:
        model: Trained classifier
        x: Input data
        y (int): Target
    """
    # add a dimension to input traindata for sample - model.predict() expects a 4d array in this case
    x = x[np.newaxis, ...] # change array shape from (130, 13, 1) to (1, 130, 13, 1)

    prediction = model.predict(x)
    predicted_index = np.argmax(prediction, axis=1) # index with max value

    print_info("Prediction: ", prediction)
    print_info("Target: {} = {}, Predicted label: {} = {}".format(y, to_genre_name(y), predicted_index[0], to_genre_name(predicted_index[0])))

def plot_history(history, trainid, show_interactive):
    """ Plots accuracy/loss for training/validation set as a function of epochs
        :param history: Training history of model
    """
    fig, axs = pt.subplots(2, figsize=(8, 6))
    traindata_filename = get_preprocess_result_meta()[Aimx.Dataprep.MOST_RECENT_OUTPUT]
    fig.canvas.set_window_title("Accuracy & Error - " + trainid)
    fig.suptitle(trainid, fontsize=12)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"],     label="train")
    axs[0].plot(history.history["val_accuracy"], label="test")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")

    # create error sublpot
    axs[1].plot(history.history["loss"],     label="train")
    axs[1].plot(history.history["val_loss"], label="test")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")

    # save the plot as most recent (often useful when comparing to a next NN run)
    Path(Aimx.Paths.GEN_PLOTS).mkdir(parents=True, exist_ok=True)
    MR_PLOT_FULLPATH = os.path.join(Aimx.Paths.GEN_PLOTS, trainid + ".png")
    print_info("\n|||||| Saving file", quote(cyansky(MR_PLOT_FULLPATH)), "... ", end="")
    pt.savefig(MR_PLOT_FULLPATH)
    print_info("[DONE]")

    if show_interactive:
        pt.show()

def save_model(model, model_id):
    dataprep_result_meta = tf.saved_model.Asset(Aimx.Dataprep.RESULT_METADATA_FULLPATH)
    trackable_obj  = tf.train.Checkpoint()
    trackable_obj.filename = dataprep_result_meta

    MODEL_FULLPATH = os.path.join(Aimx.Paths.GEN_SAVED_MODELS, "model_" + model_id)
    print_info("\n|||||| Saving model ", quote(cyansky(MODEL_FULLPATH)), "... ", end="")
    model.save(MODEL_FULLPATH)
    tf.saved_model.save(trackable_obj, MODEL_FULLPATH) # save the asset
    print_info("[DONE]")

def compose_traindata_id(dataset_depth, dataset_view, dataset_path, n_mfcc, n_fft, hop_length, num_segments, sample_rate, track_duration):
    traindata_id =  str(len(dataset_view)) + "v_"
    traindata_id += str(dataset_depth)     + "d_"
    traindata_id += PurePath(dataset_path).name # the traindata file name
    traindata_id += "_" + str(n_mfcc)         + "m" \
                 +  "_" + str(n_fft)          + "w" \
                 +  "_" + str(hop_length)     + "h" \
                 +  "_" + str(num_segments)   + "i" \
                 +  "_" + str(sample_rate)    + "r" \
                 +  "_" + str(track_duration) + "s"
    return traindata_id

def save_traindata(traindata, traindata_filename):
    Path(Aimx.Paths.GEN_TRAINDATA).mkdir(parents=True, exist_ok=True)
    GEN_TRAINDATA_FULLPATH = os.path.join(Aimx.Paths.GEN_TRAINDATA, traindata_filename)
    with open(GEN_TRAINDATA_FULLPATH, "w") as data_file:
        print_info("\n|||||| Writing file", quote(cyansky(GEN_TRAINDATA_FULLPATH)), "... ", end="")
        json.dump(traindata, data_file, indent=4)
        print_info("[DONE]")

# This function may be necessary for test pipeline automation, e.g. in scenarios when
# running multiple NNs, each requiring its own traindata. This function can in such
# cases be used to switch quickly by updating dataprep_result_meta.json contents correspondingly.
def update_dataprep_result_meta(traindata_filename, key, value):
    with open(Aimx.Dataprep.RESULT_METADATA_FULLPATH, "r") as fp:
        print_info("\n|||||| Loading file " + quote(cyansky(Aimx.Dataprep.RESULT_METADATA_FULLPATH)) + "... ", end="")
        prep_result_meta = json.load(fp)
        print_info("[DONE]")
    prep_result_meta[key] = value
    with open(Aimx.Dataprep.RESULT_METADATA_FULLPATH, 'w') as fp:
        print_info("\n|||||| Writing file", quote(cyansky(Aimx.Dataprep.RESULT_METADATA_FULLPATH)), "... ", end="")
        json.dump(prep_result_meta, fp, indent=4)
        print_info("[DONE]")