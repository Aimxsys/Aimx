from pathlib import Path

import numpy as np
import json
import os

import matplotlib.pyplot as pt

from common_utils import *

PREPROCESS_RESULT_META_FILENAME = "preprocess_result_meta.json"

def to_genre_name(label_id):
    return [
        'blues'     ,
        'classical' ,
        'country'   ,
        'disco'     ,
        'hiphop'    ,
        'jazz'      ,
        'metal'     ,
        'pop'       ,
        'reggae'    ,
        'rock'      
    ][label_id]

def get_preprocess_result_meta():
    if not hasattr(get_preprocess_result_meta, "cached"):        
        with open(PREPROCESS_RESULT_META_FILENAME, "r") as file:
            print_info("\n|||||| Loading file " + quote(cyansky(PREPROCESS_RESULT_META_FILENAME)) + "...", end="")
            preprocess_result_meta = json.load(file)
            print_info(" [DONE]")
        get_preprocess_result_meta.cached = preprocess_result_meta
    return get_preprocess_result_meta.cached

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

def plot_history(history, trainid, show_interactive):
    """ Plots accuracy/loss for training/validation set as a function of epochs
        :param history: Training history of model
    """
    fig, axs = pt.subplots(2, figsize=(8, 6))
    dataset_json_filename = get_preprocess_result_meta()["most_recent_output"]
    fig.canvas.set_window_title("Accuracy & Error - " + get_dataset_code(dataset_json_filename))
    fig.suptitle(get_dataset_code(dataset_json_filename), fontsize=14)

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
    Path("gen_plots").mkdir(parents=True, exist_ok=True)
    mr_plot_fullpath = os.path.join("gen_plots", "most_recent_plot_" + trainid + get_dataset_code(dataset_json_filename) + ".png")
    print_info("\n|||||| Saving image file", quote(cyansky(mr_plot_fullpath)), "... ", end="")
    pt.savefig(mr_plot_fullpath)
    print_info("[DONE]")

    if show_interactive:
        pt.show()

def save_current_model(model, model_id):
    mr_model_fullpath = os.path.join("gen_saved_models", "most_recent_model_" + model_id)
    print_info("\n|||||| Saving model ", quote(cyansky(mr_model_fullpath)), "... ", end="")
    model.save(mr_model_fullpath)
    print_info("[DONE]")