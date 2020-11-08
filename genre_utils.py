from pathlib import Path

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

def plot_history(history):
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

    # TODO: Finish this later
    #pt.savefig(get_dataset_code(dataset_json_filename) + ".png")

    pt.show()