#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pt
import argparse
import sys
import os

from ae       import Autoencoder
from ae_train import load_mnist

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_audex import *

def process_clargs():
    # Calling with "-traindata_path /to/file" will expect to find the file in ./to directory.
    parser = argparse.ArgumentParser(description = '[TODO: Script description].')

    parser.add_argument("-model_path", default=Aimx.MOST_RECENT_OUTPUT, type = Path, help = 'Path to the model to be loaded.')
    parser.add_argument("-inferdata_path",                              type = Path, help = 'Path to the audio files on which model inference is to be tested.')
    parser.add_argument("-inferdata_range", default=[0, 50], nargs='*', type = int,  help = 'Range in -inferdata_path on which to do inference.')

    parser.add_argument("-num_genims", default = 10,                    type = int,  help = 'Number of images to generate.')
    parser.add_argument("-randomize",  action ='store_true',                         help = 'Randomize picking from the dataset.')
    parser.add_argument("-showgencs",  action ='store_true',                         help = 'At the end, will show gencs in an interactive window.')
    parser.add_argument("-showgenims", action ='store_true',                         help = 'At the end, will show genims in an interactive window.')
    parser.add_argument("-example",    action ='store_true',                         help = 'Show a working example on how to call the script.')

    args = parser.parse_args()

    ########################## Command Argument Handling & Verification #######################

    if args.example:
        print_info(nameofthis(__file__) + "[TODO: REPLACE THIS WITH AN ACTUAL EXAMPLE]")
        exit()

    if provided(args.inferdata_path) and not args.inferdata_path.exists():
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

    args.model_path = get_actual_model_path(args.model_path)
    
    ###########################################################################################
    
    print_script_start_preamble(nameofthis(__file__), vars(args))

    return args

def pick_images(images, labels, num_genims=10, pickrandom=True):
    if pickrandom:
        x = np.random.choice(range(len(images)), num_genims)
    else:
        x = np.arange(num_genims)
    sample_images = images[x]
    sample_labels = labels[x]
    return sample_images, sample_labels

def plot_gencs(gencs, labels, modelname, showinteractive):
    pt.figure(figsize=(10, 10))
    pt.scatter(gencs[:, 0], gencs[:, 1], cmap="rainbow", c=labels, alpha=0.5, s=2)
    pt.colorbar()

    # save the plot as most recent (often useful when comparing to a next NN run)
    Path(Aimx.Paths.GEN_PLOTS_GENCS).mkdir(parents=True, exist_ok=True)
    GENCS_FULLPATH = os.path.join(Aimx.Paths.GEN_PLOTS_GENCS, modelname + ".png")
    print_info("|||||| Saving file ", quote_path(GENCS_FULLPATH), "... ", end="")
    pt.savefig(GENCS_FULLPATH)
    print_info("[DONE]")

    if showinteractive:
        pt.show()
    else:
        pt.close()

def plot_genims(genims, origimages, modelname, showinteractive):
    fig = pt.figure(figsize=(15, 3))

    num_images = len(origimages)
    if num_images > 100: return # too many genims, takes long to plot and indistinguishable to human eye
    for i, (origimage, genim) in enumerate(zip(origimages, genims)):
        # Original image
        origimage = origimage.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(origimage, cmap="gray_r")
        # Genim
        genim = genim.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(genim, cmap="gray_r")

    # save the plot as most recent (often useful when comparing to a next NN run)
    Path(Aimx.Paths.GEN_PLOTS_GENIMS).mkdir(parents=True, exist_ok=True)
    GENIM_FULLPATH = os.path.join(Aimx.Paths.GEN_PLOTS_GENIMS, modelname + ".png")
    print_info("|||||| Saving file ", quote_path(GENIM_FULLPATH), "... ", end="")
    pt.savefig(GENIM_FULLPATH)
    print_info("[DONE]")

    if showinteractive:
        pt.show()

if __name__ == "__main__":
    args = process_clargs()

    ae = Autoencoder.load_model(args.model_path)

    x_train, y_train, x_test, y_test = load_mnist()

    sample_images, sample_labels = pick_images(x_test, y_test, args.num_genims, args.randomize)

    gencs, genims = ae.regen(sample_images)

    plot_gencs(gencs,   sample_labels, extract_filename(args.model_path), args.showgencs)
    plot_genims(genims, sample_images, extract_filename(args.model_path), args.showgenims)