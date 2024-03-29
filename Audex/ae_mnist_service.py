#!/usr/bin/env python

from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as pt
import argparse
import sys
import os

from ae             import Autoencoder
from ae_mnist_train import downscale_traindata_pixels
from ae_mnist_train import reshape_traindata

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

    parser.add_argument("-repeat",      default =  1,                   type = int,  help = 'Repeat the run of the service specified number of times.')
    parser.add_argument("-num_infers",  default = 10,                   type = int,  help = 'Number of images to generate. If small, will also plot latent space points.')
    parser.add_argument("-randomize",  action ='store_true',                         help = 'Randomize picking from the dataset.')
    parser.add_argument("-showvencs",  action ='store_true',                         help = 'At the end, will show vencs in an interactive window.')
    parser.add_argument("-showgenums", action ='store_true',                         help = 'At the end, will show genums in an interactive window.')
    parser.add_argument("-mode_gen",   action ='store_true',                         help = 'This mode will generate a genum from latent space.')
    parser.add_argument("-mode_regen", action ='store_true',                         help = 'This mode will regenerate an image.')

    parser.add_argument("-example",    action ='store_true',                         help = 'Show a working example on how to call the script.')

    args = parser.parse_args()

    ########################## Command Argument Handling & Verification #######################

    if args.example:
        print_info(nameofthis(__file__) + "-model_path ../workdir/gen_models/model_aen_x10_e10_5000d_mnist -mode_regen-randomize")
        exit()

    if provided(args.inferdata_path) and not args.inferdata_path.exists():
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

    args.model_path = get_actual_model_path(args.model_path)
    
    ###########################################################################################
    
    print_script_start_preamble(nameofthis(__file__), vars(args))

    return args, parser

def pick_from(images, labels, num_samples=10, randomize=True):
    if randomize:
        indexes = np.random.choice(range(len(images)), num_samples)
    else:
        indexes = np.arange(num_samples)
    sample_images = images[indexes] # num_samples images
    sample_labels = labels[indexes] # num_samples labels
    return sample_images, sample_labels

def plot_vencs(vencs, labels, modelname, showinteractive):
    dim_latent = vencs.shape[1]
    pt.figure(figsize=(10, 10))

    # Print encodings if not too many
    if len(vencs) < 20:
        print_info("Digits and their corresponding {}-d vencs:".format(vencs.shape[1]))
        for i in range(len(vencs)):
            print(cyan(labels[i] if labels is not None else "None"), np.around(vencs[i], 2))

    # Scatterplot first two coordinates of the vencs in the latent space,
    # which will be the exact representation in case the latent space is two-dimensional.
    # Note that to map the venc to its digit, look at the corresponding color on the colormap.
    print_info("Scatter-plotting first two coordinates of the {}-d vencs...".format(vencs.shape[1]))
    pt.scatter(vencs[:, 0], vencs[:, 1], cmap="rainbow", c=labels, alpha=0.5, s=2)
    if labels is not None:
        pt.colorbar()

    # save the plot as most recent (often useful when comparing to a next NN run)
    Path(Aimx.Paths.GEN_PLOTS_VENCS).mkdir(parents=True, exist_ok=True)
    VENCS_FULLPATH = os.path.join(Aimx.Paths.GEN_PLOTS_VENCS, modelname + ".png")
    print_info("|||||| Saving file", quote_path(VENCS_FULLPATH), "... ", end="")
    pt.savefig(VENCS_FULLPATH)
    print_info("[DONE]")

    if showinteractive:
        pt.show()
    else:
        pt.close()

def plot_regenums(genums, origimages, modelname, showinteractive):
    fig = pt.figure(figsize=(15, 3))

    num_images = len(origimages)
    if num_images > 100: return # too many genums, takes long to plot and indistinguishable to human eye
    for i, (origimage, genum) in enumerate(zip(origimages, genums)):
        
        # Original image
        origimage = origimage.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(origimage, cmap="gray_r")

        # Genum
        genum = genum.squeeze() # (28, 28, 1) ===> (28, 28)
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(genum, cmap="gray_r")

    # save the plot as most recent (often useful when comparing to a next NN run)
    Path(Aimx.Paths.GEN_PLOTS_GENUM).mkdir(parents=True, exist_ok=True)
    GENUM_FULLPATH = os.path.join(Aimx.Paths.GEN_PLOTS_GENUM, modelname + ".png")
    print_info("|||||| Saving file", quote_path(GENUM_FULLPATH), "... ", end="")
    pt.savefig(GENUM_FULLPATH)
    print_info("[DONE]")

    if showinteractive:
        pt.show()
    else:
        pt.close()

def plot_genums(genums, modelname, showinteractive):
    fig = pt.figure(figsize=(15, 3))

    num_genums = len(genums)
    if num_genums > 100: return # too many genums, takes long to plot and indistinguishable to human eye
    for i, genum in enumerate(genums):
        
        genum = genum.squeeze()
        ax = fig.add_subplot(2, num_genums, i + num_genums + 1)
        ax.axis("off")
        ax.imshow(genum, cmap="gray_r")

    # save the plot as most recent (often useful when comparing to a next NN run)
    Path(Aimx.Paths.GEN_PLOTS_GENUM).mkdir(parents=True, exist_ok=True)
    GENUM_FULLPATH = os.path.join(Aimx.Paths.GEN_PLOTS_GENUM, modelname + ".png")
    print_info("|||||| Saving file ", quote_path(GENUM_FULLPATH), "... ", end="")
    pt.savefig(GENUM_FULLPATH)
    print_info("[DONE]")

    if showinteractive:
        pt.show()
    else:
        pt.close()

if __name__ == "__main__":
    args, parser = process_clargs()

    model = Autoencoder.load_model(args.model_path)

    # load, reshape, downscale
    (_, _), (x_test, y_test) = mnist.load_data() # traindata                      x_train.shape == (60000, 28, 28)
    _,  _,   x_test, y_test  = reshape_traindata(         _, _, x_test, y_test) # x_train.shape == (60000, 28, 28, 1)
    _,  _,   x_test, y_test  = downscale_traindata_pixels(_, _, x_test, y_test)

    # MNIST traindata values:
    # x_test.shape == (10000, 28, 28, 1)
    # y_test.shape == (10000,)

    while args.repeat > 0:
        args.repeat -= 1

        # Generate images from latent space vencs             <| 
        if args.mode_gen:
            labels = None # signify "unknown" (whatever the genum turns out to be)

            vencs, genums = model.gen_random(args.num_infers)

            for s, g in zip(sample_images, genums):
                print_info(purple("Euclidean distance between signum and genum"), np.linalg.norm(s - g))

            plot_vencs(vencs, labels, extract_filename(args.model_path), args.showvencs)
            plot_genums(genums,       extract_filename(args.model_path), args.showgenums)

        # Regenerate images from selected dataset samples   |><|
        elif args.mode_regen:
            sample_images, sample_labels = pick_from(x_test, y_test, args.num_infers, args.randomize)
 
            vencs, genums = model.regen(sample_images)

            for s, g in zip(sample_images, genums):
                print_info(purple("Euclidean distance between signum and genum"), np.linalg.norm(s - g))

            #plot_matrices_single_chart([sample_images[0].squeeze(), genums[0].squeeze()], "MNIST", extract_filename(args.model_path))
 
            plot_vencs(vencs,     sample_labels, extract_filename(args.model_path), args.showvencs)
            plot_regenums(genums, sample_images, extract_filename(args.model_path), args.showgenums)