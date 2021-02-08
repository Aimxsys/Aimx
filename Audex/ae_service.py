#!/usr/bin/env python

import sounddevice as sd
import librosa
import numpy as np
import matplotlib.pyplot as pt
import argparse
import sys
import os

from itertools import islice

from ae             import Autoencoder
from ae_mnist_train import normalize_traindata_pixels
from ae_mnist_train import reshape_traindata

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_audex import *
from asr_service import CreateAsrService

def process_clargs():
    # Calling with "-traindata_path /to/file" will expect to find the file in ./to directory.
    parser = argparse.ArgumentParser(description = '[TODO: Script description].')

    parser.add_argument("-model_path", default=Aimx.MOST_RECENT_OUTPUT, type=Path, help = 'Path to the model to be loaded.')
    parser.add_argument("-inferdata_path",                              type=Path, help = 'Path to the audio files on which model inference is to be tested.')
    parser.add_argument("-inferdata_range", default=[0, 50], nargs='*', type=int,  help = 'Range in -inferdata_path on which to do inference.')

    parser.add_argument("-signum_type",   default = "mel",              type=str,  help = 'Signal numerization type.')
    parser.add_argument("-n_mfcc",        default = 16,                 type=int,  help = 'Number of MFCC to extract.')
    parser.add_argument("-n_fft",         default = 2048,               type=int,  help = 'Length of the FFT window.   Measured in # of samples.')
    parser.add_argument("-hop_length",    default = 512,                type=int,  help = 'Sliding window for the FFT. Measured in # of samples.')
    parser.add_argument("-sample_rate",   default = 22050,              type=int,  help = 'Sample rate at which to read the audio files.')
    parser.add_argument("-load_duration", default = 1,                  type=int,  help = 'Only load up to this much audio (in seconds).')

    parser.add_argument("-repeat",        default =  1,                 type=int,  help = 'Repeat the run of the service specified number of times.')
    parser.add_argument("-num_infers",    default = 10,                 type=int,  help = 'Number of images to generate. If small, will also plot latent space points.')
    parser.add_argument("-randomize",     action ='store_true',                    help = 'Randomize picking from the dataset.')
    parser.add_argument("-showvencs",     action ='store_true',                    help = 'At the end, will show vencs in an interactive window.')
    parser.add_argument("-showgenums",    action ='store_true',                    help = 'At the end, will show genums in an interactive window.')
    parser.add_argument("-mode_gen",      action ='store_true',                    help = 'This mode will generate a genum from latent space.')
    parser.add_argument("-mode_regen",    action ='store_true',                    help = 'This mode will regenerate an image.')

    parser.add_argument("-example",       action ='store_true',                    help = 'Show a working example on how to call the script.')

    args = parser.parse_args()

    ########################## Command Argument Handling & Verification #######################

    if args.example:
        print_info(nameofthis(__file__) + " -inferdata_path ../workdir/infer/signal_down_backnoise_five_TRIMMED -inferdata_range 35 36")
        print_info("The command above should say", "seven")
        exit()

    if provided(args.inferdata_path) and not args.inferdata_path.exists():
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

    args.model_path = get_actual_model_path(args.model_path)

    args.signum_type   = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["type"]          if not provided(args.signum_type)   else args.signum_type
    args.n_mfcc        = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["n_mfcc"]        if not provided(args.n_mfcc)        else args.n_mfcc
    args.n_fft         = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["n_fft"]         if not provided(args.n_fft)         else args.n_fft
    args.n_hop_length  = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["hop_length"]    if not provided(args.hop_length)    else args.hop_length
    args.sample_rate   = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["sample_rate"]   if not provided(args.hop_length)    else args.sample_rate
    args.load_duration = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["load_duration"] if not provided(args.load_duration) else args.load_duration
    
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

    asr = CreateAsrService(args.model_path)

    print_info("\nPredicting with dataset view (labels):", asr.label_mapping)
    print_info("On files in:", args.inferdata_path)

    (_, _, afnames) = next(os.walk(args.inferdata_path))
    
    START = args.inferdata_range[0]; # of the range in -inferdata_path on which to do inference
    END   = args.inferdata_range[1]; # of the range in -inferdata_path on which to do inference

    # Process audio files starting from START until END
    for i, afname in enumerate(islice(afnames, START, END)):
        af_fullpath = os.path.join(args.inferdata_path, afname)
        asr.load_audiofile(af_fullpath, args.load_duration)

        if len(asr.af_signal) < args.sample_rate: # process only signals of at least 1 sec
            print_info("skipped a short (< 1s) signal")
            continue 
        
        # Play original sound
        play(asr.af_signal, asr.af_sr,
             "Playing original audio signal {} of shape {} and numerical content:".format(quote(cyan(afname)), cyan(asr.af_signal.shape)),
             "Continue on to signumerize?")
                
        # Numerize original sound for inference. signums.shape will be:
        # (1, 44,  16, 1) if MFCC
        # (1, 44, 128, 1) if Mel
        signums = asr.signumerize(signum_type=args.signum_type, n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop_length)

        #showspec_mel(signums.squeeze().T, afname) # TODO: This line causes mel_to_audio() below throw numpy.linalg.LinAlgError

        # Restore and play back immediately to compare with the original playback
        #                                            squeeze()  transpose()   to_audio()
        # by transforming mel-signumerization: (1, 44, 128, 1) => (44, 128) => (128, 44) => (22016,)
        print_info("Immediately restoring for playback signums.shape:", pinkred(signums.shape))
        #signal_restored = librosa.feature.inverse.mfcc_to_audio(signums.squeeze().T)
        signal_restored = librosa.feature.inverse.mel_to_audio(signums.squeeze().T)
        signal_restored = np.pad(signal_restored, pad_width=(0, len(asr.af_signal) - len(signal_restored)))
        print_info("\nEuclidean distance between original and immediately restored (zero-padded) signals:",
                   np.linalg.norm(asr.af_signal - signal_restored), "\n") # for some reason, not identical from run to run
        play(signal_restored, signal_restored.shape[0], # signal_restored.shape == (22016,)
             "Playing immediately restored audio signal of shape {}  and numerical content:".format(cyan(signal_restored.shape)),
             "Continue on to sending the above signums to NN?\n")

        # Normalize
        #signums = librosa.util.normalize(signums)
        #print_info("Numerization for signums[0][0] normalized:")
        #deprint(np.around(signums[0][0], 2).T)
        #decolprint(signums.shape, "signums.shape")

        print_info("/\/\\" * 20, " SENDING signums of shape {} INTO NN".format(signums.shape))

        vencs, genums = asr.model.regen(signums)

        decolprint( vencs.shape,  "vencs.shape")
        decolprint(genums.shape, "genums.shape")
        print_info("Sound files and their corresponding {}-d vencs:".format(vencs.shape[1]))
        print(cyan(afname), np.around(vencs[0], 2))

        #showspec_mel(genums.squeeze().T, quote(afname) + " genum") # TODO: This line causes mel_to_audio() below throw numpy.linalg.LinAlgError

        #genum_restored = librosa.feature.inverse.mfcc_to_audio(genums.squeeze().T)
        genum_restored = librosa.feature.inverse.mel_to_audio(genums.squeeze().T)
        genum_restored = np.pad(genum_restored, pad_width=(0, len(asr.af_signal) - len(genum_restored)))

        print_info("\nEuclidean distance between original and restored genum (zero-padded) signals:",
                   np.linalg.norm(asr.af_signal - genum_restored), "\n") # for some reason, not identical from run to run

        input(yellow("Continue on to play genums?"))
        play(genum_restored, genum_restored.shape[0], "Playing genum of shape " + cyan(genum_restored.shape), waitforanykey=False)