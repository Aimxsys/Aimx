#!/usr/bin/env python

import argparse
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as pt
import sys
import os

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Module imports from this project
from Audex.utils.utils_common     import *
from Audex.utils.utils_plot_sound import *

AUDIO_FILES_DIR_DEFAULT_NAME = "sounds"
AUDIO_FILES_DIR_DEFAULT = os.path.join(os.getcwd(), AUDIO_FILES_DIR_DEFAULT_NAME)

# Calling without -files_path               will expect to find the default ./sounds directory.
# Calling with   "-files_path mydir"        will expect to find a           ./mydir  directory.
# Calling with   "-files_path /to/file.wav" will expect to find the wav file in ./to directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' audio files and their various spectrograms.')

parser.add_argument("-files_path",       type = Path,  default = AUDIO_FILES_DIR_DEFAULT, help = 'Path to a sound files directory or a single file.')
parser.add_argument("-plot_all",         action ='store_true', help = 'Plot all available charts and spectrograms.')
parser.add_argument("-plot_signals",     action ='store_true', help = 'Plot time-domain signals of the sound files.')
parser.add_argument("-plot_frequencies", action ='store_true', help = 'Plot frequency domains of the sound files.')
parser.add_argument("-plot_specs",       action ='store_true', help = 'Plot spectrograms of the sound files.')
parser.add_argument("-plot_melspecs",    action ='store_true', help = 'Plot Mel spectrograms of the sound files.')
parser.add_argument("-plot_mfccs",       action ='store_true', help = 'Plot MFCCs of the sound files.')
parser.add_argument("-example",          action ='store_true', help = 'Show a working example on how to call the script.')

args = parser.parse_args()

########################## Command Argument Handling & Verification #######################

if args.example:
    print_info(nameofthis(__file__) + " -files_path ../workdir/sounds/two")
    exit()

if provided(args.files_path) and not args.files_path.exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.files_path)))

if not provided(args.files_path) and not Path(AUDIO_FILES_DIR_DEFAULT).exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain default audio directory " + quote(pinkred(AUDIO_FILES_DIR_DEFAULT_NAME)))

###########################################################################################

print_script_start_preamble(nameofthis(__file__), vars(args))

ARG_PLOT_FREQUENCIES = args.plot_all or args.plot_frequencies
ARG_PLOT_SIGNALS     = args.plot_all or args.plot_signals
ARG_PLOT_SPECS       = args.plot_all or args.plot_specs
ARG_PLOT_MELSPECS    = args.plot_all or args.plot_melspecs
ARG_PLOT_MFCCS       = args.plot_all or args.plot_mfccs

afiles_path = Path(args.files_path)
signal_packs = []

if afiles_path.is_file():
    print_info("|||||| Loading file " + quote_path(afiles_path) + "...", end="")
    signal_packs.append((Path(afiles_path).name, librosa.load(afiles_path)))
    print_info("[DONE]")
else: # directory
    (_, _, afnames) = next(os.walk(args.files_path)) # works
    for afname in afnames:
        af = os.path.join(args.files_path, afname)
        print_info("|||||| Loading file " + quote_path(af) + "...", end="")
        signal_packs.append((afname, librosa.load(af)))
        print_info("[DONE] & appended to signal pack")

for sigp in signal_packs:
    print_stats(sigp)

if ARG_PLOT_SIGNALS:
    plot_signals_single_chart(signal_packs)

if ARG_PLOT_FREQUENCIES:
    for sigp in signal_packs:
        plot_frequency_distribution(sigp)

if ARG_PLOT_SPECS:
    for sigp in signal_packs:
        plot_spectrogram(sigp, y_axis = "log")

if ARG_PLOT_MELSPECS:
    for sigp in signal_packs:
        plot_melspec(sigp)

if ARG_PLOT_MFCCS:
    for sigp in signal_packs:
        plot_mfcc(sigp)

pt.show()