import argparse
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as pt
import IPython.display
import numpy as np
import scipy as sp
import os
from   os import walk

# Module imports from this project
from common_utils import *
from  audex_utils import *

AUDIO_FILES_DIR_DEFAULT_NAME = "sounds"
AUDIO_FILES_DIR_DEFAULT = os.path.join(os.getcwd(), AUDIO_FILES_DIR_DEFAULT_NAME)

# Calling without -files_path               will expect to find the default ./sounds directory.
# Calling with   "-files_path mydir"        will expect to find a           ./mydir  directory.
# Calling with   "-files_path /to/file.wav" will expect to find the wav file in ./to directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' audio files and their various spectrograms.')

parser.add_argument("-files_path",       type = Path,          help = 'Path to a sound files directory or a single file.')
parser.add_argument("-plot_all",         action ='store_true', help = 'Will plot all available charts and spectrograms.')
parser.add_argument("-plot_signals",     action ='store_true', help = 'Will plot time-domain signals of the sound files.')
parser.add_argument("-plot_frequencies", action ='store_true', help = 'Will plot frequency domains of the sound files.')
parser.add_argument("-plot_specs",       action ='store_true', help = 'Will plot spectrograms of the sound files.')
parser.add_argument("-plot_melspecs",    action ='store_true', help = 'Will plot Mel spectrograms of the sound files.')
parser.add_argument("-plot_mfccs",       action ='store_true', help = 'Will plot MFCCs of the sound files.')

args = parser.parse_args()

############################## Command Argument Verification ##############################

if provided(args.files_path) and not args.files_path.exists():
    raise FileNotFoundError("Directory " + pinkred(os.getcwd()) + " does not contain requested path " + quote(pinkred(str(args.files_path))))

if not provided(args.files_path) and not Path(AUDIO_FILES_DIR_DEFAULT).exists():
    raise FileNotFoundError("Directory " + pinkred(os.getcwd()) + " does not contain default audio directory " + quote(pinkred(AUDIO_FILES_DIR_DEFAULT_NAME)))

###########################################################################################

ARG_AUDIO_FILES_DIR  = args.files_path if provided(args.files_path) else AUDIO_FILES_DIR_DEFAULT
ARG_PLOT_FREQUENCIES = args.plot_all or args.plot_frequencies
ARG_PLOT_SIGNALS     = args.plot_all or args.plot_signals
ARG_PLOT_SPECS       = args.plot_all or args.plot_specs
ARG_PLOT_MELSPECS    = args.plot_all or args.plot_melspecs
ARG_PLOT_MFCCS       = args.plot_all or args.plot_mfccs

print_info("=============================================================================")
print_info("Expecting audio files in ARG_AUDIO_FILES_DIR =", ARG_AUDIO_FILES_DIR)
print_info("=============================================================================")

path = Path(ARG_AUDIO_FILES_DIR)
signal_packs = []

if path.is_file():
    print_info("|||||| Loading...", cyansky(path))
    signal_packs.append((Path(path).name, librosa.load(path)))
else: # directory
    (_, _, filenames) = next(os.walk(ARG_AUDIO_FILES_DIR)) # works
    for filename in filenames:
        file = os.path.join(ARG_AUDIO_FILES_DIR, filename)
        print_info("|||||| Loading...", cyansky(file))
        signal_packs.append((filename, librosa.load(file)))

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