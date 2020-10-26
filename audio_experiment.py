import argparse
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as pt
import IPython.display
import numpy as np
import scipy as sp
import os
from os import walk

# Module imports from this project
from audio_experiment_utils import *

AUDIO_FILES_DIR_DEFAULT = os.path.join(os.getcwd(), "sounds")

# TODO: Write a usage text and add these arguments there.
# Calling without -file_path        will expect to find the default ./sounds directory.
# Calling with   "-file_path mydir" will expect to find a           ./mydir  directory.
parser = argparse.ArgumentParser()

parser.add_argument("-file_path", type = Path)
parser.add_argument("-plot_all",         action='store_true')
parser.add_argument("-plot_signals",     action='store_true')
parser.add_argument("-plot_frequencies", action='store_true')
parser.add_argument("-plot_specs",       action='store_true')
parser.add_argument("-plot_melspecs",    action='store_true')
parser.add_argument("-plot_mfccs",       action='store_true')

args = parser.parse_args()

################# Command Argument Verification #################
if args.file_path is not None and not args.file_path.exists():
    raise FileNotFoundError("Provided path " + quote(str(args.file_path)) + " not found.")

if not Path(AUDIO_FILES_DIR_DEFAULT).exists():
    raise FileNotFoundError("Default path 'sounds' not found.")

PAR_AUDIO_FILES_DIR = args.file_path if args.file_path is not None else AUDIO_FILES_DIR_DEFAULT
PAR_PLOT_FREQUENCIES = args.plot_all or args.plot_frequencies
PAR_PLOT_SIGNALS     = args.plot_all or args.plot_signals
PAR_PLOT_SPECS       = args.plot_all or args.plot_specs
PAR_PLOT_MELSPECS    = args.plot_all or args.plot_melspecs
PAR_PLOT_MFCCS       = args.plot_all or args.plot_mfccs

print("=============================================================================")
print("Expecting audio files in PAR_AUDIO_FILES_DIR =", PAR_AUDIO_FILES_DIR)
print("=============================================================================")

(_, _, filenames) = next(os.walk(PAR_AUDIO_FILES_DIR)) # works

signal_packs = []

for filename in filenames:
    file = os.path.join(PAR_AUDIO_FILES_DIR, filename)
    print("Loading...", file)
    signal_packs.append((filename, librosa.load(file)))

for sigp in signal_packs:
    print_stats(sigp)

if PAR_PLOT_SIGNALS:
    plot_signals_single_chart(signal_packs)

if PAR_PLOT_FREQUENCIES:
    for sigp in signal_packs:
        plot_frequency_distribution(sigp)

if PAR_PLOT_SPECS:
    for sigp in signal_packs:
        plot_spectrogram(sigp, y_axis = "log")

if PAR_PLOT_MELSPECS:
    for sigp in signal_packs:
        mel_spectrogram = librosa.feature.melspectrogram(sigp[1][0], sigp[1][1], n_fft = 2048, hop_length = 512, n_mels = 90)
        print(mel_spectrogram.shape)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        pt.figure(figsize = (15, 10)).canvas.set_window_title("MEL Spectrogram")
        pt.title("MEL of " + str(sigp[0]))
        librosa.display.specshow(log_mel_spectrogram, x_axis = "time", y_axis = "mel", sr = sigp[1][1])
        pt.colorbar(format = "%+2.f")

if PAR_PLOT_MFCCS:
    for sigp in signal_packs:
        mfccs = librosa.feature.mfcc(sigp[1][0], n_mfcc=40, sr = sigp[1][1])
        print("mfcc.shape = ", mfccs.shape)
        pt.figure(figsize=(15,10)).canvas.set_window_title("MFCC")
        pt.title("MFCC of " + sigp[0])
        librosa.display.specshow(mfccs, x_axis="time", sr = sigp[1][1])
        pt.colorbar(format="%+2f")

pt.show()