#!/usr/bin/env python

# This script checks basic sanity of all scripts in the solution, respecting the pipeline whenever relevant
# (for example: preprocess the data and then train on it). Useful when doing global renamings, etc.

import os
import sys
import time
import argparse
import subprocess
from datetime import timedelta

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import *
from Audex.utils.utils_audex  import Aimx

parser = argparse.ArgumentParser(description = 'This utility script test scripts in the solution.')

parser.add_argument("-wenv", default = '../venv_aimx_win', type=str, help = 'Relative path to the corresponding Python venv'
                                                                            ' if launching on a Windows machine.'
                                                                            ' Ignore this argument if using the default venv path'
                                                                            ' ' + quote('../venv_aimx_win') + ' or launching on Linux.')

parser.add_argument("-plot_sound",     action ='store_true', help = 'Will test plotting the sound.')
parser.add_argument("-dataprep_genre", action ='store_true', help = 'Will test data preprocessing on genre.')
parser.add_argument("-dataprep_asr",   action ='store_true', help = 'Will test data preprocessing on ASR.')
parser.add_argument("-dataprep",       action ='store_true', help = 'Will test data preprocessing on all available workflows.')
parser.add_argument("-ann",            action ='store_true', help = 'Will test the vanilla ANN.')
parser.add_argument("-cnn",            action ='store_true', help = 'Will test the CNN.')
parser.add_argument("-rnn",            action ='store_true', help = 'Will test the RNN.')
parser.add_argument("-nns",            action ='store_true', help = 'Will test all NNs only.')
parser.add_argument("-asr",            action ='store_true', help = 'Will test the entire ASR flow from dataprep to training.')
parser.add_argument("-genre",          action ='store_true', help = 'Will test the entire Genre flow from dataprep to training.')
parser.add_argument("-all",            action ='store_true', help = 'Will test all scripts in the solution.')

parser.add_argument("-dataset_depth",  default = 3, type=int, help = 'Number of files to consider from each category.')
parser.add_argument("-epochs",         default = 3, type=int, help = 'Number of epochs to train.')

args = parser.parse_args()

print_script_start_preamble(nameofthis(__file__), vars(args))

ARG_TEST_PLOT_SOUND      = args.all or args.plot_sound
ARG_TEST_WORDETECT       = args.all

ARG_TEST_DATAPREP_ASR    = args.all or args.dataprep or args.dataprep_asr   or args.asr
ARG_TEST_DATAPREP_GENRE  = args.all or args.dataprep or args.dataprep_genre or args.genre

ARG_TEST_TRAIN_ASR       = args.all or args.nns or args.cnn or args.asr
ARG_TEST_TRAIN_GENRE_ANN = args.all or args.nns or args.ann or args.genre
ARG_TEST_TRAIN_GENRE_CNN = args.all or args.nns or args.cnn or args.genre
ARG_TEST_TRAIN_GENRE_RNN = args.all or args.nns or args.rnn or args.genre

# Looks like when launching on Linux from a corresponding Aimx venv, the shebang is enough for it
# to automatically pick up the right Python interpreter (from the venv you're launching from).
# However, when launching on Windows from a corresponding Aimx venv, we need to spoonfeed it with
# the right Python executable located in the /Scripts folder of the Windows venv.
interp   = [args.wenv + '/Scripts/python.exe'] if os.name == 'nt'    else []
dotslash =                                './' if os.name == 'posix' else ''

start_time = time.time()

# IMPORTANT: Note from the official Python documentation at:
# https://docs.python.org/2/library/subprocess.html#frequently-used-arguments:
# Warning Executing shell commands that incorporate unsanitized input from an untrusted
# source makes a program vulnerable to shell injection, a serious security flaw
# which can result in arbitrary command execution. For this reason, the use of shell=True
# is strongly discouraged in cases where the command string is constructed from external input

####################################################### Sound plotting

if ARG_TEST_PLOT_SOUND: # Sound plots
    # plot_sound.py -files_path ../workdir/sounds/two
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT plot_sound.py"))
    subprocess.call(interp + [dotslash + 'plot_sound.py', '-files_path', '../workdir/sounds/two'])

####################################################### Genre-related pipeline

if ARG_TEST_DATAPREP_GENRE: # Data preprocessing
    # dataprep_genre.py -dataset_path ../workdir/datasets/dataset_c10_f100 -dataset_depth 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT dataprep_genre.py"))
    subprocess.call(interp + [dotslash + 'dataprep_genre.py', '-dataset_path', '../workdir/datasets/dataset_c10_f100', '-dataset_depth', str(args.dataset_depth)])

if ARG_TEST_TRAIN_GENRE_ANN: # Genre classification using vanilla NN (no CNN or anything)
    # train_genre_ann.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_genre_ann.py"))
    subprocess.call(interp + [dotslash + 'train_genre_ann.py', '-epochs', str(args.epochs)])

if ARG_TEST_TRAIN_GENRE_CNN: # Genre classification using CNN
    # train_genre_cnn.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_genre_cnn.py"))
    subprocess.call(interp + [dotslash + 'train_genre_cnn.py', '-epochs', str(args.epochs)])

if ARG_TEST_TRAIN_GENRE_RNN: # Genre classification using RNN
    # train_genre_rnn.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_genre_rnn.py"))
    subprocess.call(interp + [dotslash + 'train_genre_rnn.py', '-epochs', str(args.epochs)])

####################################################### ASR-related pipeline

if ARG_TEST_DATAPREP_ASR: # Data preprocessing
    # dataprep_asr.py -dataset_path ../workdir/datasets/speech_commands_few -dataset_depth 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT dataprep_asr.py"))
    subprocess.call(interp + [dotslash + 'dataprep_asr.py', '-dataset_path', '../workdir/datasets/speech_commands_few', '-dataset_depth', str(args.dataset_depth)])

if ARG_TEST_TRAIN_ASR:
    # train_asr.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_asr.py"))
    subprocess.call(interp + [dotslash + 'train_asr.py', '-epochs', str(args.epochs), '-savemodel'])

####################################################### Wordetect inference test

if ARG_TEST_WORDETECT: # Wordetect service
    # service_wordetect.py -inferdata_path ../workdir/infer/signal_bird
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT service_wordetect.py"))
    subprocess.call(interp + [dotslash + 'service_wordetect.py', '-inferdata_path', '../workdir/infer/signal_bird'])

print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT PIPELINE SANITY"))
print(magenta("ꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕ TEST COMPLETE"))

print_info("Finished {} at {} with wall clock time (total): {} ".format(cyansky(nameofthis(__file__)),
                                                                lightyellow(timestamp_now()),
                                                                lightyellow(timedelta(seconds = round(time.time() - start_time)))))