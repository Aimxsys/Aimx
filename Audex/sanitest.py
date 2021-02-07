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

parser.add_argument("-plot_sound",     action ='store_true', help = 'Test plotting the sound.')
parser.add_argument("-dataprep_genre", action ='store_true', help = 'Test data preprocessing on genre.')
parser.add_argument("-dataprep_asr",   action ='store_true', help = 'Test data preprocessing on ASR.')
parser.add_argument("-dataprep",       action ='store_true', help = 'Test data preprocessing on all available workflows.')
parser.add_argument("-ann",            action ='store_true', help = 'Test the vanilla ANN.')
parser.add_argument("-cnn",            action ='store_true', help = 'Test the CNN.')
parser.add_argument("-rnn",            action ='store_true', help = 'Test the RNN.')
parser.add_argument("-nns",            action ='store_true', help = 'Test all NNs only.')
parser.add_argument("-asr",            action ='store_true', help = 'Test the entire ASR flow from dataprep to training.')
parser.add_argument("-genre",          action ='store_true', help = 'Test the entire Genre flow from dataprep to training.')
parser.add_argument("-all",            action ='store_true', help = 'Test all scripts in the solution.')

parser.add_argument("-dataset_depth",  default = 3, type=int, help = 'Number of files to consider from each category.')
parser.add_argument("-epochs",         default = 3, type=int, help = 'Number of epochs to train.')

args = parser.parse_args()

print_script_start_preamble(nameofthis(__file__), vars(args))

ARG_TEST_PLOT_SOUND      = args.all or args.plot_sound
ARG_TEST_ASR             = args.all or args.asr

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

os.replace(Aimx.Dataprep.RESULT_METADATA_FULLPATH, Aimx.Dataprep.RESULT_METADATA_FULLPATH + ".stbkp")
os.replace(Aimx.Training.RESULT_METADATA_FULLPATH, Aimx.Training.RESULT_METADATA_FULLPATH + ".stbkp")

start_time = time.time()

####################################################### Sound plotting

if ARG_TEST_PLOT_SOUND: # Sound plots
    # plot_sound.py -files_path ../workdir/sounds/two
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT plot_sound.py"))
    subprocess.run(interp + [dotslash + 'plot_sound.py', '-files_path', '../workdir/sounds/two'], check=True)

####################################################### Genre-related pipeline

if ARG_TEST_DATAPREP_GENRE: # Data preprocessing
    # genre_dataprep.py -dataset_path ../workdir/datasets/genre_c10_f100 -dataset_depth 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_dataprep.py"))
    subprocess.run(interp + [dotslash + 'genre_dataprep.py', '-dataset_path', '../workdir/datasets/genre_c10_f100', '-dataset_depth', str(args.dataset_depth)], check=True)

if ARG_TEST_TRAIN_GENRE_ANN: # Genre classification using vanilla NN (no CNN or anything)
    # genre_train_ann.py.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_train_ann.py.py"))
    subprocess.run(interp + [dotslash + 'genre_train_ann.py', '-epochs', str(args.epochs)], check=True)

if ARG_TEST_TRAIN_GENRE_CNN: # Genre classification using CNN
    # genre_train_cnn.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_train_cnn.py"))
    subprocess.run(interp + [dotslash + 'genre_train_cnn.py', '-epochs', str(args.epochs)], check=True)

if ARG_TEST_TRAIN_GENRE_RNN: # Genre classification using RNN
    # genre_train_rnn.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_train_rnn.py"))
    subprocess.run(interp + [dotslash + 'genre_train_rnn.py', '-epochs', str(args.epochs)], check=True)

####################################################### ASR-related pipeline

if ARG_TEST_DATAPREP_ASR: # Data preprocessing
    # asr_dataprep.py -dataset_path ../workdir/datasets/speech_commands_few -dataset_depth 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT asr_dataprep.py"))
    subprocess.run(interp + [dotslash + 'asr_dataprep.py', '-dataset_path', '../workdir/datasets/speech_commands_few', '-dataset_depth', str(args.dataset_depth)], check=True)

if ARG_TEST_TRAIN_ASR:
    # asr_train.py -epochs 5
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT asr_train.py"))
    subprocess.run(interp + [dotslash + 'asr_train.py', '-epochs', str(args.epochs), '-savemodel'], check=True)

####################################################### ASR inference test

if ARG_TEST_ASR:
    # asr_service.py -inferdata_path ../workdir/infer/signal_bird
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT asr_service.py"))
    subprocess.run(interp + [dotslash + 'asr_service.py', '-inferdata_path', '../workdir/infer/signal_bird'], check=True)

print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT PIPELINE SANITY"))
print(magenta("ꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕ TEST COMPLETE"))

print_info("Finished {} at {} with wall clock time (total): {} ".format(cyansky(nameofthis(__file__)),
                                                                lightyellow(timestamp_now()),
                                                                lightyellow(timedelta(seconds = round(time.time() - start_time)))))

os.replace(Aimx.Dataprep.RESULT_METADATA_FULLPATH + ".stbkp", Aimx.Dataprep.RESULT_METADATA_FULLPATH)
os.replace(Aimx.Training.RESULT_METADATA_FULLPATH + ".stbkp", Aimx.Training.RESULT_METADATA_FULLPATH)