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
from Audex.utils.utils_audex  import get_actual_model_path

# TODO: Replace the hard-coded line below with automatically extracted dirlabels
DATASET_VIEW_DEFAULT = ['alldirlabs']

parser = argparse.ArgumentParser(description = 'This utility script tests the entire pipeline of ASR scripts in the solution.')

parser.add_argument("-wenv", default = '../venv_aimx_win', type=str, help = 'Relative path to the corresponding Python venv'
                                                                            ' if launching on a Windows machine.'
                                                                            ' Ignore this argument if using the default venv path'
                                                                            ' ' + quote('../venv_aimx_win') + ' or launching on Linux.')

parser.add_argument("-ann_type",       default = "cnn", type=str, help = 'ANN type (CNN, RNN, etc).')
parser.add_argument("-dataset_view",   nargs='*', default = DATASET_VIEW_DEFAULT, help = 'Specific directories (labels) to go through.')
parser.add_argument("-dataset_path",                type = Path,  help = 'Path to a dataset of sound files.')
parser.add_argument("-dataset_depth",  default = 5, type = int,   help = 'Number of files to consider from each category.')
parser.add_argument("-epochs",         default = 5, type = int,   help = 'Number of epochs to train.')
parser.add_argument("-inferdata_path",              type = Path,  help = 'Path to the audio files on which model inference is to be tested.')
parser.add_argument("-example",        action ='store_true',      help = 'Show a working example on how to call the script.')
                                                                  
parser.add_argument("-skip_dataprep",  action ='store_true',      help = 'Skip the dataprep stage.')
parser.add_argument("-skip_training",  action ='store_true',      help = 'Skip the training stage.')
parser.add_argument("-skip_inference", action ='store_true',      help = 'Skip the inference stage.')

args = parser.parse_args()

########################## Command Argument Handling & Verification #######################

if args.example:
    print_info(nameofthis(__file__) + " -dataset_path ../workdir/datasets/speech_commands -dataset_depth 100 -dataset_view down seven backnoise"
                                      " -epochs 100 -inferdata_path ../workdir/infer/signal_down_backnoise_five_TRIMMED")
    exit()

###########################################################################################

print_script_start_preamble(nameofthis(__file__), vars(args))

# Looks like when launching on Linux from a corresponding Aimx venv, the shebang is enough for it
# to automatically pick up the right Python interpreter (from the venv you're launching from).
# However, when launching on Windows from a corresponding Aimx venv, we need to spoonfeed it with
# the right Python executable located in the /Scripts folder of the Windows venv.
interp   = [args.wenv + '/Scripts/python.exe'] if os.name == 'nt'    else []
dotslash =                                './' if os.name == 'posix' else ''

start_time = time.time()

####################################################### ASR-related pipeline

if not args.skip_dataprep:
    # asr_dataprep.py -dataset_path ../workdir/datasets/speech_commands_v001 -dataset_depth 5
    subprocess.run(interp + [dotslash + 'asr_dataprep.py', '-dataset_path', str(args.dataset_path), '-dataset_depth', str(args.dataset_depth), '-dataset_view']
                    + args.dataset_view, check=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT asr_dataprep.py OK"))

if not args.skip_training:
    # asr_train.py -epochs 5
    subprocess.run(interp + [dotslash + 'asr_train.py', '-ann_type', args.ann_type, '-traindata_path', Aimx.MOST_RECENT_OUTPUT, '-epochs', str(args.epochs), '-savemodel'], check=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT asr_train.py OK"))

if not args.skip_inference:
    # asr_service.py -inferdata_path ../workdir/infer/signal_down
    subprocess.run(interp + [dotslash + 'asr_service.py', '-model_path', get_actual_model_path(Aimx.MOST_RECENT_OUTPUT), '-inferdata_path', str(args.inferdata_path)], check=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT asr_train.py OK"))
    
    print(magenta("ꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕ ASR FULL PIPELINE TEST OK"))
    
    print_info("Finished {} at {} with wall clock time (total): {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp_now()),
                                                                    lightyellow(timedelta(seconds = round(time.time() - start_time)))))