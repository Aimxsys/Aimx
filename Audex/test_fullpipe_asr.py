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

parser.add_argument("-dataset_view",   nargs='*', default = DATASET_VIEW_DEFAULT, help = 'Specific directories (labels) to go through.')
parser.add_argument("-dataset_path",   type = Path,           help = 'Path to a dataset of sound files.')
parser.add_argument("-dataset_depth",  default = 5, type=int, help = 'Number of files to consider from each category.')
parser.add_argument("-epochs",         default = 5, type=int, help = 'Number of epochs to train.')
parser.add_argument("-inferdata_path", type = Path,           help = 'Path to the audio files on which model inference is to be tested.')
parser.add_argument("-example",        action ='store_true',  help = 'Will show a working example on how to call the script.')

parser.add_argument("-skip_dataprep",  action ='store_true',  help = 'Will skip the dataprep step.')
parser.add_argument("-skip_training",  action ='store_true',  help = 'Will skip the training step.')
parser.add_argument("-skip_inference", action ='store_true',  help = 'Will skip the inference step.')

args = parser.parse_args()

print_script_start_preamble(nameofthis(__file__), vars(args))

########################## Command Argument Handling & Verification #######################

if args.example:
    print_info(nameofthis(__file__) + " -dataset_path ../workdir/datasets/speech_commands_v001 -dataset_depth 55 -dataset_view down five"
                                      " -epochs 50 -inferdata_path ../workdir/infer/signal_down_five_few")
    exit()

###########################################################################################

# Looks like when launching on Linux from a corresponding Aimx venv, the shebang is enough for it
# to automatically pick up the right Python interpreter (from the venv you're launching from).
# However, when launching on Windows from a corresponding Aimx venv, we need to spoonfeed it with
# the right Python executable located in the /Scripts folder of the Windows venv.
interp   = [args.wenv + '/Scripts/python.exe'] if os.name == 'nt'    else []
dotslash =                                './' if os.name == 'posix' else ''

start_time = time.time()

# IMPORTANT: TODO: Note for later resolution from official Python documentation
# https://docs.python.org/2/library/subprocess.html#frequently-used-arguments:
# Warning Executing shell commands that incorporate unsanitized input from an untrusted
# source makes a program vulnerable to shell injection, a serious security flaw
# which can result in arbitrary command execution. For this reason, the use of shell=True
# is strongly discouraged in cases where the command string is constructed from external input

####################################################### ASR-related pipeline

if not args.skip_dataprep:
    # dataprep_asr.py -dataset_path ../workdir/datasets/speech_commands_v001 -dataset_depth 5
    subprocess.call(interp + [dotslash + 'dataprep_asr.py', '-dataset_path', str(args.dataset_path), '-dataset_depth', str(args.dataset_depth), '-dataset_view']
                    + args.dataset_view)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT dataprep_asr.py OK"))

if not args.skip_training:
    # train_asr.py -traindata_path most_recent_output -epochs 5
    subprocess.call(interp + [dotslash + 'train_asr.py', '-traindata_path', Aimx.MOST_RECENT_OUTPUT, '-epochs', str(args.epochs), '-savemodel'])
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_asr.py OK"))

if not args.skip_inference:
    # service_wordetect.py -model_path most_recent_output -inferdata_path ../workdir/infer/signal_down
    subprocess.call(interp + [dotslash + 'service_wordetect.py', '-model_path', get_actual_model_path(Aimx.MOST_RECENT_OUTPUT), '-inferdata_path', str(args.inferdata_path)])
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_asr.py OK"))
    
    print(magenta("ꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕ ASR FULL PIPELINE TEST OK"))
    
    print_info("Finished {} at {} with wall clock time (total): {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp_now()),
                                                                    lightyellow(timedelta(seconds = round(time.time() - start_time)))))