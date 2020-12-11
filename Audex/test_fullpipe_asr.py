#!/usr/bin/env python

# This script checks basic sanity of all scripts in the solution, respecting the pipeline whenever relevant
# (for example: preprocess the data and then train on it). Useful when doing global renamings, etc.
from datetime import timedelta
import time
import argparse
import subprocess

from utils_common import *
from utils_audex  import Aimx
from utils_audex  import get_actual_model_path

# TODO: Replace the hard-coded line below with automatically extracted dirlabels
DATASET_VIEW_DEFAULT = ['alldirlabs']

parser = argparse.ArgumentParser(description = 'This utility script tests the entire pipeline of ASR scripts in the solution.')

parser.add_argument("-dataset_view",   nargs='*', default = DATASET_VIEW_DEFAULT, help = 'Specific directories (labels) to go through.')
parser.add_argument("-dataset_path",   type = Path,           help = 'Path to a dataset of sound files.')
parser.add_argument("-dataset_depth",  default = 5, type=int, help = 'Number of files to consider from each category.')
parser.add_argument("-epochs",         default = 5, type=int, help = 'Number of epochs to train.')
parser.add_argument("-inferdata_path", type = Path,           help = 'Path to the audio files on which model inference is to be tested.')

args = parser.parse_args()

print_script_start_preamble(nameofthis(__file__), vars(args))

########################## Command Argument Handling & Verification #######################

# ...

###########################################################################################

start_time = time.time()

# IMPORTANT: TODO: Note for later resolution from official Python documentation
# https://docs.python.org/2/library/subprocess.html#frequently-used-arguments:
# Warning Executing shell commands that incorporate unsanitized input from an untrusted
# source makes a program vulnerable to shell injection, a serious security flaw
# which can result in arbitrary command execution. For this reason, the use of shell=True
# is strongly discouraged in cases where the command string is constructed from external input

####################################################### ASR-related pipeline

# dataprep_asr.py -dataset_path ../workdir/speech_commands_v001 -dataset_depth 5
subprocess.call(['dataprep_asr.py', '-dataset_path', str(args.dataset_path), '-dataset_depth', str(args.dataset_depth), '-dataset_view']
                + args.dataset_view, shell=True)
print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT dataprep_asr.py OK"))

# train_asr_cnn.py -traindata_path most_recent_output -epochs 5
subprocess.call(['train_asr_cnn.py', '-traindata_path', Aimx.MOST_RECENT_OUTPUT, '-epochs', str(args.epochs), '-savemodel'], shell=True)
print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_asr_cnn.py OK"))

# service_wordetect.py -model_path most_recent_output -inferdata_path ../workdir/infer_down
subprocess.call(['service_wordetect.py', '-model_path', get_actual_model_path(Aimx.MOST_RECENT_OUTPUT), '-inferdata_path', str(args.inferdata_path)], shell=True)
print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_asr_cnn.py OK"))

print(magenta("ꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕ ASR FULL PIPELINE TEST OK"))

print_info("Finished {} at {} with wall clock time (total): {} ".format(cyansky(nameofthis(__file__)),
                                                                lightyellow(timestamp_now()),
                                                                lightyellow(timedelta(seconds = round(time.time() - start_time)))))