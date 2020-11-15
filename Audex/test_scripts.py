# This script checks basic sanity of all scripts in the solution, respecting the pipeline whenever relevant
# (for example: preprocess the data and then train on it). Useful when doing global renamings, etc.
import argparse
import subprocess

from common_utils import *

# IMPORTANT: TODO: Note for later resolution from official Python documentation
# https://docs.python.org/2/library/subprocess.html#frequently-used-arguments:
# Warning Executing shell commands that incorporate unsanitized input from an untrusted
# source makes a program vulnerable to shell injection, a serious security flaw
# which can result in arbitrary command execution. For this reason, the use of shell=True
# is strongly discouraged in cases where the command string is constructed from external input

parser = argparse.ArgumentParser(description = 'This utility script test scripts in the solution.')

parser.add_argument("-plot_sound", action ='store_true', help = 'Will test plotting the sound.')
parser.add_argument("-dataprep",   action ='store_true', help = 'Will test data preprocessing.')
parser.add_argument("-ann",        action ='store_true', help = 'Will test the vanilla ANN.')
parser.add_argument("-cnn",        action ='store_true', help = 'Will test the CNN.')
parser.add_argument("-rnn",        action ='store_true', help = 'Will test the RNN.')
parser.add_argument("-nns",        action ='store_true', help = 'Will test all NNs only.')
parser.add_argument("-all",        action ='store_true', help = 'Will test all scripts in the solution.')

args = parser.parse_args()

ARG_TEST_PLOT_SOUND = args.all or args.plot_sound
ARG_TEST_DATAPREP   = args.all or args.dataprep
ARG_TEST_ANN        = args.all or args.nns or args.ann
ARG_TEST_CNN        = args.all or args.nns or args.cnn
ARG_TEST_RNN        = args.all or args.nns or args.rnn

if ARG_TEST_DATAPREP: # Sound plots
    # plot_sound.py -files_path sounds/two
    subprocess.call(['plot_sound.py', '-files_path', 'sounds/two'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT plot_sound.py OK"))

if ARG_TEST_DATAPREP: # Data preprocessing
    # genre_preprocess.py -dataset_path dataset_c10_f3
    subprocess.call(['genre_preprocess.py', '-dataset_path', 'dataset_c10_f3'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_preprocess.py OK"))

if ARG_TEST_ANN: # Genre classification using vanilla NN (no CNN or anything)
    # genre_classifier.py -data_path most_recent_output -epochs 5
    subprocess.call(['genre_classifier.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_classifier.py OK"))

if ARG_TEST_CNN: # Genre classification using CNN
    # genre_classifier_cnn.py -data_path most_recent_output -epochs 5
    subprocess.call(['genre_classifier_cnn.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_classifier_cnn.py OK"))

if ARG_TEST_RNN: # Genre classification using RNN
    # genre_classifier_rnn.py -data_path most_recent_output -epochs 5
    subprocess.call(['genre_classifier_rnn.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_classifier_rnn.py OK"))

print(magenta("ꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕ PIPELINE TEST OK"))