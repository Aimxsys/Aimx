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

parser.add_argument("-plot_sound",     action ='store_true', help = 'Will test plotting the sound.')
parser.add_argument("-dataprep_genre", action ='store_true', help = 'Will test data preprocessing on genre.')
parser.add_argument("-dataprep_asr",   action ='store_true', help = 'Will test data preprocessing on ASR.')
parser.add_argument("-ann",            action ='store_true', help = 'Will test the vanilla ANN.')
parser.add_argument("-cnn",            action ='store_true', help = 'Will test the CNN.')
parser.add_argument("-rnn",            action ='store_true', help = 'Will test the RNN.')
parser.add_argument("-nns",            action ='store_true', help = 'Will test all NNs only.')
parser.add_argument("-all",            action ='store_true', help = 'Will test all scripts in the solution.')

args = parser.parse_args()

ARG_TEST_PLOT_SOUND      = args.all or args.plot_sound

ARG_TEST_DATAPREP_ASR    = args.all or args.dataprep_asr
ARG_TEST_TRAIN_ASR_CNN   = args.all or args.cnn

ARG_TEST_DATAPREP_GENRE  = args.all or args.dataprep_genre
ARG_TEST_TRAIN_GENRE_ANN = args.all or args.nns or args.ann
ARG_TEST_TRAIN_GENRE_CNN = args.all or args.nns or args.cnn
ARG_TEST_TRAIN_GENRE_RNN = args.all or args.nns or args.rnn

if ARG_TEST_PLOT_SOUND: # Sound plots
    # plot_sound.py -files_path ../workdir/sounds/two
    subprocess.call(['plot_sound.py', '-files_path', '../workdir/sounds/two'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT plot_sound.py OK"))

if ARG_TEST_DATAPREP_ASR: # Data preprocessing
    # dataprep_asr.py -dataset_path ../workdir/speech_commands_v0.01 -dataset_depth 5
    subprocess.call(['dataprep_asr.py', '-dataset_path', '../workdir/speech_commands_v0.01', '-dataset_depth', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT dataprep_asr.py OK"))

if ARG_TEST_TRAIN_ASR_CNN: # ASR using CNN
    # train_asr_cnn.py -data_path most_recent_output -epochs 5
    subprocess.call(['train_asr_cnn.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_asr_cnn.py OK"))

if ARG_TEST_DATAPREP_GENRE: # Data preprocessing
    # dataprep_genre.py -dataset_path ../workdir/dataset_c10_f100 -dataset_depth 5
    subprocess.call(['dataprep_genre.py', '-dataset_path', '../workdir/dataset_c10_f100', '-dataset_depth', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT dataprep_genre.py OK"))

if ARG_TEST_TRAIN_GENRE_ANN: # Genre classification using vanilla NN (no CNN or anything)
    # train_genre_ann.py -data_path most_recent_output -epochs 5
    subprocess.call(['train_genre_ann.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_genre_ann.py OK"))

if ARG_TEST_TRAIN_GENRE_CNN: # Genre classification using CNN
    # train_genre_cnn.py -data_path most_recent_output -epochs 5
    subprocess.call(['train_genre_cnn.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_genre_cnn.py OK"))

if ARG_TEST_TRAIN_GENRE_RNN: # Genre classification using RNN
    # train_genre_rnn.py -data_path most_recent_output -epochs 5
    subprocess.call(['train_genre_rnn.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)
    print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT train_genre_rnn.py OK"))

print(magenta("ꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕꓕ PIPELINE SANITY TEST OK"))