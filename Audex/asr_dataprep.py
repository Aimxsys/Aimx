#!/usr/bin/env python

from itertools import islice
from pathlib import PurePath
from pathlib import Path
from datetime import timedelta
import time
import argparse
import librosa
import json
import math
import sys
import os

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import *
from Audex.utils.utils_audex  import *

# Download from https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

DATASET_DIR_DEFAULT_NAME = "dataset"
DATASET_DIR_DEFAULT      = os.path.join(os.getcwd(), DATASET_DIR_DEFAULT_NAME)

# TODO: Replace the hard-coded line below with automatically extracted dirlabels
DATASET_VIEW_DEFAULT = ['alldirlabs']

# Calling without -dataset_path               will expect to find the default ./dataset directory.
# Calling with   "-dataset_path mydir"        will expect to find a           ./mydir   directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' preprocessing audio files to extract the training'
                                               ' data later to be fed into a neural network.')

parser.add_argument("-dataset_view",   nargs='*',   default = DATASET_VIEW_DEFAULT, help = 'Specific directories (labels) to go through.')
parser.add_argument("-dataset_path",   type = Path, default = DATASET_DIR_DEFAULT,  help = 'Path to a dataset of sound files.')
parser.add_argument("-dataset_depth",  default =     5, type=int, help = 'Number of files to consider from each category.')
parser.add_argument("-n_mfcc",         default =    13, type=int, help = 'Number of MFCC to extract.')
parser.add_argument("-n_fft",          default =  2048, type=int, help = 'Length of the FFT window.   Measured in # of samples.')
parser.add_argument("-hop_length",     default =   512, type=int, help = 'Sliding window for the FFT. Measured in # of samples.')
parser.add_argument("-sample_rate",    default = 22050, type=int, help = 'Sample rate at which to read the audio files.')
parser.add_argument("-load_duration",  default =     1, type=int, help = 'Only load up to this much audio (in seconds).')
parser.add_argument("-cutname",        action ='store_true',      help = 'Generate a json name with no details (cut).')
parser.add_argument("-verbose",        action ='store_true',      help = 'Print more detailed output messages.')
parser.add_argument("-example",        action ='store_true',      help = 'Show a working example on how to call the script.')

args = parser.parse_args()

########################## Command Argument Handling & Verification #######################

if args.example:
    print_info(nameofthis(__file__) + " -dataset_path ../workdir/datasets/speech_commands -dataset_view backnoise down seven -dataset_depth 5")
    exit()

if provided(args.dataset_path) and not args.dataset_path.exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.dataset_path)))

if not provided(args.dataset_path) and not Path(DATASET_DIR_DEFAULT).exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain default dataset directory " + quote(pinkred(DATASET_DIR_DEFAULT_NAME)))

if Aimx.Dataprep.ALL_DIR_LABELS in args.dataset_view: # special value ok for now, may need to be rewritten in a better way
    args.dataset_view = get_all_dirnames_in(args.dataset_path)

###########################################################################################

print_script_start_preamble(nameofthis(__file__), vars(args))

def dataprep(dataset_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, sample_rate = 22050, load_duration = 30):
    """
    Extracts MFCC from music dataset and saves them into a json file along witgh genre labels.
        :param  dataset_path (str): Path to dataset.
        :param        n_mfcc (int): Number of MFCC to extract.
        :param         n_fft (int): Length of the FFT window.   Measured in # of samples.
        :param    hop_length (int): Sliding window for the FFT. Measured in # of samples.
    """
    traindata_id = compose_traindata_id(args.dataset_depth, args.dataset_view, dataset_path,
                                       n_mfcc, n_fft, hop_length, 1, sample_rate, load_duration)

    # dictionary to store mapping, labels, and MFCC
    traindata = {
       Aimx.TIMESTAMP          : [],
       Aimx.DURATION           : [],
       Aimx.TrainData.MAPPING  : [],
       Aimx.TrainData.LABELS   : [],
       Aimx.TrainData.FILES    : [],
       Aimx.TrainData.MFCC     : []
    }

    print_info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv dataprep()")
    print_info("traindata_id   =", traindata_id)
    print_info("n_mfcc         =", n_mfcc)
    print_info("n_fft          =", n_fft)
    print_info("hop_length     =", hop_length)
    print_info("sample_rate    =", sample_rate)
    print_info("load_duration  =", load_duration)

    label_id = 0

    # loop through all subfolders
    for dirpath, _, afnames in os.walk(dataset_path):
        
        # ensure we're processing at subfolder level
        if extract_filename(dirpath) is extract_filename(dataset_path):
            continue

        # process only those dir-labels that are in the requested view
        if extract_filename(dirpath) not in args.dataset_view:
            continue

        # save genre label (i.e. subfolder name) in the mapping
        label_name = extract_filename(dirpath)
        traindata[Aimx.TrainData.MAPPING].append(label_name)
        print_info("\nProcessing label {} {}".format(cyan(label_id), label_name))

        # process all audio files in subfolders
        for pbi, afname in enumerate(islice(afnames, args.dataset_depth)):

            if not afname.endswith(".wav"):
                continue

            global total_audios_length_sec
            
            progress_bar(pbi, min(len(afnames), args.dataset_depth))

		    # load audio file
            af_path                  = os.path.join(dirpath, afname)
            signal, sample_rate      = librosa.load(af_path, sr = sample_rate, duration = load_duration)
            total_audios_length_sec += librosa.get_duration(signal, sample_rate)

            print_info("\nTotal samples in signal (audio track) {} = {}".format(extract_filename(af_path), len(signal)), verbose = args.verbose)

            # drop audio files with less than pre-decided number of samples
            if len(signal) >= args.sample_rate: # i.e. only those longer than 1 sec

                # ensure strict consistency of the length of the signal (exactly 1 second)
                signal = signal[:args.sample_rate]

                # extract MFCCs (mfcc() does FFT under the hood)
                signums  = librosa.feature.mfcc(          signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                #signums = librosa.feature.melspectrogram(signal, sample_rate,                n_fft=n_fft, hop_length=hop_length)

                # store data for analysed track
                traindata[Aimx.TrainData.MFCC  ].append(signums.T.tolist())
                traindata[Aimx.TrainData.LABELS].append(label_id)
                traindata[Aimx.TrainData.FILES ].append(af_path)
                print_info("{}: {}".format(cyansky(af_path), label_id), verbose = args.verbose)

        label_id += 1
    
    print("\n")
    return traindata, traindata_id
                
if __name__ == "__main__":

    signal_numerization_params = {
        "n_mfcc"        : args.n_mfcc,
        "n_fft"         : args.n_fft,
        "hop_length"    : args.hop_length,
        "sample_rate"   : args.sample_rate, 
        "load_duration" : args.load_duration
    }
    
    start_time = time.time()

    traindata, traindata_id = dataprep(args.dataset_path, n_mfcc = args.n_mfcc,        
                                                           n_fft = args.n_fft,         
                                                      hop_length = args.hop_length,
                                                     sample_rate = args.sample_rate, 
                                                   load_duration = args.load_duration)
    traindata_filename = traindata_id + ".json"

    dataprep_duration = timedelta(seconds = round(time.time() - start_time))
    timestamp = timestamp_now()

    traindata[Aimx.TIMESTAMP] = timestamp
    traindata[Aimx.DURATION]  = str(dataprep_duration)

    # save MFCCs to the traindata file
    save_traindata(traindata, traindata_filename)

    # save as most recent data preprocess result metadata
    save_dataprep_result_meta(traindata_filename, traindata[Aimx.TrainData.MAPPING], timestamp, str(dataprep_duration), total_audios_length_sec,
                              signal_numerization_params)
    
    print_info("Finished {} at {} with wall clock time: {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp),
                                                                    lightyellow(dataprep_duration)))