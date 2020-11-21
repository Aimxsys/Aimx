from itertools import islice
from pathlib import PurePath
from pathlib import Path
import argparse
import librosa
import json
import math
import os

from common_utils import *

# Download from https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

AUDIO_DATASET_DIR_DEFAULT_NAME = "dataset"
AUDIO_DATASET_DIR_DEFAULT      = os.path.join(os.getcwd(), AUDIO_DATASET_DIR_DEFAULT_NAME)

# Calling without -dataset_path               will expect to find the default ./dataset directory.
# Calling with   "-dataset_path mydir"        will expect to find a           ./mydir   directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' preprocessing audio files to extract the dataset'
                                               ' later to be fed into a neural network for training.')

parser.add_argument("-dataset_path",   type = Path,               help = 'Path to a dataset of sound files.')
parser.add_argument("-cutname",        action ='store_true',      help = 'Will generate a json name with no details (cut).')
parser.add_argument("-verbose",        action ='store_true',      help = 'Will print more detailed output messages.')
parser.add_argument("-dataset_depth",  default =     5, type=int, help = 'Number of files to consider from each category.')
parser.add_argument("-n_mfcc",         default =    13, type=int, help = 'Number of MFCC to extract.')
parser.add_argument("-n_fft",          default =  2048, type=int, help = 'Length of the FFT window.   Measured in # of samples.')
parser.add_argument("-hop_length",     default =   512, type=int, help = 'Sliding window for the FFT. Measured in # of samples.')
parser.add_argument("-num_segments",   default =     5, type=int, help = 'Number of segments we want to divide sample tracks into.')
parser.add_argument("-sample_rate",    default = 22050, type=int, help = 'Sample rate at which to read the audio files.')
parser.add_argument("-track_duration", default =     1, type=int, help = 'Only load up to this much audio (in seconds).')

args = parser.parse_args()

############################## Command Argument Verification ##############################

if provided(args.dataset_path) and not args.dataset_path.exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.dataset_path)))

if not provided(args.dataset_path) and not Path(AUDIO_DATASET_DIR_DEFAULT).exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain default dataset directory " + quote(pinkred(AUDIO_DATASET_DIR_DEFAULT_NAME)))

###########################################################################################
# Example command:
# dataprep_asr.py -dataset_path ../workdir/dataset_c1_f1 -n_mfcc 13 -n_fft 2048 -hop_length 512 -num_segments 5 -sample_rate 22050 -track_duration 30

ARG_AUDIO_DATASET_FILES_DIR  = args.dataset_path if provided(args.dataset_path) else AUDIO_DATASET_DIR_DEFAULT

print_info("=============================================================================")
print_info("Expecting audio files in ARG_AUDIO_DATASET_FILES_DIR =", ARG_AUDIO_DATASET_FILES_DIR)
print_info("=============================================================================")

def preprocess_dataset(dataset_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5, sample_rate = 22050, track_duration = 30):
    """
    Extracts MFCC from music dataset and saves them into a json file along witgh genre labels.
        :param  dataset_path (str): Path to dataset.
        :param        n_mfcc (int): Number of MFCC to extract.
        :param         n_fft (int): Length of the FFT window.   Measured in # of samples.
        :param    hop_length (int): Sliding window for the FFT. Measured in # of samples.
        :param: num_segments (int): Number of segments we want to divide sample tracks into.
    """
    json_filename = compose_json_filename(args.dataset_depth, dataset_path, n_mfcc, n_fft, hop_length, num_segments, sample_rate, track_duration)

    # dictionary to store mapping, labels, and MFCC
    traindata = {
        "mapping": [],
         "labels": [],
          "files": [],
           "mfcc": []
    }

#    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
#    expected_num_of_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length) # mfccs are calculater per hop

    print_info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv preprocess_dataset()")
    print_info("json_filename  =", json_filename)
    print_info("n_mfcc         =", n_mfcc)
    print_info("n_fft          =", n_fft)
    print_info("hop_length     =", hop_length)
    print_info("num_segments   =", num_segments)
    print_info("sample_rate    =", sample_rate)
    print_info("track_duration =", track_duration)

    # loop through all subfolders
    for dir_index, (dirpath, subdirpaths, audio_filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing at subfolder level
        if PurePath(dirpath).name is PurePath(dataset_path).name:
            continue

        # save genre label (i.e. subfolder name) in the mapping
        category_label = PurePath(dirpath).name
        traindata["mapping"].append(category_label)
        print_info("\nProcessing: {}".format(category_label))

        # process all audio files in subfolders
        for pbi, audio_filename in enumerate(islice(audio_filenames, args.dataset_depth)):

            if not audio_filename.endswith(".wav"):
                continue
            
            progress_bar(pbi, min(len(audio_filenames), args.dataset_depth))

		    # load audio file
            audio_file_path     = os.path.join(dirpath, audio_filename)
            signal, sample_rate = librosa.load(audio_file_path, sr = sample_rate, duration = track_duration)
            print_info("\nTotal samples in signal (audio track) {} = {}".format(PurePath(audio_file_path).name, len(signal)),
                       verbose = args.verbose)

            # drop audio files with less than pre-decided number of samples
            if len(signal) >= args.sample_rate: # i.e. only those longer than 1 sec

                # ensure strict consistency of the length of the signal (exactly 1 second)
                signal = signal[:args.sample_rate]

                # extract MFCCs
                mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                # store data for analysed track
                traindata["mfcc" ].append(mfcc.T.tolist())
                traindata["labels"].append(dir_index-1)
                traindata["files" ].append(audio_file_path)
                print_info("{}: {}".format(cyansky(audio_file_path), dir_index-1), verbose = args.verbose)

    # save MFCCs to json file
    save_traindata(traindata, json_filename)

    # save recent data preprocess result metadata
    save_dataprep_result_meta(json_filename)
                
if __name__ == "__main__":
    preprocess_dataset(ARG_AUDIO_DATASET_FILES_DIR, n_mfcc = args.n_mfcc,        
                                            n_fft = args.n_fft,         
                                       hop_length = args.hop_length,
                                     num_segments = args.num_segments,
                                      sample_rate = args.sample_rate, 
                                   track_duration = args.track_duration)