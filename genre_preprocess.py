from pathlib import PurePath
from pathlib import Path
import argparse
import librosa
import json
import math
import os

from common_utils import *
from  genre_utils import DATA_PREPROCESS_RESULT_METADATA_FILENAME

# Download from https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

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
parser.add_argument("-n_mfcc",         default =    13, type=int, help = 'Number of MFCCs to extract.')
parser.add_argument("-n_fft",          default =  2048, type=int, help = 'Length of the FFT window.   Measured in # of samples.')
parser.add_argument("-hop_length",     default =   512, type=int, help = 'Sliding window for the FFT. Measured in # of samples.')
parser.add_argument("-num_segments",   default =     5, type=int, help = 'Number of segments we want to divide sample tracks into.')
parser.add_argument("-sample_rate",    default = 22050, type=int, help = 'Sample rate at which to read the audio files.')
parser.add_argument("-track_duration", default =    30, type=int, help = 'Only load up to this much audio (in seconds).')

args = parser.parse_args()

############################## Command Argument Verification ##############################

if provided(args.dataset_path) and not args.dataset_path.exists():
    raise FileNotFoundError("Directory " + pinkred(os.getcwd()) + " does not contain requested path " + quote(pinkred(str(args.dataset_path))))

if not provided(args.dataset_path) and not Path(AUDIO_DATASET_DIR_DEFAULT).exists():
    raise FileNotFoundError("Directory " + pinkred(os.getcwd()) + " does not contain default dataset directory " + quote(pinkred(AUDIO_DATASET_DIR_DEFAULT_NAME)))

###########################################################################################
# Example command:
# preprocess.py -dataset_path dataset_c1_f1 -n_mfcc 13 -n_fft 2048 -hop_length 512 -num_segments 5 -sample_rate 22050 -track_duration 30

ARG_AUDIO_DATASET_FILES_DIR  = args.dataset_path if provided(args.dataset_path) else AUDIO_DATASET_DIR_DEFAULT

SAMPLES_PER_TRACK = args.sample_rate * args.track_duration

print_info("=============================================================================")
print_info("Expecting audio files in ARG_AUDIO_DATASET_FILES_DIR =", ARG_AUDIO_DATASET_FILES_DIR)
print_info("=============================================================================")

def save_mfcc(dataset_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5, sample_rate = 22050, track_duration = 30):
    """
    Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param  dataset_path (str): Path to dataset.
        :param        n_mfcc (int): Number of MFCCs to extract.
        :param         n_fft (int): Length of the FFT window.   Measured in # of samples.
        :param    hop_length (int): Sliding window for the FFT. Measured in # of samples.
        :param: num_segments (int): Number of segments we want to divide sample tracks into.
    """
    json_filename = PurePath(dataset_path).name # the data json file name
    if args.cutname:
        json_filename += "_cut"
    else:
        json_filename += "_" + str(n_mfcc)         + "m" \
                      +  "_" + str(n_fft)          + "w" \
                      +  "_" + str(hop_length)     + "h" \
                      +  "_" + str(num_segments)   + "i" \
                      +  "_" + str(sample_rate)    + "r" \
                      +  "_" + str(track_duration) + "s"
    json_filename += ".json"

    # dictionary to store mapping, labels, and MFCCs
    datann = {
        "mapping": [],
         "labels": [],
           "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_of_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length) # mfccs are calculater per hop

    print_info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv save_mfcc()")
    print_info("json_filename  =", json_filename)
    print_info("n_mfcc         =", n_mfcc)
    print_info("n_fft          =", n_fft)
    print_info("hop_length     =", hop_length)
    print_info("num_segments   =", num_segments)
    print_info("sample_rate    =", sample_rate)
    print_info("track_duration =", track_duration)

    # loop through all genre subfolder
    for dir_index, (dirpath, subdirpaths, audio_filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre subfolder level
        if dirpath is not PurePath(dataset_path).name:

            # save genre label (i.e. subfolder name) in the mapping
            semantic_label = PurePath(dirpath).name
            datann["mapping"].append(semantic_label)
            print_info("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for i, audio_filename in enumerate(audio_filenames):
                
                progress_bar(i, len(audio_filenames))

		        # load audio file
                audio_file_path     = os.path.join(dirpath, audio_filename)
                signal, sample_rate = librosa.load(audio_file_path, sr = sample_rate, duration = track_duration)
                print_info("Total samples in signal (audio track) {} = {}".format(PurePath(audio_file_path).name, len(signal)),
                           verbose = args.verbose)

                # process all segments of the audio file, extract mfccs
                # and store the data to be fed to the NN for processing
                for segment in range(num_segments):

                    # calculate first and last sample for the current segment
                    seg_first_sample = samples_per_segment * segment
                    seg_last_sample  = seg_first_sample + samples_per_segment

                    # extract mfccs for each segment
                    mfcc = librosa.feature.mfcc(signal[seg_first_sample:seg_last_sample],
                                                sample_rate, n_mfcc = n_mfcc, n_fft = n_fft, hop_length = hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == expected_num_of_mfcc_vectors_per_segment:
                        datann["mfcc"].append(mfcc.tolist())
                        datann["labels"].append(dir_index-1) # -1 is to eliminate the top-level dir
                        print_info("{}, segment:{}".format(cyansky(audio_file_path), segment+1), verbose = args.verbose)

    # save MFCCs to json file
    Path("data_json").mkdir(parents=True, exist_ok=True)
    data_json_fullpath = os.path.join("data_json", json_filename)
    with open(data_json_fullpath, "w") as data_file:
        print_info("\n|||||| Writing data file", cyansky(data_json_fullpath), "... ", end="")
        json.dump(datann, data_file, indent=4)
        print_info("[DONE]")

    # save recent data preprocess result metadata
    prep_result_meta = {"most_recent_output": {}, "duration": {} }
    prep_result_meta["most_recent_output"] = data_json_fullpath
    with open(DATA_PREPROCESS_RESULT_METADATA_FILENAME, 'w') as fp: 
        json.dump(prep_result_meta, fp)
        
if __name__ == "__main__":
    save_mfcc(ARG_AUDIO_DATASET_FILES_DIR, n_mfcc = args.n_mfcc,        
                                            n_fft = args.n_fft,         
                                       hop_length = args.hop_length,
                                     num_segments = args.num_segments,
                                      sample_rate = args.sample_rate, 
                                   track_duration = args.track_duration)