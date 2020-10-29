from pathlib import Path
import argparse
import json
import os
import math
import librosa
from pathlib import PurePath

# TODO: Move these to Common/utils.py
def exists(x):
    return x is not None

def provided(cmd_arg):
    return cmd_arg is not None

def quote(me):
    return '\'' + me + '\''

# Download from https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

AUDIO_DATASET_DIR_DEFAULT_NAME = "dataset"
AUDIO_DATASET_DIR_DEFAULT      = os.path.join(os.getcwd(), AUDIO_DATASET_DIR_DEFAULT_NAME)

# Calling without -dataset_path               will expect to find the default ./dataset directory.
# Calling with   "-dataset_path mydir"        will expect to find a           ./mydir   directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' preprocessing audio files to extract the dataset'
                                               ' later to be fed into a neural network for training.')

parser.add_argument("-dataset_path", type = Path,               help = 'Path to a dataset of sound files.')
parser.add_argument("-n_mfcc",       default = 13,    type=int, help = 'Number of MFCCs to extract.')
parser.add_argument("-n_fft",        default = 2048,  type=int, help = 'Length of the FFT window.   Measured in # of samples.')
parser.add_argument("-hop_length",   default = 512,   type=int, help = 'Sliding window for the FFT. Measured in # of samples.')
parser.add_argument("-num_segments", default = 5,     type=int, help = 'Number of segments we want to divide sample tracks into.')
parser.add_argument("-sample_rate",  default = 22050, type=int, help = 'Sample rate at which to read the audio files.')

args = parser.parse_args()

############################## Command Argument Verification ##############################

if provided(args.dataset_path) and not args.dataset_path.exists():
    raise FileNotFoundError("Provided directory " + quote(str(args.dataset_path)) + " not found.")

if not provided(args.dataset_path) and not Path(AUDIO_DATASET_DIR_DEFAULT).exists():
    raise FileNotFoundError("Default directory " + quote(AUDIO_DATASET_DIR_DEFAULT_NAME) + " not found.")

###########################################################################################
# Example command:
# preprocess.py -dataset_path dataset_1class_1file -n_mfcc 13 -n_fft 2048 -hop_length 512 -num_segments 5

PAR_AUDIO_DATASET_FILES_DIR  = args.dataset_path if exists(args.dataset_path) else AUDIO_DATASET_DIR_DEFAULT
PAR_N_MFCC                   = args.n_mfcc       # default: 13
PAR_N_FFT                    = args.n_fft        # default: 2048
PAR_HOP_LENGTH               = args.hop_length   # default: 512
PAR_NUM_SEGMENTS             = args.num_segments # default: 5
PAR_SAMPLE_RATE              = args.sample_rate  # default: 22050

TRACK_DURATION    = 30    # seconds
SAMPLES_PER_TRACK = PAR_SAMPLE_RATE * TRACK_DURATION

print("=============================================================================")
print("Expecting audio files in PAR_AUDIO_DATASET_FILES_DIR =", PAR_AUDIO_DATASET_FILES_DIR)
print("=============================================================================")

def save_mfcc(dataset_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    """
    Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param  dataset_path (str): Path to dataset.
        :param     json_path (str): Path to json file used to save MFCCs.
        :param        n_mfcc (int): Number of MFCCs to extract.
        :param         n_fft (int): Length of the FFT window.   Measured in # of samples.
        :param    hop_length (int): Sliding window for the FFT. Measured in # of samples.
        :param: num_segments (int): Number of segments we want to divide sample tracks into.
    """
    json_path = PurePath(dataset_path).name + ".json"

    # dictionary to store mapping, labels, and MFCCs
    datann = {
        "mapping": [],
         "labels": [],
           "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_of_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length) # mfccs are calculater per hop

    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv save_mfcc()")
    print("json_path    =", json_path)
    print("n_mfcc       =", n_mfcc)
    print("n_fft        =", n_fft)
    print("hop_length   =", hop_length)
    print("num_segments =", num_segments)

    # loop through all genre subfolder
    for dir_index, (dirpath, subdirpaths, audio_filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre subfolder level
        if dirpath is not PurePath(dataset_path).name:

            # save genre label (i.e. subfolder name) in the mapping
            semantic_label = PurePath(dirpath).name
            datann["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for audio_filename in audio_filenames:

		        # load audio file
                audio_file_path     = os.path.join(dirpath, audio_filename)
                signal, sample_rate = librosa.load(audio_file_path, sr=PAR_SAMPLE_RATE)
                print("Total samples in the signal (audio track) =", len(signal))

                # process all segments of the audio file, extract mfccs
                # and store the data to be fed to the for NN processing
                for segment in range(num_segments):

                    # calculate first and last sample for the current segment
                    seg_first_sample = samples_per_segment * segment
                    seg_last_sample  = seg_first_sample + samples_per_segment

                    # extract mfccs for each segment
                    mfcc = librosa.feature.mfcc(signal[seg_first_sample:seg_last_sample],
                                                sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == expected_num_of_mfcc_vectors_per_segment:
                        datann["mfcc"].append(mfcc.tolist())
                        datann["labels"].append(dir_index-1) # -1 is to eliminate the top-level dir
                        print("{}, segment:{}".format(audio_file_path, segment+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(datann, fp, indent=4)
        
if __name__ == "__main__":
    save_mfcc(PAR_AUDIO_DATASET_FILES_DIR, n_mfcc = PAR_N_MFCC,
                                            n_fft = PAR_N_FFT,
                                       hop_length = PAR_HOP_LENGTH,
                                     num_segments = PAR_NUM_SEGMENTS)