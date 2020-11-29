from itertools import islice
from pathlib import PurePath
from pathlib import Path
from datetime import timedelta
import time
import argparse
import librosa
import math
import os

from common_utils import *
from audex_utils  import *

# Download from https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

DATASET_DIR_DEFAULT_NAME = "dataset"
DATASET_DIR_DEFAULT      = os.path.join(os.getcwd(), DATASET_DIR_DEFAULT_NAME)

# TODO: Replace the hard-coded line below with automatically extracted dirlabels
DATASET_VIEW_DEFAULT = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Calling without -dataset_path               will expect to find the default ./dataset directory.
# Calling with   "-dataset_path mydir"        will expect to find a           ./mydir   directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' preprocessing audio files to extract the training'
                                               ' data later to be fed into a neural network.')

parser.add_argument("-dataset_view",   nargs='*', default = DATASET_VIEW_DEFAULT, help = 'Specific directories (labels) to go through.')
parser.add_argument("-dataset_path",   type = Path,               help = 'Path to a dataset of sound files.')
parser.add_argument("-dataset_depth",  default =     5, type=int, help = 'Number of files to consider from each category.')
parser.add_argument("-n_mfcc",         default =    13, type=int, help = 'Number of MFCC to extract.')
parser.add_argument("-n_fft",          default =  2048, type=int, help = 'Length of the FFT window.   Measured in # of samples.')
parser.add_argument("-hop_length",     default =   512, type=int, help = 'Sliding window for the FFT. Measured in # of samples.')
parser.add_argument("-num_segments",   default =     5, type=int, help = 'Number of segments we want to divide sample tracks into.')
parser.add_argument("-sample_rate",    default = 22050, type=int, help = 'Sample rate at which to read the audio files.')
parser.add_argument("-track_duration", default =    30, type=int, help = 'Only load up to this much audio (in seconds).')
parser.add_argument("-cutname",        action ='store_true',      help = 'Will generate a json name with no details (cut).')
parser.add_argument("-verbose",        action ='store_true',      help = 'Will print more detailed output messages.')
parser.add_argument("-example",        action ='store_true',      help = 'Will show a working example on how to call the script.')

args = parser.parse_args()

########################## Command Argument Handling & Verification #######################

if args.example:
    print_info(nameofthis(__file__) + " -dataset_path ../workdir/dataset -dataset_depth 5")
    exit()

if provided(args.dataset_path) and not args.dataset_path.exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.dataset_path)))

if not provided(args.dataset_path) and not Path(DATASET_DIR_DEFAULT).exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain default dataset directory " + quote(pinkred(DATASET_DIR_DEFAULT_NAME)))

###########################################################################################
# Example command:
# dataprep_genre.py -dataset_path ../workdir/dataset_c1_f1 -n_mfcc 13 -n_fft 2048 -hop_length 512 -num_segments 5 -sample_rate 22050 -track_duration 30

ARG_DATASET_FILES_DIR = args.dataset_path if provided(args.dataset_path) else DATASET_DIR_DEFAULT
if Aimx.Dataprep.ALL_DIR_LABELS in args.dataset_view: # special value ok for now, may need to be rewritten in a better way
    args.dataset_view = get_all_dirs_in(ARG_DATASET_FILES_DIR)

SAMPLES_PER_TRACK = args.sample_rate * args.track_duration

print_info("=============================================================================")
print_info("Expecting audio files in ARG_DATASET_FILES_DIR =", ARG_DATASET_FILES_DIR)
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
    traindata_id = compose_traindata_id(args.dataset_depth, args.dataset_view, dataset_path,
                                       n_mfcc, n_fft, hop_length, num_segments, sample_rate, track_duration)

    # dictionary to store mapping, labels, and MFCC
    traindata = {
        Aimx.TIMESTAMP          : [],
        Aimx.DURATION           : [],
        Aimx.TrainData.MAPPING  : [],
        Aimx.TrainData.LABELS   : [],
        Aimx.TrainData.MFCC     : []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_of_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length) # mfccs are calculater per hop

    print_info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv preprocess_dataset()")
    print_info("traindata_id   =", traindata_id)
    print_info("n_mfcc         =", n_mfcc)
    print_info("n_fft          =", n_fft)
    print_info("hop_length     =", hop_length)
    print_info("num_segments   =", num_segments)
    print_info("sample_rate    =", sample_rate)
    print_info("track_duration =", track_duration)

    label_id = 0

    # loop through all subfolders
    for dirpath, _, audio_filenames in os.walk(dataset_path):

        # ensure we're processing at subfolder level
        if PurePath(dirpath).name is PurePath(dataset_path).name:
            continue
        
        if PurePath(dirpath).stem not in args.dataset_view:
            continue

        # save genre label (i.e. subfolder name) in the mapping
        label_name = PurePath(dirpath).name
        traindata[Aimx.TrainData.MAPPING].append(label_name)
        print_info("\n\nProcessing label {} {}".format(cyan(label_id), label_name))

        # process all audio files in subfolders
        for i, audio_filename in enumerate(islice(audio_filenames, args.dataset_depth)):
                
            if not audio_filename.endswith(".wav"):
                continue
                
            progress_bar(i, min(len(audio_filenames), args.dataset_depth))

		    # load audio file
            audio_file_path     = os.path.join(dirpath, audio_filename)
            signal, sample_rate = librosa.load(audio_file_path, sr = sample_rate, duration = track_duration)
            print_info("\nTotal samples in signal (audio track) {} = {}".format(PurePath(audio_file_path).name, len(signal)),
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
                    traindata[Aimx.TrainData.MFCC  ].append(mfcc.tolist())
                    traindata[Aimx.TrainData.LABELS].append(label_id)
                    print_info("{}, segment:{}".format(cyansky(audio_file_path), segment+1), verbose = args.verbose)

        label_id += 1

    return traindata, traindata_id
                
if __name__ == "__main__":

    start_time = time.time()

    traindata, traindata_id = preprocess_dataset(ARG_DATASET_FILES_DIR, n_mfcc = args.n_mfcc,        
                                                                         n_fft = args.n_fft,         
                                                                    hop_length = args.hop_length,
                                                                  num_segments = args.num_segments,
                                                                   sample_rate = args.sample_rate, 
                                                                track_duration = args.track_duration)
    traindata_filename = traindata_id + ".json"

    dataprep_duration = timedelta(seconds = round(time.time() - start_time))
    traindata[Aimx.TIMESTAMP] = timestamp_now()
    traindata[Aimx.DURATION]  = str(dataprep_duration)

    # save MFCCs to the traindata file
    save_traindata(traindata, traindata_filename)

    # save as most recent data preprocess result metadata
    save_dataprep_result_meta(traindata_filename, traindata[Aimx.TrainData.MAPPING], str(dataprep_duration))

    print_info("Finished {} at {} with wall clock time: {} ".format(cyansky(nameofthis(__file__)),
                                                                    lightyellow(timestamp_now()),
                                                                    lightyellow(dataprep_duration)))