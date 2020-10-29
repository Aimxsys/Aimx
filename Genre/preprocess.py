﻿import json
import os
import math
import librosa
from pathlib import PurePath

# Download from https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

# TODO: Turn these into command line arguments.
DATASET_PATH      = "dataset_3class_1file"
JSON_PATH         = "data_10.json"
SAMPLE_RATE       = 22050 # Hz
TRACK_DURATION    = 30    # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    """
    Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param  dataset_path (str): Path to dataset.
        :param     json_path (str): Path to json file used to save MFCCs.
        :param        n_mfcc (int): Number of MFCCs to extract.
        :param         n_fft (int): Length of the FFT window. Measured in # of samples.
        :param    hop_length (int): Sliding window for FFT.   Measured in # of samples.
        :param: num_segments (int): Number of segments we want to divide sample tracks into.
    """

    # dictionary to store mapping, labels, and MFCCs
    datann = {
        "mapping": [],
         "labels": [],
           "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_of_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length) # mfccs are calculater per hop

    # loop through all genre subfolder
    for dir_index, (dirpath, subdirpaths, audio_filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre subfolder level
        if dirpath is not dataset_path:

            # save genre label (i.e., subfolder name) in the mapping
            semantic_label = PurePath(dirpath).name
            datann["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for audio_filename in audio_filenames:

		        # load audio file
                audio_file_path     = os.path.join(dirpath, audio_filename)
                signal, sample_rate = librosa.load(audio_file_path, sr=SAMPLE_RATE)

                # process all segments of the audio file, extract mfccs
                # and store the data to be fed to the for NN processing
                for segment in range(num_segments):

                    # calculate first and last sample for the current segment
                    seg_first_sample = samples_per_segment * segment
                    seg_last_sample  = seg_first_sample + samples_per_segment

                    # extract mfcc
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
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=5)