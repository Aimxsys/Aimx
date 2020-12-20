#!/usr/bin/env python

import librosa
import argparse
import tensorflow.keras as keras
import numpy as np
import sys
import os

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import *
from Audex.utils.utils_audex  import Aimx
from Audex.utils.utils_audex  import get_dataprep_result_meta
from Audex.utils.utils_audex  import get_actual_model_path

def process_clargs():
    # Calling with "-inferdata_path /to/file" will expect to find the file in ./to directory.
    parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with inference on audio files.')

    parser.add_argument("-model_path", default=Aimx.MOST_RECENT_OUTPUT, type = Path, help = 'Path to the model to be loaded.')
    parser.add_argument("-inferdata_path",       type = Path,                        help = 'Path to the audio files on which model inference is to be tested.')
    parser.add_argument("-confidence_threshold", default = 0.9, type=float,          help = 'Highlight results if confidence is higher than this threshold.')

    parser.add_argument("-n_mfcc",         default =    13, type=int, help = 'Number of MFCC to extract.')
    parser.add_argument("-n_fft",          default =  2048, type=int, help = 'Length of the FFT window.   Measured in # of samples.')
    parser.add_argument("-hop_length",     default =   512, type=int, help = 'Sliding window for the FFT. Measured in # of samples.')
    parser.add_argument("-num_segments",   default =     5, type=int, help = 'Number of segments we want to divide sample tracks into.')
    parser.add_argument("-sample_rate",    default = 22050, type=int, help = 'Sample rate at which to read the audio files.')
    parser.add_argument("-track_duration", default =     1, type=int, help = 'Only load up to this much audio (in seconds).')
    parser.add_argument("-example",        action ='store_true',      help = 'Will show a working example on how to call the script.')

    args = parser.parse_args()

    print_script_start_preamble(nameofthis(__file__), vars(args))

    ############################## Command Argument Handling & Verification ##############################

    if args.example:
        print_info(nameofthis(__file__) + " -model_path most_recent_output -inferdata_path ../workdir/infer/signal_down_five")
        exit()

    if provided(args.inferdata_path) and not args.inferdata_path.exists():
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

    args.model_path = get_actual_model_path(args.model_path)

    ######################################################################################################
    
    return args

class _WordetectService:
    """
    Singleton class for word detecting inference with trained models.
    """
    model     = None
    modelType = None

    afile_fullpath    = None
    afile_signal      = None
    afile_sample_rate = None
    afile_duration    = None

    _instance = None

    label_mapping = get_dataprep_result_meta()[Aimx.Dataprep.DATASET_VIEW]

    # Numbers in the two rows below are related and go together,
    # their calculation may be automated some time in the future.
    inference_report_headers = "{:<5}  {:<4}  {:<16} {:<20}"
    inference_report_columns = "{:>5.2f}  {:<4}  {:<25} {:<20}"

    def load_audiofile(self, audiofile_fullpath, track_duration):
        self.afile_fullpath = audiofile_fullpath
        self.afile_signal, self.afile_sample_rate = librosa.load(audiofile_fullpath, duration = track_duration)
        full_afile_signal, _                      = librosa.load(audiofile_fullpath)
        self.afile_duration = librosa.get_duration(full_afile_signal, self.afile_sample_rate)

    # This dataprep is for ASR CNN inference
    def dataprep(self, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        # Eextract mfccs from an audio file.
        :param     n_mfcc (int): # of coefficients to extract
        :param      n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return mfccs (ndarray): 2-d numpy array with MFCC data of shape (# time steps, # coefficients)
        """
        #mfccs = np.empty([n_mfcc, 44]) # TODO: Revisit this line later

        # trim longer signals at exactly 1 second to ensure consistency of the lengths
        self.afile_signal = self.afile_signal[:self.afile_sample_rate]

        mfccs = librosa.feature.mfcc(self.afile_signal,
                                     self.afile_sample_rate,
                                     n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        if self.modelType == 'cnn':
            # convert the 2d MFCC array into a 4d array to feed to the model for prediction:
            #            (# segments, # coefficients)
            # (# samples, # segments, # coefficients, # channels)
            mfccs = mfccs[np.newaxis, ..., np.newaxis] # shape for CNN model
        elif self.modelType == 'rnn':
            mfccs = mfccs[..., np.newaxis]             # shape for RNN model
        else:
            raise Exception("Wordetect received an unknown model type: " + self.modelType)

        return mfccs.T

    def predict(self, mfccs):
        # make a prediction and get the predicted label and confidence
        predictions   = self.model.predict(mfccs)
        confidence    = np.max(predictions)
        predmax_index = np.argmax(predictions)
        inference     = self.label_mapping[predmax_index]

        return inference, confidence

    def highlight(self, predicted_word, confidence, confidence_threshold=0.9):
        if predicted_word in extract_filename(self.afile_fullpath):
            # inference is correct
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(self.afile_duration,    cyan("{:.2f}".format(confidence)), yellow(extract_filename(self.afile_fullpath)), cyan(predicted_word)))
            else:
                print(self.inference_report_columns.format(self.afile_duration, pinkred("{:.2f}".format(confidence)), yellow(extract_filename(self.afile_fullpath)), cyan(predicted_word)))
        else:
            # inference is wrong
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(self.afile_duration,     red("{:.2f}".format(confidence)), yellow(extract_filename(self.afile_fullpath)), pinkred(predicted_word)))
            else:
                print(self.inference_report_columns.format(self.afile_duration, pinkred("{:.2f}".format(confidence)), yellow(extract_filename(self.afile_fullpath)), pinkred(predicted_word)))

def CreateWordetectService(model_path):
    """
    Factory function for WordetectService class.
    """
    # ensure an instance is created only on first call
    if  _WordetectService._instance is None:
        _WordetectService._instance = _WordetectService()
        try:
            print_info("|||||| Loading model " + quote_path(model_path) + "... ", end="")
            _WordetectService.model     = keras.models.load_model(model_path)
            _WordetectService.modelType = extract_filename(model_path)[6:9]
            print_info("[DONE]")
        except Exception as e:
            print(pinkred("\nException caught while trying to load the model: " + quote_path(model_path)))
            print(pinkred("Exception message: ") + red(str(e)))
    return _WordetectService._instance

if __name__ == "__main__":

    args = process_clargs()

    wds = CreateWordetectService(args.model_path)
    
    print_info("\nPredicting with dataset view (labels):", wds.label_mapping)
    print_info("On files in:", args.inferdata_path)
    print_info(wds.inference_report_headers.format("Len", "Con", "Filename", "Inference"))

    (_, _, filenames) = next(os.walk(args.inferdata_path))

    for filename in filenames:
        audiofile_fullpath = os.path.join(args.inferdata_path, filename)
        wds.load_audiofile(audiofile_fullpath, args.track_duration)
        if len(wds.afile_signal) >= args.sample_rate: # process only signals of at least 1 sec
            mfccs = wds.dataprep(args.n_mfcc, args.n_fft, args.hop_length)
            w, c  = wds.predict(mfccs)
            wds.highlight(w, c, args.confidence_threshold)