#!/usr/bin/env python

import librosa
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

class _AsrServiceRT:
    """
    Singleton class for word detecting inference with trained models.
    """
    model     = None
    modelType = None

    _instance = None

    # Extract label mapping from the dataprep result metadata file
    label_mapping = get_dataprep_result_meta()[Aimx.Dataprep.DATASET_VIEW]

    #                           Con    Infer  dB
    inference_report_columns = "{:<4}  {:<15} {:<5}"

    # This dataprep is for ASR CNN inference
    def signumerize(self, audio_signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        # Extract mfccs from an audio signal.
        :param     n_mfcc (int): # of coefficients to extract
        :param      n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return mfccs (ndarray): 2-d numpy array with MFCC data of shape (# time steps, # coefficients)
        """
        # extract mfccs (mfcc() does FFT under the hood)
        signums = librosa.feature.mfcc(audio_signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        #signums = librosa.feature.melspectrogram(audio_signal, sample_rate, n_fft=n_fft, hop_length=hop_length)
        if self.modelType == 'cnn':
            # convert the 2d MFCC array into a 4d array to feed to the model for prediction:
            #            (# segments, # coefficients)
            # (# samples, # segments, # coefficients, # channels)
            signums = signums[np.newaxis, ..., np.newaxis] # shape for CNN model
        elif self.modelType == 'rnn' or self.modelType == 'ann':
            signums = signums[..., np.newaxis]             # shape for RNN model
        else:
            raise Exception(pinkred("ASR received an unknown model type: " + quote(self.modelType)))

        return signums.T

    def predict(self, mfccs):
        # make a prediction and get the predicted label and confidence
        predictions   = self.model.predict(mfccs)
        confidence    = np.max(predictions)
        predmax_index = np.argmax(predictions)
        inference     = self.label_mapping[predmax_index]

        return inference, confidence

    def report(self, predicted_word, confidence, confidence_threshold=0.9, addinfo=""):
        if True: # Original criteria (TODO: change when ready): predicted_word in extract_filename(self.af_fullpath):
            # inference is correct
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(   cyan("{:.2f}".format(confidence)), cyan(predicted_word), addinfo))
            else:
                print(self.inference_report_columns.format(pinkred("{:.2f}".format(confidence)), cyan(predicted_word), addinfo))
        else:
            # inference is wrong
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(    red("{:.2f}".format(confidence)), pinkred(predicted_word), addinfo))
            else:
                print(self.inference_report_columns.format(pinkred("{:.2f}".format(confidence)), pinkred(predicted_word), addinfo))

def CreateAsrServiceRT(model_path):
    """
    Factory function for AsrServiceRT class.
    """
    # ensure an instance is created only on first call
    if  _AsrServiceRT._instance is None:
        _AsrServiceRT._instance = _AsrServiceRT()
        try:
            print_info("|||||| Loading model " + quote_path(model_path) + "... ", end="")
            _AsrServiceRT.model     = keras.models.load_model(model_path)
            _AsrServiceRT.modelType = extract_filename(model_path)[6:9] # from name: model_cnn_...
            print_info("[DONE (model loaded)]", quote_path(model_path))
        except Exception as e:
            print(pinkred("\nException caught while trying to load the model: " + quote_path(model_path)))
            print(pinkred("Exception message: ") + red(str(e)))
    return _AsrServiceRT._instance