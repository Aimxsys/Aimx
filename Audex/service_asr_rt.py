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

class _AsrServiceRT:
    """
    Singleton class for word detecting inference with trained models.
    """
    model     = None
    modelType = None

    _instance = None

    # Extract label mapping from the dataprep result metadata file
    label_mapping = get_dataprep_result_meta()[Aimx.Dataprep.DATASET_VIEW]

    # Numbers in the two rows below are related and go together,
    # their calculation may be automated some time in the future.
    #                           Len    Con  Filename Inference
    inference_report_headers = "{:<10}  {:<4}  {:<16} {:<20}"
    inference_report_columns = "{:>5.2f} - {:<3} {:<4}  {:<25} {:<20}"

    # This dataprep is for ASR CNN inference
    def numerize(self, audio_signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        # Extract mfccs from an audio signal.
        :param     n_mfcc (int): # of coefficients to extract
        :param      n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return mfccs (ndarray): 2-d numpy array with MFCC data of shape (# time steps, # coefficients)
        """
        #mfccs = np.empty([n_mfcc, 44]) # TODO: Revisit this line later

        # Trim longer signals so they're exactly 1 second in length to ensure consistency of the lengths
        # Otherwise you'll get an error that starts with a warning (here a 1-second TF model was called on a 2-second audio interval):
        #  "WARNING:tensorflow:Model was constructed with shape (None, 44, 13, 1) for input Tensor("conv2d_input:0", shape=(None, 44, 13, 1), dtype=float32),
        # but it was called on an input with incompatible shape (None, 87, 13, 1)."
        # Therefore, TODO: Generalize the line below so that the array interval length is extracted from the model.
        # LENGTH_SEC = 1
        # self.af_signalsec = self.af_signal[startsec*self.af_sr : (startsec + LENGTH_SEC)*self.af_sr] # resulting shape (22020,)

        mfccs = librosa.feature.mfcc(audio_signal, sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        if self.modelType == 'cnn':
            # convert the 2d MFCC array into a 4d array to feed to the model for prediction:
            #            (# segments, # coefficients)
            # (# samples, # segments, # coefficients, # channels)
            mfccs = mfccs[np.newaxis, ..., np.newaxis] # shape for CNN model
        elif self.modelType == 'rnn':
            mfccs = mfccs[..., np.newaxis]             # shape for RNN model
        else:
            raise Exception("ASR received an unknown model type: " + self.modelType)

        return mfccs.T

    def predict(self, mfccs):
        # make a prediction and get the predicted label and confidence
        predictions   = self.model.predict(mfccs)
        confidence    = np.max(predictions)
        predmax_index = np.argmax(predictions)
        inference     = self.label_mapping[predmax_index]

        return inference, confidence

    def report(self, predicted_word, confidence, confidence_threshold=0.9):
        if True: # Original criteria (TODO: change when ready): predicted_word in extract_filename(self.af_fullpath):
            # inference is correct
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(   cyan("{:.2f}".format(confidence)), cyan(predicted_word)))
            else:
                print(self.inference_report_columns.format(pinkred("{:.2f}".format(confidence)), cyan(predicted_word)))
        else:
            # inference is wrong
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(    red("{:.2f}".format(confidence)), pinkred(predicted_word)))
            else:
                print(self.inference_report_columns.format(pinkred("{:.2f}".format(confidence)), pinkred(predicted_word)))

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
            print_info("[DONE]")
        except Exception as e:
            print(pinkred("\nException caught while trying to load the model: " + quote_path(model_path)))
            print(pinkred("Exception message: ") + red(str(e)))
    return _AsrServiceRT._instance