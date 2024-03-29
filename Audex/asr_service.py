#!/usr/bin/env python

from itertools import islice
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
from Audex.utils.utils_audex  import get_training_result_meta
from Audex.utils.utils_audex  import get_actual_model_path

from ae import Autoencoder

def process_clargs():
    # Calling with "-inferdata_path /to/file" will expect to find the file in ./to directory.
    parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with inference on audio files.')

    parser.add_argument("-model_path", default=Aimx.MOST_RECENT_OUTPUT, type = Path,         help = 'Path to the model to be loaded.')
    parser.add_argument("-inferdata_path",                              type=Path,           help = 'Path to the audio files on which model inference is to be tested.')
    parser.add_argument("-inferdata_range",      default = [0, 50],     type=int, nargs='*', help = 'Range in -inferdata_path on which to do inference.')
    parser.add_argument("-confidence_threshold", default = 0.9,         type=float,          help = 'Highlight results if confidence is higher than this threshold.')

    parser.add_argument("-signum_type",          default = "mel",       type=str,            help = 'Signal numerization type.')
    parser.add_argument("-n_mfcc",               default =    13,       type=int,            help = 'Number of MFCC to extract.')
    parser.add_argument("-n_fft",                default =  2048,       type=int,            help = 'Length of the FFT window.   Measured in # of samples.')
    parser.add_argument("-hop_length",           default =   512,       type=int,            help = 'Sliding window for the FFT. Measured in # of samples.')
    parser.add_argument("-sample_rate",          default = 22050,       type=int,            help = 'Sample rate at which to read the audio files.')
    parser.add_argument("-load_duration",        default =     1,       type=int,            help = 'Only load up to this much audio (in seconds).')

    parser.add_argument("-example",       action ='store_true',      help = 'Show a working example on how to call the script.')

    args = parser.parse_args()

    ############################## Command Argument Handling & Verification ##############################

    if args.example:
        print_info(nameofthis(__file__) + " -inferdata_path ../workdir/infer/signal_down_backnoise_five_TRIMMED")
        exit()

    if provided(args.inferdata_path) and not args.inferdata_path.exists():
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

    args.model_path = get_actual_model_path(args.model_path)

    args.signum_type   = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["type"]          if not provided(args.signum_type)   else args.signum_type
    args.n_mfcc        = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["n_mfcc"]        if not provided(args.n_mfcc)        else args.n_mfcc
    args.n_fft         = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["n_fft"]         if not provided(args.n_fft)         else args.n_fft
    args.n_hop_length  = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["hop_length"]    if not provided(args.hop_length)    else args.hop_length
    args.sample_rate   = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["sample_rate"]   if not provided(args.hop_length)    else args.sample_rate
    args.load_duration = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["load_duration"] if not provided(args.load_duration) else args.load_duration
    
    ######################################################################################################
    
    print_script_start_preamble(nameofthis(__file__), vars(args))

    return args

class _AsrService:
    """
    Singleton class for word detecting inference with trained models.
    """
    model     = None
    modelType = None

    # audio file currently being analyzed
    af_fullpath        = None
    af_signal          = None
    af_signalsec       = None # which second
    af_sr              = None 
    af_loaded_duration = None # seconds
    af_currsec         = None # the second currently being processed (signumerized)

    _instance = None

    label_mapping = get_dataprep_result_meta()[Aimx.Dataprep.DATASET_VIEW]

    # TODO: Use this to deduce file name column length automatically for inference report
    #filename_column_len = len(max(get_all_filenames_in(args.inferdata_path), key=len))
    
    # Numbers in the two rows below are related and go together,
    # their calculation may be automated some time in the future.
    #                           Len    Con  Filename Inference
    inference_report_headers = "{:<10}  {:<4}  {:<16} {:<20}"
    inference_report_columns = "{:>5.2f} - {:<3} {:<4}  {:<25} {:<20}"

    def load_audiofile(self, af_fullpath, load_duration):
        self.af_fullpath = af_fullpath
        self.af_signal, self.af_sr = librosa.load(af_fullpath, duration=load_duration)
        self.af_loaded_duration    = librosa.get_duration(self.af_signal, self.af_sr)

    # This dataprep is for inference
    def signumerize(self, signum_type="mel", startsec=0, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        # Extract mfccs from an audio file.
        :param   startsec (int): second in the signal from which signumerization starts
        :param     n_mfcc (int): # of coefficients to extract
        :param      n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return mfccs (ndarray): 2-d numpy array with MFCC data of shape (# time steps, # coefficients)
        """
        #mfccs = np.empty([n_mfcc, 44]) # TODO: Revisit this line later

        self.af_currsec = startsec

        # Trim longer signals so they're exactly 1 second in length to ensure consistency of the lengths
        # Otherwise you'll get an error that starts with a warning (here a 1-second TF model is called on a 2-second audio interval):
        #  "WARNING:tensorflow:Model was constructed with shape (None, 44, 13, 1) for input Tensor("conv2d_input:0", shape=(None, 44, 13, 1), dtype=float32),
        # but it was called on an input with incompatible shape (None, 87, 13, 1)."
        # Therefore, TODO: Generalize the line below so that the array interval length is extracted from the model.
        LENGTH_SEC = 1
        self.af_signalsec = self.af_signal[startsec*self.af_sr : (startsec + LENGTH_SEC)*self.af_sr] # (22050,) next undergo mfcc-ing

        # signumerize into spectrograms (FFT is done under the hood of melspectrogram() and mfcc())
        if signum_type == "mel":
            signums = librosa.feature.melspectrogram(self.af_signalsec, self.af_sr,                n_fft=n_fft, hop_length=hop_length) # (44, 128)
        else: # MFCC
            signums = librosa.feature.mfcc(          self.af_signalsec, self.af_sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length) # (44, 13)

        if self.modelType == 'cnn' or self.modelType == 'aen':
            # convert the 2d feature array into a 4d array to feed to the model for prediction:
            #            (# segments, # coefficients)
            # (# samples, # segments, # coefficients, # channels)
            signums = signums[np.newaxis, ..., np.newaxis] # shape for CNN model: # (1, 44, 13, 1)
        elif self.modelType == 'rnn' or self.modelType == 'ann':
            signums = signums[..., np.newaxis]             # shape for RNN model  #    (44, 13, 1)
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

    def report(self, predicted_word, confidence, confidence_threshold=0.9):
        if predicted_word in extract_filename(self.af_fullpath):
            # inference is correct
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(self.af_loaded_duration, self.af_currsec,    cyan("{:.2f}".format(confidence)), yellow(extract_filename(self.af_fullpath)), cyan(predicted_word)))
            else:
                print(self.inference_report_columns.format(self.af_loaded_duration, self.af_currsec, pinkred("{:.2f}".format(confidence)), yellow(extract_filename(self.af_fullpath)), cyan(predicted_word)))
        else:
            # inference is wrong
            if confidence > confidence_threshold:
                print(self.inference_report_columns.format(self.af_loaded_duration, self.af_currsec,     red("{:.2f}".format(confidence)), yellow(extract_filename(self.af_fullpath)), pinkred(predicted_word)))
            else:
                print(self.inference_report_columns.format(self.af_loaded_duration, self.af_currsec, pinkred("{:.2f}".format(confidence)), yellow(extract_filename(self.af_fullpath)), pinkred(predicted_word)))

def CreateAsrService(model_path):
    """
    Factory function for AsrService class.
    """
    # ensure an instance is created only on first call
    if  _AsrService._instance is None:
        _AsrService._instance = _AsrService()
        _AsrService.modelType = extract_filename(model_path)[6:9] # from name: model_cnn_...
        try:
            print_info("|||||| Loading model " + quote_path(model_path) + "... ", end="")
            if _AsrService.modelType == "aen": # autoencoder
                _AsrService.model = Autoencoder.load_model(model_path)
            else:
                _AsrService.model = keras.models.load_model(model_path)
            print_info("[DONE (model loaded)]", quote_path(model_path))
        except Exception as e:
            print(pinkred("\nException caught while trying to load the model: " + quote_path(model_path)))
            print(pinkred("Exception message: ") + red(str(e)))
    return _AsrService._instance

if __name__ == "__main__":

    args = process_clargs()

    asr = CreateAsrService(args.model_path)
    
    print_info("\nPredicting with dataset view (labels):", asr.label_mapping)
    print_info("On files in:", args.inferdata_path)
    print_info(asr.inference_report_headers.format("Loaded Sec", "Con", "Filename", "Inference"))

    (_, _, afnames) = next(os.walk(args.inferdata_path))
    
    START = args.inferdata_range[0]; # of the range in -inferdata_path on which to do inference
    END   = args.inferdata_range[1]; # of the range in -inferdata_path on which to do inference

    # Process audio files starting from START until END
    for i, afname in enumerate(islice(afnames, START, END)):
        af_fullpath = os.path.join(args.inferdata_path, afname)
        asr.load_audiofile(af_fullpath, args.load_duration)
        if len(asr.af_signal) < args.sample_rate: # process only signals of at least 1 sec
            print_info("skipped a short (< 1s) signal")
            continue
        for i in range(int(asr.af_loaded_duration)): # signumerize and infer on each second of the loaded file
            signums = asr.signumerize(signum_type=args.signum_type, startsec=i, n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop_length)
            w, c    = asr.predict(signums)
            asr.report(w, c, args.confidence_threshold)