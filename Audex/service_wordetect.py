import librosa
import argparse
import tensorflow.keras as keras
import numpy as np

from common_utils import *
from audex_utils  import Aimx

# Calling with "-inferdata_path /to/file" will expect to find the file in ./to directory.
parser = argparse.ArgumentParser(description = 'This utility script allows you to experiment with'
                                               ' audio files and their various spectrograms.')

parser.add_argument("-inferdata_path",       type = Path,               help = 'Path to the audio files on which model inference is to be tested.')
parser.add_argument("-model_path",           type = Path,               help = 'Path to the model to be loaded.')
parser.add_argument("-highlight_confidence", default = 0.9, type=float, help = 'Highlight results if confidence is higher than this threshold.')

parser.add_argument("-n_mfcc",         default =    13, type=int, help = 'Number of MFCC to extract.')
parser.add_argument("-n_fft",          default =  2048, type=int, help = 'Length of the FFT window.   Measured in # of samples.')
parser.add_argument("-hop_length",     default =   512, type=int, help = 'Sliding window for the FFT. Measured in # of samples.')
parser.add_argument("-num_segments",   default =     5, type=int, help = 'Number of segments we want to divide sample tracks into.')
parser.add_argument("-sample_rate",    default = 22050, type=int, help = 'Sample rate at which to read the audio files.')
parser.add_argument("-track_duration", default =     1, type=int, help = 'Only load up to this much audio (in seconds).')

args = parser.parse_args()

############################## Command Argument Verification ##############################

if provided(args.inferdata_path) and not args.inferdata_path.exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

###########################################################################################

class _WordetectService:
    """
    Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """
    model = None
    _mapping = [
        "down"  ,
        "off"   ,
        "on"    ,
        "no"    ,
        "yes"   ,
        "stop"  ,
        "up"    ,
        "right" ,
        "left"  ,
        "go"
    ]
    _instance = None

    def predict(self, audio_file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """
        # extract MFCCs into an array: # (# segments, # coefficients)
        MFCCs = self.preprocess(audio_file_path, num_mfcc = args.n_mfcc,
                                                    n_fft = args.n_fft,
                                               hop_length = args.hop_length,
                                           track_duration = args.track_duration)

        # convert the 2d MFCC array into a 4d array to feed to the model for prediction:
        #            (# segments, # coefficients)
        # (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make a prediction and get the predicted label
        predictions = self.model.predict(MFCCs)
        confidence  = np.max(predictions)
        if (confidence > args.highlight_confidence):
            print(cyan(confidence), cyan(Path(audio_file_path).stem))
        else:
            print(pinkred(confidence), pinkred(Path(audio_file_path).stem))
        predicted_index   = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, audio_file_path, num_mfcc=13, n_fft=2048, hop_length=512, track_duration=1):
        """
        Extract MFCCs from audio file.
        :param  file_path (str): Path of audio file
        :param   num_mfcc (int): # of coefficients to extract
        :param      n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """
        # load audio file
        signal, sample_rate = librosa.load(audio_file_path, duration = track_duration)

        if len(signal) >= args.sample_rate:            
            signal = signal[:args.sample_rate] # resize the signal to ensure consistency of the lengths
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def WordetectService():
    """
    Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """
    # ensure an instance is created only the first time the factory function is called
    if  _WordetectService._instance is None:
        _WordetectService._instance = _WordetectService()
        _WordetectService.model = keras.models.load_model(args.model_path)
    return _WordetectService._instance

if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    wds  = WordetectService()

    (_, _, filenames) = next(os.walk(args.inferdata_path))
    for filename in filenames:
        file = os.path.join(args.inferdata_path, filename)
        word = wds.predict(file)
        print(word)