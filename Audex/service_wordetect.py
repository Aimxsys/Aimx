import librosa
import tensorflow.keras as keras
import numpy as np

from common_utils import *

SAVED_MODEL_PATH = "../workdir/gen_saved_models/model_cnn_e50_train_asr_cnn"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
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
        # extract MFCCs
        MFCCs = self.preprocess(audio_file_path) # (# segments, # coefficients)

        # convert the 2d MFCC array into a 4d array to feed to the model for prediction:
        #            (# segments, # coefficients)
        # (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make a prediction and get the predicted label
        predictions = self.model.predict(MFCCs)
        confidence  = np.max(predictions)
        if (confidence > 0.9):
            print(cyan(confidence), cyan(Path(audio_file_path).stem))
        else:
            print(pinkred(confidence), pinkred(Path(audio_file_path).stem))
        predicted_index   = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, audio_file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """
        Extract MFCCs from audio file.
        :param  file_path (str): Path of audio file
        :param   num_mfcc (int): # of coefficients to extract
        :param      n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """
        # load audio file
        signal, sample_rate = librosa.load(audio_file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:            
            signal = signal[:SAMPLES_TO_CONSIDER] # resize the signal to ensure consistency of the lengths
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():
    """
    Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """
    # ensure an instance is created only the first time the factory function is called
    if  _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss  = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    ARG_AUDIO_FILES_DIR = "../workdir/test"

    (_, _, filenames) = next(os.walk(ARG_AUDIO_FILES_DIR))
    for filename in filenames:
        file = os.path.join(ARG_AUDIO_FILES_DIR, filename)
        word = kss.predict(file)
        print(word)