import librosa
import argparse
import tensorflow.keras as keras
import numpy as np

from common_utils import *
from audex_utils  import Aimx
from audex_utils  import get_dataprep_result_meta
from audex_utils  import get_actual_model_path

if __name__ == "__main__":

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

    print_script_start_preamble(nameofthis(__file__), vars(args))

class _WordetectService:
    """
    Singleton class for word detecting inference with trained models.
    """
    model = None

    afile_fullpath    = None
    afile_signal      = None
    afile_sample_rate = None
    afile_duration    = None

    _instance = None

    label_mapping = get_dataprep_result_meta()[Aimx.Dataprep.DATASET_VIEW]

    def load_audiofile(self, audiofile_fullpath, track_duration):
        self.afile_fullpath = audiofile_fullpath
        self.afile_signal, self.afile_sample_rate = librosa.load(audiofile_fullpath, duration = track_duration)
        full_afile_signal, _                      = librosa.load(audiofile_fullpath)
        self.afile_duration = librosa.get_duration(full_afile_signal, self.afile_sample_rate)

    def dataprep(self, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        # Eextract mfccs from an audio file.
        :param     n_mfcc (int): # of coefficients to extract
        :param      n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return mfccs (ndarray): 2-d numpy array with MFCC data of shape (# time steps, # coefficients)
        """
        #mfccs = np.empty([n_mfcc, 44]) # TODO: Revisit this line later

         # resize the signal to ensure consistency of the lengths
        self.afile_signal = self.afile_signal[:args.sample_rate]

        mfccs = librosa.feature.mfcc(self.afile_signal,
                                     self.afile_sample_rate,
                                     n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # convert the 2d MFCC array into a 4d array to feed to the model for prediction:
        #            (# segments, # coefficients)
        # (# samples, # segments, # coefficients, # channels)
        mfccs = mfccs[np.newaxis, ..., np.newaxis]

        return mfccs.T

    def predict(self, mfccs):
        # make a prediction and get the predicted label and confidence
        predictions   = self.model.predict(mfccs)
        confidence    = np.max(predictions)
        predmax_index = np.argmax(predictions)
        pred_word     = self.label_mapping[predmax_index]

        return pred_word, confidence

    def highlight(self, predicted_word, confidence):
        if predicted_word in extract_filename(self.afile_fullpath):
            if confidence > args.highlight_confidence:
                print("{:.2f}".format(self.afile_duration), cyan("{:.2f}".format(confidence)), pinkred(extract_filename(self.afile_fullpath)), cyan(predicted_word))
            else:
                print("{:.2f}".format(self.afile_duration), pinkred("{:.2f}".format(confidence)), pinkred(extract_filename(self.afile_fullpath)), cyan(predicted_word))
        else:
            print("{:.2f}".format(self.afile_duration), pinkred("{:.2f}".format(confidence)), pinkred(extract_filename(self.afile_fullpath)), pinkred(predicted_word))

def CreateWordetectService():
    """
    Factory function for WordetectService class.
    """
    # ensure an instance is created only on first call
    if  _WordetectService._instance is None:
        _WordetectService._instance = _WordetectService()
        try:
            model_path = get_actual_model_path(args.model_path)
            print_info("|||||| Loading model " + quote(cyansky(model_path)) + "... ", end="")
            _WordetectService.model = keras.models.load_model(model_path)
            print_info("[DONE]")
        except Exception as e:
            print(pinkred("Exception caught: " + str(e)))
    return _WordetectService._instance

if __name__ == "__main__":

    wds = CreateWordetectService()

    print_info("\nPredicting with dataset view (labels):", wds.label_mapping)
    print_info("On files in:", args.inferdata_path)
    print_info("Len  Con  Filename  Inference")

    (_, _, filenames) = next(os.walk(args.inferdata_path))

    for filename in filenames:
        audiofile_fullpath = os.path.join(args.inferdata_path, filename)
        wds.load_audiofile(audiofile_fullpath, args.track_duration)
        if len(wds.afile_signal) >= args.sample_rate: # is cut to exact in dataprep()
            mfccs = wds.dataprep(args.n_mfcc, args.n_fft, args.hop_length)
            w, c  = wds.predict(mfccs)
            wds.highlight(w, c)