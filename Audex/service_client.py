import requests

from common_utils import *

SERVER_URL = "http://127.0.0.1:5000/predict"

# audio file we'd like to send for predicting keyword
AUDIO_FILE_PATH = "../workdir/test/ds_down_1.wav"

if __name__ == "__main__":

    # open files
    audio_file = open(AUDIO_FILE_PATH, "rb")

    # package stuff to send and perform POST request
    payload = {"file": (AUDIO_FILE_PATH, audio_file, "audio/wav")}

    # send the package
    response      = requests.post(SERVER_URL, files=payload)
    response_data = response.json()

    print_info("Predicted keyword: {}".format(response_data["pred_word"]))