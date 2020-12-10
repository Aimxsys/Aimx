#!/usr/bin/env python

import argparse
import requests

from utils_common import *

parser = argparse.ArgumentParser(description = 'This scrip launches an ASR client.')

parser.add_argument("-inferdata_path", type = Path, help='Path to the audio files on which model inference is to be tested.')
parser.add_argument("-server_url",  default = "http://127.0.0.1:5000", type=str, help='Server URL.')
parser.add_argument("-server_view", default = "/predict",              type=str, help='Server view.')

args = parser.parse_args()

########################## Command Argument Handling & Verification #######################

if provided(args.inferdata_path) and not args.inferdata_path.exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

###########################################################################################

print_script_start_preamble(nameofthis(__file__), vars(args))

if __name__ == "__main__":

    (_, _, filenames) = next(os.walk(args.inferdata_path))

    for filename in filenames:
        audiofile_fullpath = os.path.join(args.inferdata_path, filename)
        
        with open(audiofile_fullpath, "rb") as audio_file:

            # package stuff to send and perform POST request
            files_payload = {"file": (audiofile_fullpath, audio_file, "audio/wav")}

            request_destination = args.server_url + args.server_view

            # send the package
            print_info("Sending request to:", request_destination)
            print_info("Request contents:  ", files_payload)
            response      = requests.post(request_destination, files=files_payload)
            response_data = response.json()

            print_info("Response came back:", response_data["inference"])