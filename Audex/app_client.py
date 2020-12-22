#!/usr/bin/env python

import os
import sys
import argparse
import requests

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward.
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import *

DEFAULT_FLASK_APP_PORT = ":5000"

parser = argparse.ArgumentParser(description = 'This scrip launches an ASR client.')

parser.add_argument("-inferdata_path", type = Path, help='Path to the audio files on which model inference is to be tested.')
parser.add_argument("-server_endpoint", default = "http://127.0.0.1" + DEFAULT_FLASK_APP_PORT, type=str, help='Server URL.')
parser.add_argument("-server_view",     default = "/predict",  type=str, help='Server view.')
parser.add_argument("-example",         action  ='store_true',           help='Will show a working example on how to call the script.')

args = parser.parse_args()

print_script_start_preamble(nameofthis(__file__), vars(args))

########################## Command Argument Handling & Verification #######################

if args.example:
    print_info(nameofthis(__file__) + " -inferdata_path ../workdir/infer/signal_down_five_few")
    exit()

if provided(args.inferdata_path) and not args.inferdata_path.exists():
    raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.inferdata_path)))

###########################################################################################

if __name__ == "__main__":

    (_, _, afnames) = next(os.walk(args.inferdata_path))

    for afname in afnames:
        af_fullpath = os.path.join(args.inferdata_path, afname)
        
        with open(af_fullpath, "rb") as af:

            # package stuff to send and perform POST request
            files_payload = {"file": (af_fullpath, af, "audio/wav")}

            request_destination = args.server_endpoint + args.server_view

            # send the package
            print_info("Sending request to:", request_destination)
            print_info("Request contents:  ", files_payload)
            response      = requests.post(request_destination, files=files_payload)
            response_data = response.json()

            print_info("Response came back:", response_data["inference"])