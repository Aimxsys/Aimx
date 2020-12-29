#!/usr/bin/env python

import argparse
import random
import sys
import os

from pathlib            import Path
from flask              import Flask, request, jsonify

# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.service_asr        import CreateAsrService
from Audex.utils.utils_common import *
from Audex.utils.utils_audex  import get_actual_model_path
from Audex.utils.utils_audex  import WORKDIR
from Audex.utils.utils_audex  import Aimx

# Calling with "-inferdata_path /to/file" will expect to find the file in ./to directory.
parser = argparse.ArgumentParser(description = 'Inference service')

parser.add_argument("-model_path", default=Aimx.MOST_RECENT_OUTPUT, type=Path, help='Path to the model to be loaded.')
parser.add_argument("-example", action='store_true',                           help='Show a working example on how to call the script.')

args = parser.parse_args()

############################## Command Argument Handling & Verification ##############################

if args.example:
    print_info(nameofthis(__file__))
    exit()

if provided(args.model_path) and not args.model_path.exists():
    if str(args.model_path) is not Aimx.MOST_RECENT_OUTPUT:
        raise FileNotFoundError("Directory " + quote(pinkred(os.getcwd())) + " does not contain requested path " + quote(pinkred(args.model_path)))

args.model_path = get_actual_model_path(args.model_path)

######################################################################################################

print_script_start_preamble(nameofthis(__file__), vars(args))

flask_app_server = Flask(__name__) # instantiate Flask app

@flask_app_server.route("/predict", methods=["POST"])
def predict():
    """
    Word detection endpoint
    :return (json): This endpoint returns a json file with the following format:
        {
            "inference": "down"
        }
    """
    local_temp_af_path = os.path.join(WORKDIR, extract_filename(request.files["file"].filename))

    # get audio file from POST request and
    # save it locally for further processing
    af_received = request.files["file"]
    af_received.save(local_temp_af_path)

    # instantiate keyword spotting service singleton and get prediction
    asr = CreateAsrService(args.model_path)
    
    asr.load_audiofile(local_temp_af_path, load_duration=1)
    if len(asr.af_signal) >= asr.af_sr: # process only signals of at least 1 sec
        mfccs = asr.numerize()
        w, c  = asr.predict(mfccs)
        asr.report(w, c)
                
        prediction = w
    else:
        prediction = pinkred("SERVER PROCESSING ERROR: Received audio file shorter than 1 second, must be at least 1 second.")
    
    response = {"inference": prediction}
    os.remove(local_temp_af_path) # delete the temporary audio file that's no longer needed
    return jsonify(response) # send back the result as a json file

if __name__ == "__main__":
    flask_app_server.run(debug=False)