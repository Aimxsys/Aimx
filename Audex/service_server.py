import argparse
import random
import os

from pathlib           import Path
from flask             import Flask, request, jsonify
from service_wordetect import CreateWordetectService
from common_utils      import *
from audex_utils       import get_actual_model_path
from audex_utils       import WORKDIR

# Calling with "-inferdata_path /to/file" will expect to find the file in ./to directory.
parser = argparse.ArgumentParser(description = 'Inference service')

parser.add_argument("-model_path", type = Path, help = 'Path to the model to be loaded.')

args = parser.parse_args()

############################## Command Argument Handling & Verification ##############################

# ...

######################################################################################################

print_script_start_preamble(nameofthis(__file__), vars(args))

args.model_path = get_actual_model_path(args.model_path)

# instantiate flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Word detection endpoint
    :return (json): This endpoint returns a json file with the following format:
        {
            "keyword": "down"
        }
    """
    audiofile_localpath = os.path.join(WORKDIR, extract_filename(request.files["file"].filename))

    # get audio file from POST request and
    # save it locally for further processing
    audiofile_received = request.files["file"]
    audiofile_received.save(audiofile_localpath) # temporary local file, to be deleted later

    # instantiate keyword spotting service singleton and get prediction
    wds = CreateWordetectService(args.model_path)
    
    wds.load_audiofile(audiofile_localpath, track_duration=1)
    if len(wds.afile_signal) >= wds.afile_sample_rate: # process only signals of at least 1 sec
        mfccs = wds.dataprep()
        w, c  = wds.predict(mfccs)
        wds.highlight(w, c)

        # send back result as a json file
        result = {"pred_word": w}
    else:
        # send back result as a json file
        result = {"pred_word": pinkred("SERVER PROCESSING ERROR: Received audio file shorter than 1 second, must be at least 1 second.")}

    os.remove(audiofile_localpath) # delete the audio file that's no longer needed

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)