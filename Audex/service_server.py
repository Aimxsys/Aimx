import argparse
import random
import os

from pathlib           import Path
from flask             import Flask, request, jsonify
from service_wordetect import CreateWordetectService
from common_utils      import *
from audex_utils       import get_actual_model_path

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
    audiofile_local_fullpath = str("temp_down_" + str(random.randint(0, 100_000)))

    # get audio file from POST request and save it
    audiofile_received = request.files["file"]
    audiofile_received.save(audiofile_local_fullpath) # temporary local file, to be deleted later

    # instantiate keyword spotting service singleton and get prediction
    wds = CreateWordetectService(args.model_path)
    
    wds.load_audiofile(audiofile_local_fullpath, track_duration=1)
    mfccs = wds.dataprep()
    w, c  = wds.predict(mfccs)
    wds.highlight(w, c)

    os.remove(audiofile_local_fullpath) # delete the audio file that's no longer needed

    # send back result as a json file
    result = {"pred_word": w}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False)