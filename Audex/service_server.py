import random
import os

from flask import Flask, request, jsonify
from service_wordetect import CreateWordetectService

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

	# get audio file from POST request and save it
	audio_file = request.files["file"]
	file_name  = str(random.randint(0, 100_000))
	audio_file.save(file_name)

	# instantiate keyword spotting service singleton and get prediction
	wds = CreateWordetectService()
	pred_word = wds.predict(file_name)

	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"keyword": pred_word}
	return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False)