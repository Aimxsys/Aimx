import json
import os

from common_utils import *

DATA_PREPROCESS_RESULT_METADATA_FILENAME = "preprocess_result_meta.json"

def get_recent_preprocess_result_metadata():
    with open(DATA_PREPROCESS_RESULT_METADATA_FILENAME, "r") as file:
        print_info("\n|||||| Loading file " + cyansky(DATA_PREPROCESS_RESULT_METADATA_FILENAME) + "...", end="")
        data = json.load(file)
        print_info(" [DONE]")
    return data["most_recent_output"]