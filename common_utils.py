import glob
import os

# TODO: Move these to common utils module
def exists(x):
    return x is not None

def provided(cmd_arg):
    return cmd_arg is not None

def quote(me):
    return '\'' + str(me) + '\''

def get_most_recent_file_in_dir(data_json_path, ext):
    files = glob.glob(data_json_path + '/*.' + ext)
    return max(files, key = os.path.getctime)