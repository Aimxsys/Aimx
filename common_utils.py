import glob
import os

# TODO: Move these to common utils module
def exists(x):
    return x is not None

def provided(cmd_arg):
    return cmd_arg is not None

def quote(me):
    return '\'' + str(me) + '\''

def mydir_most_recent_data(ext):
    files = glob.glob('./dataset*.' + ext)
    return max(files, key = os.path.getctime)