#!/usr/bin/env python

from termcolor import colored
from colorama  import init
from pathlib   import Path
from pathlib   import PurePath
from datetime  import datetime

import numpy       as np
import sounddevice as sd
import pprint
import json
import glob
import sys
import os

# On Windows, calling init() will filter ANSI escape sequences out of any text
# On other platforms, calling init() has no effect
init() # colorama

# Useful regexp to count total LOC.
# Tested in VS by running a regexp search with the
# string below on Python files in the entire solution:
# ^(?([^\r\n])\s)*[^\s+?/]+[^\n]*$

def play(signal, sr, message, continuemessage="Continue?", waitforanykey=True):
    print_info(message + "\n", cyan(np.around(signal, 2).T))
    sd.play(signal, sr)
    sd.wait() # this wait is not necessary for hearing the sound if waitforanykey == False
    if waitforanykey:
        input(yellow(continuemessage))

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def print_script_start_preamble(script_filename, args):
    print_info("========================================================================================")
    print_info("===========================  SCRIPT START STANDARD PREAMBLE  ===========================")
    print_info("===========================  RUNNING WITH THESE PARAMETERS:  ===========================")
    print_info(cyan(script_filename))
    pprint.pprint(args) #-----------------------------------------------------------
    print_info("==================================================================== {}".format(lightyellow(timestamp_now())))

def prompt_user_warning(warning_text, strictness='suggestion'):
    yes = {'yes','y', 'ye', ''}
    no  = {'no','n'}
    choice = input(cyan(warning_text)).lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        if strictness == 'yesorno':
            print(pinkred("Unable to proceed without a 'yes' or 'no' answer. Exiting..."))
            exit()
        return None

def exists(x):
    return x is not None

def provided(cmd_arg):
    return exists(cmd_arg)

def quote(s):
    return '\'' + str(s) + '\''

def dquote(s):
    return '\"' + str(s) + '\"'

def quote_path(path, frompoint='workdir'):
    s = str(path)
    return '\'' + cyansky(s[s.find(frompoint):]) + '\''

def get_all_dirnames_in(dir):
    return [extract_filename(f) for f in os.scandir(dir) if f.is_dir()]

def get_all_filenames_in(dir):
    return [extract_filename(f) for f in os.scandir(dir) if f.is_file()]

def timestamp_now(precision='seconds'):
    return datetime.now().isoformat(' ', precision)

def progress_bar(current, total):
    j = (current + 1) / total
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('█'*int(20*j), 100*j))
    sys.stdout.flush()

def extract_filename(fullpath):
    return PurePath(fullpath).stem

def extract_fileext(fullpath):
    return PurePath(fullpath).suffix

def nameofthis(fullpath):
    return os.path.basename(fullpath)

class Colors:
    PURPLE      = '\033[95m'
    BLUE        = '\033[94m'
    CYAN        = '\033[96m'
    GREENBRIGHT = '\033[92m'
    LIGHTYELLOW = '\033[93m'
    PINKRED     = '\033[91m'
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.PURPLE      = ''
        self.BLUE        = ''
        self.GREENBRIGHT = ''
        self.LIGHTYELLOW = ''
        self.PINKRED     = ''
        self.ENDC        = ''

# Forwarding string colorization functions using termcolor Python module
"""
Available text colors:
    red, green, yellow, blue, magenta, cyan, white.

Available text highlights:
    on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

Available attributes:
    bold, dark, underline, blink, reverse, concealed.

Example:
    colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
    colored('Hello, World!', 'green')
"""
def yellow(s):
    return colored(s, 'yellow')

def magenta(s):
    return colored(s, 'magenta')

def cyansky(s):
    return colored(s, 'cyan')

def white(s):
    return colored(s, 'white')

def red(s):
    return colored(s, 'red')

def green(s):
    return colored(s, 'green')

# Forwarding string colorization functions using ASCII escape sequences
def purple(s):
    return f"{Colors.PURPLE}" + str(s) + f"{Colors.ENDC}"

def blue(s):
    return f"{Colors.BLUE}" + str(s) + f"{Colors.ENDC}"

def cyan(s):
    return f"{Colors.CYAN}" + str(s) + f"{Colors.ENDC}"

def greenbright(s):
    return f"{Colors.GREENBRIGHT}" + str(s) + f"{Colors.ENDC}"

def lightyellow(s):
    return f"{Colors.LIGHTYELLOW}" + str(s) + f"{Colors.ENDC}"

def pinkred(s):
    return f"{Colors.PINKRED}" + str(s) + f"{Colors.ENDC}"

# Semantic print functions
def print_info(s, *args, verbose = True, **kwargs):
    if verbose:
        print(yellow(s), *args, **kwargs)

# print-function for debugging
def deprint(s, *args, **kwargs):
    print(pinkred(s), *args, **kwargs)

# print-function for debugging, columnizes
def decolprint(x, xname, n=15):
    left_align = "{:<" + str(n) + "}"
    deprint(left_align.format("{}".format(x)), xname)

################################ PROTOTYPE / UNTESTED / NON-PRODUCTION FUNCTIONS BELOW THIS LINE
################################ May likely be useful in the future.

def read_json_file(json_fullpath):
    with open(json_fullpath, "r") as file:
        print_info("|||||| Loading file " + quote_path(json_fullpath) + "... ", end="")
        objson = json.load(file)
        print_info("[DONE]")
        return objson

def write_json_file(json_fullpath, objson):
    with open(json_fullpath, 'w') as file:
        print_info("|||||| Writing file", quote_path(json_fullpath), "... ", end="")
        json.dump(objson, file, indent=4)
        print_info("[DONE]")

def update_json_file(json_fullpath, key, value):
    objson = read_json_file(json_fullpath)
    objson[key] = value
    write_json_file(json_fullpath, objson)

# TODO: This function seems to not always return as expected
# Currently disabled, but kept as it looks useful if perfected.
def get_most_recent_file_in_dir(traindata_path, ext):
    files = glob.iglob(traindata_path + '/*.' + ext)
    return max(files, key = os.path.getctime)