import glob
import os
from termcolor import colored

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
def printy(s, **kwargs):
    print(colored(s, 'yellow'), **kwargs)

def printm(s, **kwargs):
    print(colored(s, 'magenta'), **kwargs)

def printc(s, **kwargs):
    print(colored(s, 'cyan'), **kwargs)

def printw(s, **kwargs):
    print(colored(s, 'white'), **kwargs)

def printr(s, **kwargs):
    print(colored(s, 'red'), **kwargs)