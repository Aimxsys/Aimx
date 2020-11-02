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

class Colors:
    PURPLE      = '\033[95m'
    BLUE        = '\033[94m'
    CYAN        = '\033[96m'
    GREEN       = '\033[92m'
    LIGHTYELLOW = '\033[93m'
    PINK        = '\033[91m'
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.PURPLE      = ''
        self.BLUE        = ''
        self.GREEN       = ''
        self.LIGHTYELLOW = ''
        self.PINK        = ''
        self.ENDC        = ''

# Forwarding colored print functions using ASCII escape sequences
def printP(s, *args, **kwargs):
    print(f"{Colors.PURPLE}" + s + f"{Colors.ENDC}", *args, **kwargs)

def printB(s, *args, **kwargs):
    print(f"{Colors.BLUE}" + s + f"{Colors.ENDC}", *args, **kwargs)

def printC(s, *args, **kwargs):
    print(f"{Colors.CYAN}" + s + f"{Colors.ENDC}", *args, **kwargs)

def printG(s, *args, **kwargs):
    print(f"{Colors.GREEN}" + s + f"{Colors.ENDC}", *args, **kwargs)

def printL(s, *args, **kwargs):
    print(f"{Colors.LIGHTYELLOW}" + s + f"{Colors.ENDC}", *args, **kwargs)

def printK(s, *args, **kwargs):
    print(f"{Colors.PINK}" + s + f"{Colors.ENDC}", *args, **kwargs)

# Forwarding colored print functions using termcolor Python module
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
def printy(s, *args, **kwargs):
    print(colored(s, 'yellow'), *args, **kwargs)

def printm(s, *args, **kwargs):
    print(colored(s, 'magenta'), *args, **kwargs)

def printc(s, *args, **kwargs):
    print(colored(s, 'cyan'), *args, **kwargs)

def printw(s, *args, **kwargs):
    print(colored(s, 'white'), *args, **kwargs)

def printr(s, *args, **kwargs):
    print(colored(s, 'red'), *args, **kwargs)

def printg(s, *args, **kwargs):
    print(colored(s, 'green'), *args, **kwargs)