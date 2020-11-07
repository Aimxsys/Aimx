import subprocess

from common_utils import *

# IMPORTANT: TODO: Note for later resolution from official Python documentation
# https://docs.python.org/2/library/subprocess.html#frequently-used-arguments:
# Warning Executing shell commands that incorporate unsanitized input from an untrusted
# source makes a program vulnerable to shell injection, a serious security flaw
# which can result in arbitrary command execution. For this reason, the use of shell=True
# is strongly discouraged in cases where the command string is constructed from external input

# Audex
subprocess.call(['audex.py', '-files_path', 'sounds/two'], shell=True)

print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT audex.py OK"))

# Data preprocessing
subprocess.call(['genre_preprocess.py', '-dataset_path', 'dataset_c10_f3'], shell=True)

print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_preprocess.py OK"))

# Genre classification
subprocess.call(['genre_classifier.py', '-data_path', 'most_recent_output', '-epochs', '5'], shell=True)

print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT genre_classifier.py OK"))

print(magenta("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT PIPELINE TEST OK"))