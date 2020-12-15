#!/usr/bin/env python

import argparse
import librosa

parser = argparse.ArgumentParser(description="This utility creates new audio files by manipulating given audio.")
parser.add_argument("-in", default='../working_dir', type=str, help="Relative path to the audio file directory.")
parser.add_argument("-out", default='../working_dir', type=str, help="Relative path to the output directory.")
args = parser.parse_args()
