#!/usr/bin/env python

import argparse
import librosa
import numpy as np


class CMentex(object):

    def __init__(self):
        self.args = None
        self.parser()

    def parser(self):

        parser = argparse.ArgumentParser(description="This utility creates new audio files by manipulating given audio.")
        parser.add_argument("-in", dest='input_file', default='../working_dir',
                            type=str, help="Relative path to the audio file.")
        parser.add_argument("-out", dest='output_path', default='../working_dir',
                            type=str, help="Relative path to the output directory.")
        self.args = parser.parse_args()

    def load_audio_file(self):
        input_length = 16000
        data = librosa.core.load(self.args.input_file)[0]  #, sr=16000
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data