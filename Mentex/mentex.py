#!/usr/bin/env python

import argparse
import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt


class CMentex(object):

    def __init__(self):
        self.args = None
        self.parser()

    def parser(self):

        parser = argparse.ArgumentParser(description="This utility creates new audio files by manipulating given audio")
        parser.add_argument("-in", dest='input_file', default='../working_dir',
                            type=str, help="Relative path to the audio file.")
        parser.add_argument("-out", dest='output_path', default='../working_dir',
                            type=str, help="Relative path to the output directory.")
        self.args = parser.parse_args()

    def load_audio_file(self):

        # TODO: Hard coded length
        input_length = 16000
        data = librosa.core.load(self.args.input_file)[0]
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def plot_time_series(self, data):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()

    def run(self):
        data = self.load_audio_file()
        self.plot_time_series(data)
        ipd.Audio(data, rate=16000)


if __name__ == "__main__":
    mentex = CMentex()
    mentex.run()
