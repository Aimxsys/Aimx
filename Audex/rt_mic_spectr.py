#!/usr/bin/env python

# Copyright (c) 2015-2020 Matthias Geier
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Show a text-mode spectrogram using live microphone data."""
import argparse
import shutil
import math

import numpy as np
import sounddevice as sd

import sys
import os
# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import int_or_str

usage_line = ' press <enter> to quit, +<enter> or -<enter> to change scaling '

try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')

args, remaining = parser.parse_known_args()

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(description=__doc__ + '\n\nSupported keys:' + usage_line, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])

parser.add_argument('-block-duration', type=float, metavar='DURATION',      default=50,          help='Block size (default %(default)s milliseconds)')
parser.add_argument('-columns',        type=int,                            default=columns,     help='Width of spectrogram')
parser.add_argument('-device',         type=int_or_str,                                          help='Input device (numeric ID or substring)')
parser.add_argument('-gain',           type=float, default=10,                                   help='Initial gain factor (default %(default)s)')
parser.add_argument('-range', type=float, nargs=2, metavar=('LOW', 'HIGH'), default=[100, 2000], help='Frequency range (default %(default)s Hz)')

args = parser.parse_args(remaining)

low, high = args.range
if high <= low:
    parser.error('HIGH must be greater than LOW')

# Create a nice output gradient using ANSI escape sequences.
# Stolen from https://gist.github.com/maurisvh/df919538bcef391bc89f
colors = 30, 34, 35, 91, 93, 97
chars = ' :%#\t#%:'
gradient = []

for bg, fg in zip(colors, colors[1:]):
    for char in chars:
        if char == '\t':
            bg, fg = fg, bg
        else:
            gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))
try:
    samplerate = sd.query_devices(args.device, 'input')['default_samplerate']

    delta_f = (high - low) / (args.columns - 1)
    fftsize = math.ceil(samplerate / delta_f)
    low_bin = math.floor(low / delta_f)

    def spectrogram_callback(indata, frames, time, status):
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(args.columns, '#'), '\x1b[0m', sep='')
        if any(indata):
            magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
            magnitude *= args.gain / fftsize
            line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
                    for x in magnitude[low_bin:low_bin + args.columns])
            print(*line, sep='', end='\x1b[0m\n')
        else:
            print('no input')

    with sd.InputStream(device = args.device, channels = 1, callback = spectrogram_callback,
                        blocksize  = int(samplerate * args.block_duration / 1000),
                        samplerate = samplerate):
        while True:
            response = input()
            if response in ('', 'q', 'Q'):
                break
            for ch in response:
                if ch == '+':
                    args.gain *= 2
                elif ch == '-':
                    args.gain /= 2
                else:
                    print('\x1b[31;40m', usage_line.center(args.columns, '#'), '\x1b[0m', sep='')
                    break
except KeyboardInterrupt:
    parser.exit('Interrupted by user')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

# This is an independent section.
# This section will instead of spectrogram display perceived volume.
#def print_volume(indata, outdata, frames, time, status):
#    volume_norm = np.linalg.norm(indata) * 10
#    print ("|" * int(volume_norm))
#
#with sd.Stream(callback=print_volume):
#    sd.sleep(5_000) # milliseconds