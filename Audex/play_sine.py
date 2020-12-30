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

""" Play a sine signal. """
import argparse
import sys

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
from Audex.utils.utils_common import print_info

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-list_devices', action='store_true',             help='Show the list of audio devices and exits')
parser.add_argument('-frequency', nargs='?', type=float, default=500, help='Frequency in Hz (default: %(default)s)')
parser.add_argument('-device',               type=int_or_str,         help='Output device (numeric ID or substring)')
parser.add_argument('-amplitude',            type=float, default=0.2, help='Amplitude (default: %(default)s)')

args = parser.parse_args()

########################## Command Argument Handling & Verification #######################
    
if args.list_devices:
    print_info(sd.query_devices())
    exit()

###########################################################################################

start_idx = 0

try:
    sr = sd.query_devices(args.device, 'output')['default_samplerate']

    def audio_callback(outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        global start_idx                                                     # For frames == 1136:
        t = (start_idx + np.arange(frames)) / sr                             # vector of shape (1136,)
        t = t.reshape(-1, 1)                                                 # matrix of shape (1136, 1)
        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t) # matrix of shape (1136, 1)
        start_idx += frames

    with sd.OutputStream(device = args.device, channels=1, callback=audio_callback, samplerate = sr):
        print_info('####' * 20)
        print_info("Playing sine with device's default sample rate of: ", sr)
        print_info("Press 'Enter' to quit")
        print_info('####' * 20)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))