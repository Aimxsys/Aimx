#!/usr/bin/env python

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

try:
    start_idx = 0

    sr = sd.query_devices(args.device, 'output')['default_samplerate']

    # outdata.shape == (1136, 1) where the first number is the blocksize argument in sd.InputStream() below.
    def audio_callback(outdata, frames, time, status): 
        if status:
            print(status, file=sys.stderr)
        global start_idx                                                     # For frames == 1136:
        t = (start_idx + np.arange(frames)) / sr                             # vector of shape (1136,)
        t = t.reshape(-1, 1)                                                 # matrix of shape (1136, 1)
        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t) # matrix of shape (1136, 1) <- A*sin(2*pi*f*t)
        start_idx += frames # prepare for the next batch of frames to render
        print_info("CPU utilization:", "{:.2f}".format(output_stream.cpu_load), end='\r')

    with sd.OutputStream(samplerate = sr,
                         blocksize  = None, # Number of frames passed to audio_callback(), i.e. granularity for a blocking r/w stream.
                                            # Default and special value 0 means audio_callback() will receive an optimal (and possibly
                                            # varying) number of frames based on host requirements and the requested latency settings.
                                            # Will deduce optimal size automatically, for example 1136
                         latency    = None,
                         device     = args.device,
                         channels   = 1,
                         callback   = audio_callback) as output_stream:
        print_info('####' * 20)
        print_info("Playing sine with device's default sample rate of: ", sr)
        print_info("Press 'Enter' to quit")
        print_info('####' * 20)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))