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

"""
Plot the live microphone signal(s) with matplotlib.
Matplotlib and NumPy have to be installed.
"""
import argparse
import queue
import sys

from   matplotlib.animation import FuncAnimation
import matplotlib.pyplot as pt
import numpy as np
import sounddevice as sd

import sys
import os
# Add this directory to path so that package is recognized.
# Looks like a hack, but is ok for now to allow moving forward
# Source: https://stackoverflow.com/a/23891673/4973224
# TODO: Replace with the idiomatic way.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Audex.utils.utils_common import *

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true',  help='show list of audio devices and exit')

args, remaining = parser.parse_known_args()

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,  parents=[parser])

parser.add_argument('-channels',        type=int,   default=[1], nargs='*', metavar='CHANNEL',  help='Input channels to plot (default: the first)')
parser.add_argument('-device',          type=int_or_str,                                        help='Input device (numeric ID or substring)')
parser.add_argument('-duration_window', type=float, default=200,            metavar='DURATION', help='Visible time slot (default: %(default)s ms)')
parser.add_argument('-interval',        type=float, default=30,                                 help='Minimum time between plot updates (default: %(default)s ms)')
parser.add_argument('-blocksize',       type=int,                                               help='Block size (in samples)')
parser.add_argument('-samplerate',      type=float,                                             help='Sampling rate of audio device')
parser.add_argument('-downsample',      type=int,   default=10,             metavar='N',        help='Display every Nth sample (default: %(default)s)')

args = parser.parse_args(remaining)

print_script_start_preamble(nameofthis(__file__), vars(args))

if any(c < 1 for c in args.channels):
    parser.error('Argument CHANNEL: must be >= 1')

channel_mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """ Called (from a separate thread) for each audio block. """
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    audio_queue.put(indata[::args.downsample, channel_mapping]) # indata of shape (1136, 1)

def update_plot_callback(frame):
    """ This is called by matplotlib for each plot update.
    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.
    """
    global plotdata
    while True:
        try:
            audio_queue_data = audio_queue.get_nowait() # of shape (114, 1)
        except queue.Empty:
            break
        shift    = len(audio_queue_data)
        plotdata = np.roll(plotdata, -shift, axis=0) # of shape (882, 1)
        try:
            plotdata[-shift:, :] = audio_queue_data # broadcasting audio_queue_data of shape (114, 1) into plotdata shape (882, 1)
        except ValueError as e:
            sys.exit("Captured audio stream data does not fit into target array:\n   " + repr(e))
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

try:
    if args.samplerate is None:
        device_info     = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    plotdata_len = int(args.duration_window * args.samplerate / (1000 * args.downsample))
    plotdata     = np.zeros((plotdata_len, len(args.channels)))

    fig, ax = pt.subplots()
    lines   = ax.plot(plotdata) # later used in update_plot_callback()
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels], loc='lower left', ncol = len(args.channels))

    # Adjust some plot parameters for a prettier realtime show
    ax.axis((0, len(plotdata), -1, 1)) # (-1, 1) amounts to suppressing sensitivity
    ax.set_yticks([0])                 # hides all horizontal grid lines on the canvas but the central one
    ax.yaxis.grid(True)                # shows the horizontal grid lines defined in the lines above
    #ax.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False) # hides all (irrelevant) axis texts
    fig.tight_layout(pad=0)            # adjusts the plot padding

    input_stream = sd.InputStream(device     = args.device,
                                  #blocksize = 0, # Number of frames passed to audio_callback(), i.e. granularity for a blocking r/w stream.
                                                  # Default and special value 0 means audio_callback() will receive an optimal (and possibly
                                                  # varying) number of frames based on host requirements and the requested latency settings.
                                  channels   = max(args.channels),
                                  samplerate = args.samplerate,
                                  callback   = audio_callback)
    animation = FuncAnimation(fig, update_plot_callback, interval = args.interval, blit=True)
    with input_stream:
        pt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))