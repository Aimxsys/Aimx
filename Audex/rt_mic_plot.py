#!/usr/bin/env python

"""
Plot the live microphone signal(s) with matplotlib.
Matplotlib and NumPy have to be installed.
"""
import argparse
import queue
import sys

import tensorflow.keras as keras
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
from Audex.utils.utils_audex  import Aimx
from Audex.utils.utils_audex  import get_dataprep_result_meta
from Audex.utils.utils_audex  import get_training_result_meta
from Audex.utils.utils_audex  import get_actual_model_path

from Audex.service_asr_rt import *

def process_clargs():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class = argparse.RawDescriptionHelpFormatter)

    # ANN-related arguments
    parser.add_argument("-model_path", default=Aimx.MOST_RECENT_OUTPUT, type = Path, help = 'Path to the model to be loaded.')
    parser.add_argument("-inferdata_path",       type = Path,                        help = 'Path to the audio files on which model inference is to be tested.')
    parser.add_argument("-confidence_threshold", default = 0.9, type=float,          help = 'Highlight results if confidence is higher than this threshold.')
    
    parser.add_argument("-n_mfcc",     type=int, default = 13,   help = 'Number of MFCC to extract.')
    parser.add_argument("-n_fft",      type=int, default = 2048, help = 'Length of the FFT window.   Measured in # of samples.')
    parser.add_argument("-hop_length", type=int, default = 512,  help = 'Sliding window for the FFT. Measured in # of samples.')
    
    # Original, mic-related arguments
    parser.add_argument('-list_devices',    action='store_true',                                    help='Show the list of audio devices and exits')
    parser.add_argument('-channels',        type=int,   default=[1], nargs='*', metavar='CHANNEL',  help='Input channels to plot (default: the first)')
    parser.add_argument('-device',          type=int_or_str,                                        help='Input device (numeric ID or substring)')
    parser.add_argument('-duration_window', type=float, default=200,            metavar='DURATION', help='Visible time slot (default: %(default)s ms)')
    parser.add_argument('-interval',        type=float, default=30,                                 help='Minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument('-blocksize',       type=int,   default=0,                                  help='Block size (in samples)')
    parser.add_argument('-sample_rate',     type=int,                                               help='Sampling rate of audio device')
    parser.add_argument('-downsample',      type=int,   default=1,              metavar='N',        help='Display every Nth sample (default: %(default)s)')
    
    args = parser.parse_args()
    
    ########################## Command Argument Handling & Verification #######################
    
    if args.list_devices:
        print_info(sd.query_devices())
        exit()

    if any(c < 1 for c in args.channels):
        parser.error('Argument CHANNEL: must be >= 1')

    if args.sample_rate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.sample_rate = device_info['default_samplerate'] # 44100
        print_info("Device info:")
        pprint.pprint(device_info)

    args.model_path = get_actual_model_path(args.model_path)

    args.n_mfcc       = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["n_mfcc"]      if not provided(args.n_mfcc)     else args.n_mfcc
    args.n_fft        = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["n_fft"]       if not provided(args.n_fft)      else args.n_fft
    args.n_hop_length = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["hop_length"]  if not provided(args.hop_length) else args.hop_length
    args.blocksize    = get_training_result_meta()[Aimx.Dataprep.SIGNAL_NUMERIZATION_PARAMS]["sample_rate"] if not provided(args.blocksize)  else args.sample_rate
        
    ###########################################################################################
    
    print_script_start_preamble(nameofthis(__file__), vars(args))
    
    return args, parser

# This is the audio callback function which consumes, processes or generates audio data in response to requests from an active stream.
# When a stream is running (e.g. coming from a mic), PortAudio calls the stream callback periodically. The callback function is
# responsible for processing and filling input and output buffers, respectively. As usual, the PortAudio stream callback runs at
# very high or real-time priority. It is required to consistently meet its time deadlines. Do not allocate memory, access the file system,
# call library functions or call other functions from the stream callback that may block or take an unpredictable amount of time to complete.
# With the exception of property object cpu_load it is not permissible to call PortAudio API functions from within the stream callback.
def audio_callback(indata, frames, time, status):
    """ Called (from a separate thread) for each audio block of size blocksize.
    indata.shape == (1136, 1) where the first number is the blocksize argument in sd.InputStream() below.
    In order for a stream to maintain glitch-free operation the callback must consume and return audio data faster than it is recorded
    and/or played. PortAudio anticipates that each callback invocation may execute for a duration approaching the duration of 'frames'
    audio frames at the stream's sampling frequency. It is reasonable to expect to be able to utilise 70% or more of the available
    CPU time in the PortAudio callback. However, due to buffer size adaption and other factors, not all host APIs are able to guarantee
    audio stability under heavy CPU load with arbitrary fixed callback buffer sizes. When high callback CPU utilisation is required
    the most robust behavior can be achieved by using blocksize=0.
    """
    if status:
        print(status, file = sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy of size args.blocksize downsampled by args.downsample
    audio_signal = indata[::args.downsample, channel_mapping]
    audio_signals_plotq.put(audio_signal)
    do_asr(audio_signal)
    #print_info("CPU utilization:", "{:.2f}".format(input_stream.cpu_load), end='\r')

def do_asr(audio_signal):
    global previously_silence
    dB = int(np.average(librosa.core.amplitude_to_db(audio_signal)))
    if dB > -71:
        audio_signal_squeezed = np.squeeze(audio_signal)
        mfccs = asr.numerize(audio_signal_squeezed, args.sample_rate, n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop_length)
        w, c  = asr.predict(mfccs)
        if previously_silence: # show numerization params as a reminder
            print('') # just so next text starts from a new line
            decolprint(audio_signal.shape,          "audio_signal")          # (114, 1) while (22050, 1) in the static working ASR
            decolprint(audio_signal_squeezed.shape, "audio_signal_squeezed") # (114,)   while (22050,) in the static working ASR
            decolprint(mfccs.shape, "audio_signal_squeezed numerized into mfccs (transposed) with arg.sample_rate " + str(args.sample_rate)) # (1,  1, 13, 1) with default args, working ASR has (1, 44, 13, 1)
            previously_silence = False
        asr.report(w, c, args.confidence_threshold, str(dB) + " dB")
    else: # silence
        print(blue(int(np.average(librosa.core.amplitude_to_db(audio_signal)))), end="dB") # silence in dB
        previously_silence = True

def plotsound_callback(frame):
    """ This is called by matplotlib for each plot update.
    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.
    """
    global empty_queue_ticks
    # Plot mic data
    global plotdata
    while True:
        try:
            # Extract an audio_signal from audio_signals_plotq whose size varies
            # from 1 up to about 5 observed in Leo's original environment
            audio_signal = audio_signals_plotq.get_nowait() # (114, 1) with default args.downsample == 10
            print_info(" audio_signals_plotq GOT SIGNAL")
            empty_queue_ticks = 0
            #do_asr(audio_signal)
        except queue.Empty:
            # Empty queue just means no audio data to render,
            # that's ok, just break and move on to the next cycle
            print_info(empty_queue_ticks, end="")
            empty_queue_ticks += 1
            break
        shift    = len(audio_signal)
        plotdata = np.roll(plotdata, -shift, axis=0) # roll old chunk to make room for new; of shape (882, 1)
        try:
            decolprint(plotdata.shape, "plotdata containing current audio_signal\n")
            plotdata[-shift:, :] = audio_signal # broadcast audio_signal of shape (114, 1) into plotdata of shape (882, 1) with default args
        except ValueError as e:
            sys.exit(pinkred("Captured audio stream data chunk (audio_signal) does not fit into target array 'plotdata':\n   ") + repr(e))
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

try:
    args, parser = process_clargs()

    channel_mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1

    audio_signals_plotq = queue.Queue()
    empty_queue_ticks = 0

    # Original defaults:    200               44100                          10
    plotdata_len = int(args.duration_window * args.sample_rate / (1000 * args.downsample)) # original
    plotdata      = np.zeros((plotdata_len, len(args.channels))) # resulting shape (882, 1) with arg defaults

    fig, ax = pt.subplots()
    lines   = ax.plot(plotdata) # later used in update_plot_callback()
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels], loc='lower left', ncol = len(args.channels))

    # Adjust some plot parameters for a prettier realtime show
    ax.axis((0, len(plotdata), -1, 1)) # (-1, 1) amounts to suppressing sensitivity
    ax.set_yticks([0])                 # hides all horizontal grid lines on the canvas but the central one
    ax.yaxis.grid(True)                # shows the horizontal grid lines defined in the lines above
    #ax.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False) # hides all axis texts
    fig.tight_layout(pad=0)            # adjusts the plot padding

    print_info("|||||| Loading model " + quote_path(args.model_path) + "... ", end="")
    model     = keras.models.load_model(args.model_path)
    modelType = extract_filename(args.model_path)[6:9] # from name: model_cnn_...
    print_info("[DONE]")

    asr = CreateAsrServiceRT(args.model_path)
    asr.model.summary()
    
    print_info("\nPredicting with dataset view (labels):", asr.label_mapping)
    previously_silence = True

    with sd.InputStream(samplerate = args.sample_rate,
                        blocksize  = args.blocksize, # Number of frames passed to audio_callback(), i.e. granularity for a blocking r/w stream.
                                                     # Default and special value 0 means audio_callback() will receive an optimal (and possibly
                                                     # varying) number of frames based on host requirements and the requested latency settings.
                                                     # Will deduce optimal size automatically, for example 1136 with default args
                        latency    = None,
                        device     = args.device,
                        channels   = max(args.channels),                        
                        callback   = audio_callback) as input_stream:

        #animation = FuncAnimation(fig, plotsound_callback, interval = args.interval, blit=True)
        #pt.show()
        print_info('####' * 20)
        print_info("Processing audio stream with mic default sample rate of:", args.sample_rate)
        print_info("Model input shape (adjust the audio chunks accordingly):", get_training_result_meta()[Aimx.Training.INPUT_SHAPE])
        print_info("Press 'Enter' to quit")
        print_info('####' * 20)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))