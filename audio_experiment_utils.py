import librosa
import pygame
import matplotlib.pyplot as pt
import scipy as sp
import numpy as np

def print_stats(signal_pack):
    sample_duration = 1/signal_pack[1][1]
    signal_duration = sample_duration * len(signal_pack[1][0])
    ft = np.fft.fft(signal_pack[1][0])
    print(f"{signal_pack[0]} signal shape:     {signal_pack[1][0].shape}")
    print(f"{signal_pack[0]} signal size:      {signal_pack[1][0].size}")
    print(f"{signal_pack[0]} sampling rate:    {signal_pack[1][1]}")
    print(f"{signal_pack[0]} sample duration: {sample_duration: .6f} seconds")
    print(f"{signal_pack[0]} signal duration: {signal_duration: .2f} seconds")
    print(f"{signal_pack[0]} FFT:              {ft[0]}")

def plot_signals_single_chart(signal_packs):
    pt.figure(figsize=(20, 12)).canvas.set_window_title("Signals")
    rows = len(signal_packs)
    for index, sigp in enumerate(signal_packs, start = 1):
        pt.subplot(rows, 1, index)
        librosa.display.waveplot(sigp[1][0], alpha = 0.5)
        pt.title(sigp[0])
        pt.ylim((-1, 1))

def plot_frequency_distribution(signal_pack, f_ratio=1):
    ft   = sp.fft.fft(signal_pack[1][0])
    magn = np.abs(ft)
    freq = np.linspace(0, signal_pack[1][1], len(magn))
    num_freq_bins = int(len(freq) * f_ratio)
    pt.figure(figsize = (18, 4)).canvas.set_window_title("Frequency Distribution")
    pt.title(signal_pack[0])
    pt.plot(freq[:num_freq_bins], magn[:num_freq_bins]) # magnitude spectrum
    pt.xlabel("Frequency (Hz)")
    pt.ylabel("Magnitude")

def plot_spectrogram(Y, signal_pack, hop_length, y_axis = "linear"):
    pt.figure(figsize = (15, 8)).canvas.set_window_title("Spectrogram")
    pt.title(signal_pack[0])
    librosa.display.specshow(Y, sr = signal_pack[1][1], hop_length = hop_length, x_axis = "time", y_axis = y_axis)
    pt.colorbar(format = "%+2f")