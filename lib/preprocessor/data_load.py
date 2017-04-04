#!/usr/bin/env python
 #-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import scipy.io.wavfile
from data.signal_data import Signal
import matplotlib.pyplot as plt
import seaborn
from scipy.fftpack import rfft, irfft, fftfreq
seaborn.set(style="white")
from scipy.signal import medfilt
class LoadSignals():
    """Simple class for data loading"""

    def __init__(self):
        """Empty contructor"""
        pass

    def load_signals(self, data_path, class_id, max_signals=100):
        all_signals = []
        filenames = os.listdir(data_path)
        num_signal = 0
        for filename in filenames:
            if filename.find('wav') == -1:
                print ("Skipped not .wav file {}".format(filename).encode('utf-8'))
                continue
            full_file_name = data_path +  '/' + filename
            signal_id = (int) (filename.split('-')[0])
            rate_data = scipy.io.wavfile.read(full_file_name)
            all_signals.append(Signal(signal_id, class_id, rate_data[1], rate_data[0]))
            num_signal += 1
            if num_signal ==  max_signals:
                break
        print ('Load ', len(all_signals), ' signals from ', data_path)
        return all_signals


def plot_signal(test_signal):
    plt.plot(test_signal)
    plt.show()

def calc_fft(data):
    return rfft(data)

def calc_ifft(data):
    return irfft(data)

def high_filter(data, sample_rate=1000):
    f_signal = rfft(data)
    l = int(len(f_signal)*50.0/sample_rate);
    cut_f_signal = f_signal.copy()
    cut_f_signal[l:len(f_signal)-1] = 0
    cut_signal = irfft(cut_f_signal)
    return cut_signal

def low_filter(data, sample_rate=1000):
    f_signal = rfft(data)
    l = int(len(f_signal)*5.0/sample_rate);
    cut_f_signal = f_signal.copy()
    cut_f_signal[0:l+1] = 0
    cut_f_signal[len(f_signal) + 1 - l :] = 0
    cut_signal = irfft(cut_f_signal)
    return cut_signal

def a_g_filter(data, sample_rate=1000):
    temp = high_filter(data,sample_rate)
    new =  low_filter(temp, sample_rate)
    return new.astype('int16')

def simple_lf_filter(signal, sample_rate=1000):
    W = fftfreq(signal.size, d=1/float(sample_rate))
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W<5)] = 0
    return irfft(cut_f_signal).astype('int16')


def simple_hf_filter(signal, sample_rate=1000):
    W = fftfreq(signal.size, d=1/float(sample_rate))
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W>70)] = 0
    return irfft(cut_f_signal).astype('int16')

def simple_hf_lf_filter(data, sample_rate=1000):
    temp = simple_hf_filter(data,sample_rate)
    new =  simple_lf_filter(temp, sample_rate)
    return new.astype('int16')

def median_filter(data):
    return medfilt(data, kernel_size=3)

if __name__ == "__main__":
    test_load_singnal = LoadSignals()
    all_signals = test_load_singnal.load_signals('/home/vsevolod/IBS_data/IBS_false', 2)
    test_signal = all_signals[3].get_feature()['data']
    sample_rate = all_signals[3].get_feature()['anything']
    plot_signal(test_signal)
    plot_signal(median_filter(test_signal))
