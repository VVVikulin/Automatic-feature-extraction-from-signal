#!/usr/bin/env python
 #-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import scipy.io.wavfile
from data.signal_data import Signal


class LoadSignals():
    """Simple class for data loading"""

    def __init__(self):
        """Empty contructor"""
        pass

    def load_signals(self, data_path, class_id):
        all_signals = []
        filenames = os.listdir(data_path)
        for filename in filenames:
            if filename.find('wav') == -1:
                print ("Skipped not .wav file {}".format(filename).encode('utf-8'))
                continue
            full_file_name = data_path +  '/' + filename
            signal_id = (int) (filename.split('-')[0])
            rate_data = scipy.io.wavfile.read(full_file_name)
            all_signals.append(Signal(signal_id, class_id, rate_data[1], rate_data[0]))
        print ('Load ', len(all_signals), ' signals from ', data_path)
        return all_signals


if __name__ == "__main__":
    test_load_singnal = LoadSignals()
    test_load_singnal.load_signals('/home/vsevolod/IBS_data/IBS_true', 2)
