#!/usr/bin/env python
 #-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessor.data_load import LoadSignals


class SignalExtractor():
    def __init__(self, data_path1, data_path0):
        data_loader = LoadSignals();
        self.true_class_data = data_loader.load_signals(data_path1, 1)
        self.false_class_data = data_loader.load_signals(data_path0, 0)


if __name__ == '__main__':
    path_1 = '/home/vsevolod/IBS_data/IBS_true'
    path_0 = '/home/vsevolod/IBS_data/IBS_false'
    test = SignalExtractor(path_1, path_0)
