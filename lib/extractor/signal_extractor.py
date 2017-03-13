#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessor.data_load import LoadSignals
import numpy as np

class SignalExtractor():

    def __init__(self, data_path1, data_path0):
        data_loader = LoadSignals();
        self.true_class_data = data_loader.load_signals(data_path1, 1)
        self.false_class_data = data_loader.load_signals(data_path0, 0)
        self.target = np.array([1] * len(self.true_class_data) + [0] * self.false_class_data)
        self.all_data = np.array(self.true_class_data + self.false_class_data)

    def calc_quality(self, feature_vector, quality='NWP'):
        #sort_data_p = np.array([x for (y,x) in sorted(zip(feature_vector, self.target))])
        #sort_data_n = np.array([x for (y,x) in sorted(zip(-1.0 * feature_vector, self.target))])
        pass

    @staticmethod
    def calc_nwp(sorted_vector):
        #all_pairs = float( (len(sorted_vector) - 1) * len(sorted_vector)  ) / 2.0
        return sum([sum(sorted_vector[0:i] > sorted_vector[i]) for i in range(0, len(sorted_vector))]) / 1.0


if __name__ == '__main__':
    path_1 = '/home/vsevolod/IBS_data/IBS_true'
    path_0 = '/home/vsevolod/IBS_data/IBS_false'
    #test = SignalExtractor('/home/vsevolod/IBS_data', '/home/vsevolod/IBS_data')
    print (SignalExtractor.calc_nwp(np.array([1,1,1,0])))
