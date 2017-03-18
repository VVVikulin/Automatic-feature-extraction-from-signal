#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessor.data_load import LoadSignals
from optimization.optimize import SimpleGreedyOptimizer
import numpy as np
from scipy.signal import medfilt

class QualityMeasure():
    """Class which measure quality of features"""

    def __init__(self):
        pass

    def basic_quality(self, target, feature_vector, quality='NWP'):
        sort_data_p = np.array([x for (y,x) in sorted(zip(feature_vector, target))])
        sort_data_n = np.array([x for (y,x) in sorted(zip(-1.0 * feature_vector, target))])
        if quality == 'NWP':
            p_nwp = QualityMeasure.calc_nwp(sort_data_p)
            n_nwp = QualityMeasure.calc_nwp(sort_data_n)
            return {'NWP' : min(n_nwp, p_nwp)}
        return 'WRONG QUALITY NAME'

    @staticmethod
    def calc_nwp(sorted_vector):
        all_pairs = float( (len(sorted_vector) - 1) * len(sorted_vector)  ) / 2.0
        return sum([sum(sorted_vector[0:i] > sorted_vector[i]) for i in range(0, len(sorted_vector))]) / all_pairs


class SignalExtractor():
    """Main class for feature extraction"""

    def __init__(self, data_path1, data_path0):
        data_loader = LoadSignals();
        self.true_class_data = data_loader.load_signals(data_path1, 1)
        self.false_class_data = data_loader.load_signals(data_path0, 0)
        self.target = np.array([1] * len(self.true_class_data) + [0] * len(self.false_class_data))
        self.all_data = self.true_class_data + self.false_class_data

    def fit(self, quality='NWP'):
        optimizer = SimpleGreedyOptimizer([np.abs, np.diff, np.sort, medfilt], [np.mean, np.max, np.min, np.std, np.median], 6, QualityMeasure())
        return optimizer.fit(self.all_data)


if __name__ == '__main__':
    path_1 = '/home/vsevolod/IBS_data/IBS_true_test'
    path_0 = '/home/vsevolod/IBS_data/IBS_false_test'
    test = SignalExtractor(path_1, path_0)
    print (test.fit())


