#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessor.data_load import LoadSignals
from optimization.optimize import SimpleGreedyOptimizer
import numpy as np
from scipy.signal import medfilt,  detrend


class QualityMeasure():
    """Class which measure quality of features"""

    def __init__(self, quality='NWP'):
        self.quality = quality

    def basic_quality(self, target, feature_vector):
        assert (len(target) == len(feature_vector))
        if self.quality == 'NWP':
            sort_data_p = np.array([x for (y,x) in sorted(zip(feature_vector, target), key=lambda x: x[0])])
            sort_data_n = np.array([x for (y,x) in sorted(zip(-1.0 * feature_vector, target), key=lambda x: x[0])])
            p_nwp = QualityMeasure.calc_nwp(sort_data_p)
            n_nwp = QualityMeasure.calc_nwp(sort_data_n)
            return min(n_nwp, p_nwp)
        if self.quality == 'corrcoef':
            return 1 - abs(np.corrcoef(target, feature_vector)[0][1])
        if self.quality == 'mutual_info':
            from sklearn.metrics import normalized_mutual_info_score
            mi = normalized_mutual_info_score(target, feature_vector)
            return mi
        return 'WRONG QUALITY NAME'

    @staticmethod
    def calc_nwp(sorted_vector):
        all_pairs = float( (len(sorted_vector) - 1) * len(sorted_vector)  ) / 2.0
        return sum([sum(sorted_vector[0:i] > sorted_vector[i]) for i in range(0, len(sorted_vector))]) / all_pairs


class Function():
    """Wraper class for funcions"""

    def __init__(self, func, description="Simple function", fun_type='transform'):
        self.func = func
        self.description = description
        self.func_type = fun_type

    def __call__(self, data):
        return self.func(data)

    def __repr__(self):
        return self.func_type + " function: " + self.description


class SignalExtractor():
    """Main class for feature extraction"""

    def __init__(self, data_path1, data_path0):
        data_loader = LoadSignals();
        self.true_class_data = data_loader.load_signals(data_path1, 1)
        self.false_class_data = data_loader.load_signals(data_path0, 0)
        self.all_data = self.true_class_data + self.false_class_data
        np.random.shuffle(self.all_data)
        self.target = [i.get_class() for i in self.all_data]

    def fit(self, quality='NWP'):
        print ("Start fitting Extractor")
        print ('Choosing quality ', quality)
        ag1 = Function(np.mean,  'mean function', 'agg')
        ag2 = Function(np.max, 'max function', 'agg')
        ag3 = Function(np.min, 'min funtion', 'agg')
        ag4 = Function(np.std, 'std function', 'agg')
        ag5 = Function(np.median, 'median function', 'agg')
        ag6 = Function(lambda x: np.max(x) + np.min(x), 'max + min function', 'agg')
        ag7 = Function(lambda x: len(np.unique(x)), 'len unique function', 'agg')
        ag8 = Function(lambda x: np.min(x) / (np.max(x) + 0.0001), 'min over max', 'agg')
        ag9 = Function(lambda x: np.mean(x) / (np.median(x) + 0.0001), 'mean over median', 'agg')
        ts1 = Function(lambda x: np.log(1.0 + np.abs(np.min(x)) + x), 'log function')
        ts2 = Function(np.abs, 'abs function')
        ts3 = Function(np.diff, 'diff 1 function')
        ts4 = Function(lambda x: np.diff(x, 2), 'diff 2 function')
        ts7 = Function(np.sort, 'sort function')
        ts8 = Function(lambda x: (x - np.min(x)) / (np.max(x) - np.mean(x)), 'min max scaller')
        ts9 = Function(medfilt, 'median filter')
        ts11 = Function(lambda x: (x - np.mean(x)) / (np.std(x)), 'standart scaller')
        ts12 = Function(detrend, 'remove linear trend')
        agg_functions= [ag1,ag2, ag3,ag4, ag5, ag6, ag7, ag8, ag9]
        trans_functions = [ts1, ts2, ts3, ts4, ts7, ts8, ts9, ts11, ts12]
        optimizer = SimpleGreedyOptimizer(trans_functions, agg_functions, 6, QualityMeasure(quality))
        return optimizer.fit(self.all_data)


if __name__ == '__main__':
    path_1 = '/home/vsevolod/IBS_data/IBS_true'
    path_0 = '/home/vsevolod/IBS_data/IBS_false'
    test = SignalExtractor(path_1, path_0)
    print (test.fit(quality='NWP'))


