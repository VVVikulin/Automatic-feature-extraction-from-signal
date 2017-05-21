#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from optimization.optimize import SimpleGreedyOptimizer
import numpy as np
from sklearn.feature_selection import chi2
from extractor.additional_func import distcorr
from minepy import MINE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

class QualityMeasure():
    """Class which measure quality of features"""

    def __init__(self, quality='NWP', feature_len=200):
        self.quality = quality
        self.random_feature = np.random.uniform(0, 1, feature_len)

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
            m = MINE()
            m.compute_score(target, feature_vector)
            return 1.0 - m.mic()
        if self.quality == 'chi2':
            return 1 - chi2(abs(feature_vector.reshape(len(feature_vector), 1)), target)[0][0]
        if self.quality == 'distcorr':
            return 1 - distcorr(target, feature_vector)
        if self.quality == 'distree':
            data = np.column_stack((feature_vector, self.random_feature))
            clf = DecisionTreeClassifier(max_depth=5,  random_state=0)
            clf.fit(data, target)
            return 1.0 - clf.feature_importances_[0]
        if self.quality == 'knnscore':
            errors = []
            clf = KNeighborsClassifier()
            data = np.array([feature_vector]).transpose()
            loo = LeaveOneOut()
            for train, test in loo.split(data):
                clf = KNeighborsClassifier()
                clf.fit(data[train], target[train])
                errors.append(accuracy_score(target[test], clf.predict(data[test])))
            return 1.0 - np.mean(errors)
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
        return self.func_type + ":" + self.description


class SignalExtractor():
    """Main class for feature extraction"""

    def __init__(self, all_data):
        #data_loader = LoadSignals();
        #self.true_class_data = data_loader.load_signals(data_path1, 1)
        #self.false_class_data = data_loader.load_signals(data_path0, 0)
        #self.all_data = self.true_class_data + self.false_class_data
        self.all_data = all_data
        #np.random.shuffle(self.all_data)
        self.target = [i.get_class() for i in self.all_data]

    def fit(self, quality='NWP'):
        print ("Start fitting Extractor")
        print ('Choosing quality ', quality)
        ag1 = Function(np.mean,  'mean', 'agg')
        ag2 = Function(np.max, 'max', 'agg')
        ag3 = Function(np.min, 'min', 'agg')
        ag4 = Function(np.std, 'std', 'agg')
        ag5 = Function(np.median, 'median', 'agg')
        ag7 = Function(lambda x: np.min(x) / (np.max(x) + 0.0001), 'min_over_max', 'agg')
        ag8 = Function(lambda x: np.min(x) + np.max(x), 'min_plus_max', 'agg')
        ag9 = Function(lambda x: np.mean(x) / (np.median(x) + 0.0001), 'mean_over_median', 'agg')
        ag9 = Function(lambda x: np.dot(np.arange(len(x)), x), 'signal_center', 'agg')


        ts1 = Function(lambda x: np.log(1.0 + np.abs(np.min(x)) + x), 'log_function')
        ts2 = Function(np.abs, 'abs')
        ts3 = Function(np.diff, 'diff1')
        ts4 = Function(lambda x: np.diff(x, 2), 'diff2')
        ts5 =  Function(lambda x: np.power(2.0, x / float(max(x))), 'pow2')
        ts6 =  Function(lambda x: np.power(x / float(max(x)), 2), 'squared ')
        ts7 =  Function(lambda x: np.power(np.abs(x), 0.5), 'square_root')
        ts8 = Function(lambda x: (x - np.min(x)) / (np.max(x) - np.mean(x)), 'min_max_scaller')
        ts9 = Function(lambda x: np.power(x + 0.0001, -1.0), 'inverse')
        ts10 = Function(lambda x: np.sin(x), 'sinus')
        ts11 = Function(lambda x: (x - np.mean(x)) / (np.std(x)), 'standart_scaller')
        agg_functions= [ag1,ag2, ag3,ag4, ag5, ag7, ag8, ag9]
        trans_functions = [ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10, ts11]
        optimizer = SimpleGreedyOptimizer(trans_functions, agg_functions, 8, QualityMeasure(quality))
        return optimizer.fit(self.all_data)


if __name__ == '__main__':
    path_1 = '/home/vsevolod/IBS_data/IBS_true'
    path_0 = '/home/vsevolod/IBS_data/IBS_false'
    test = SignalExtractor(path_1, path_0)
    print (test.fit(quality='NWP'))


