#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import preprocessor.data_load as data_load
from extractor.signal_extractor import SignalExtractor, Function
import numpy as np
import copy


class Feature():
    def __init__(self, init_function, transforms, agg_fun):
        self.init_function = init_function
        self.transforms = transforms
        self.agg_fun = agg_fun

    def __eq__(self, other):
        return self.init_function == other.init_function \
               and self.agg_fun == other.agg_fun \
               and (self.transforms == other.transforms)

    def __repr__(self):
        return "Feature:\t{}.\t{}.\t{}".format(str(self.init_function), ";".join(map(str, self.transforms)), str(self.agg_fun))

class FeatureMaker():
    """Main class for feature extraction"""

    def __init__(self, data_path1, data_path0, batch_size=50):
        data_loader = data_load.LoadSignals();
        self.true_class_data = data_loader.load_signals(data_path1, 1)
        self.false_class_data = data_loader.load_signals(data_path0, 0)
        self.all_quality = ['NWP', 'mutual_info', 'corrcoef']
        init_0 = Function(lambda x : x, 'Empty function', 'init')
        init_1 = Function(data_load.median_filter, 'Median filter', 'init')
        init_2 = Function(data_load.high_filter, 'A.G. high freq filter', 'init')
        init_3 = Function(data_load.low_filter, 'A.G. low freq filter', 'init')
        init_4 = Function(data_load.a_g_filter, 'A.G. filter', 'init')
        init_5 = Function(data_load.simple_hf_filter, 'Simple high freq filter', 'init')
        init_6 = Function(data_load.simple_lf_filter, 'Simple  low freq filter', 'init')
        init_7 = Function(data_load.simple_hf_lf_filter, 'Simple high low freq filter', 'init')
        init_8 = Function(data_load.calc_fft, 'FFT from signal', 'init')
        init_9 = Function(data_load.calc_ifft, 'IFFT from signal', 'init')
        self.init_functions = [init_0, init_1, init_2, init_3, init_4, init_5, init_6, init_7, init_8, init_9]
        self.batch_size = batch_size

    def fit_features(self, features_num=10):
        print ("Start creating new_features")
        num = 0
        all_features = []
        while num < features_num:
            np.random.shuffle(self.true_class_data)
            np.random.shuffle(self.false_class_data)
            new_batch = copy.deepcopy(self.true_class_data[0:self.batch_size] + self.false_class_data[0:self.batch_size])
            new_init_fun = self.init_functions[np.random.randint(len(self.init_functions))]
            print (new_init_fun)
            new_batch = [i.change_sginal(new_init_fun) for i in new_batch]
            new_extractor = SignalExtractor(new_batch)
            new_quality = self.all_quality[np.random.randint(len(self.all_quality))]
            best_transform, best_agg, score = new_extractor.fit(new_quality)
            new_feature = Feature(new_init_fun, best_transform, best_agg)
            if new_feature not in all_features:
                num += 1
                all_features.append(new_feature)
                print ('Create feature number {} '.format(str(num)))
                print ('{} with {} score by {}'.format(str(new_feature), str(score), new_quality))
            else:
                print ('Create feature which has been already existed')
        return all_features

if __name__ == '__main__':
    path_1 = '/home/vsevolod/IBS_data/IBS_true'
    path_0 = '/home/vsevolod/IBS_data/IBS_false'
    test = FeatureMaker(path_1, path_0)
    test.fit_features()
