#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import preprocessor.data_load as data_load
from extractor.signal_extractor import SignalExtractor, Function
import numpy as np
import copy
import time

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
        return "{} {} {}".format(str(self.init_function), ",".join(map(str, self.transforms)), str(self.agg_fun))

    def __call__(self, feature):
        new_data = copy.deepcopy(feature.data)
        new_data = self.init_function(new_data)
        for new_trans in self.transforms:
            new_data = new_trans(new_data)
        return self.agg_fun(new_data)


class FeatureMaker():
    """Main class for feature extraction"""

    def __init__(self, data_path1, data_path0, batch_size=100, test_size=0.3):
        self.all_features = None
        data_loader = data_load.LoadSignals();
        self.true_class_data = data_loader.load_signals(data_path1, 1)
        self.false_class_data = data_loader.load_signals(data_path0, 0)
        train_size = 1 - test_size
        self.test_true_class = self.true_class_data[int(len(self.true_class_data) * train_size) : ]
        self.true_class_data = self.true_class_data[0:int(len(self.true_class_data) * train_size)]
        self.test_false_class = self.false_class_data[int(len(self.false_class_data) * train_size) : ]
        self.false_class_data = self.false_class_data[0:int(len(self.false_class_data) * train_size)]
        print ('Train true size is {} test size is {}'.format(str(len(self.true_class_data)), str(len(self.test_true_class)) ))
        print ('Train false size is {} test size is {}'.format(str(len(self.false_class_data)), str(len(self.test_false_class)) ))
        self.all_quality = ['distree', 'knnscore', 'mutual_info', 'corrcoef', 'NWP']
        init_0 = Function(lambda x : x, 'empty', 'init')
        init_1 = Function(data_load.median_filter, 'median_filter', 'init')
        init_2 = Function(data_load.high_filter, 'A.G. high_freq_filter', 'init')
        init_3 = Function(data_load.low_filter, 'A.G.low_freq_filter', 'init')
        init_4 = Function(data_load.a_g_filter, 'A.G.filter', 'init')
        init_5 = Function(data_load.simple_hf_filter, 'simple_high_freq_filter', 'init')
        init_6 = Function(data_load.simple_lf_filter, 'simple_low freq_filter', 'init')
        init_8 = Function(data_load.simple_hf_lf_filter, 'simple_high_low_freq_filter', 'init')
        init_8 = Function(data_load.calc_fft, 'FFT_from signal', 'init')
        init_9 = Function(data_load.calc_ifft, 'IFFT_from_signal', 'init')
        init_10 = Function(data_load.idx_peaks_detection, 'peak_idxs', 'init')
        init_11 = Function(data_load.value_peaks_detection, 'peak values', 'init')
        self.init_functions = [init_0, init_1, init_2, init_3, init_4, init_5, init_6, init_8, init_9, init_10, init_11]
        self.batch_size = batch_size

    def fit_features(self, features_num=100):
        print ("Start creating new_features")
        num = 0
        all_features = []
        start_time = time.time()
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
            f = open('result1.txt', 'a')
            f.write('%s\t%5d\t%s\t%.3f\n' % (new_quality, num, str(new_feature), (time.time() - start_time) / float(60)))
            f.close()
            #if new_feature not in all_features:
            num += 1
            all_features.append(new_feature)
            print ('Create feature number {} '.format(str(num)))
            print ('{} with {} score by {}'.format(str(new_feature), str(score), new_quality))
            #else:
            #    print ('Create feature which has been already existed')
        self.all_features = all_features
        return all_features

    def create_data(self):
        print ('Now start creating data set, please wait...')
        y_train = [0] * len(self.false_class_data) + [1] * len(self.true_class_data)
        X_train = []
        y_test = [0] * len(self.test_false_class) + [1] * len(self.test_true_class)
        X_test = []
        if self.all_features is None:
            print ('Firstly you need to call fit_features!')
            return None
        feature_num = 0
        for new_feature in self.all_features:
            feature_num += 1
            print ('Now creating feature num {}'.format(str(feature_num)))
            new_col = [new_feature(i) for i in self.false_class_data] + [new_feature(j) for j in self.true_class_data]
            X_train.append(new_col)
            new_test_col = [new_feature(i) for i in self.test_false_class] + [new_feature(j) for j in self.test_true_class]
            X_test.append(new_test_col)
        return np.array(X_train).T, np.array(y_train), np.array(X_test).T, np.array(y_test)


if __name__ == '__main__':
    path_1 = '/home/vsevolod/IBS_data/IBS_true'
    path_0 = '/home/vsevolod/IBS_data/IBS_false'
    test = FeatureMaker(path_1, path_0, batch_size=100, test_size=0.0)
    test.fit_features(features_num=300)
    X_train, y_train, X_test, y_test = test.create_data()
    np.save('X_train_big', X_train)
    np.save('y_train_big', y_train)
    #np.save('X_test', X_test)
    #np.save('y_test', y_test)
