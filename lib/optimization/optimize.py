#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
#from extractor.signal_extractor import QualityMeasure
from scipy.signal import medfilt
import numpy as np

class SimpleGreedyOptimizer():
    def __init__(self, trans_functions, agg_functions, max_size, qual_measure):
       self.trans_functions = trans_functions
       self.agg_functions = agg_functions
       self.max_size = max_size
       self.qual_measure = qual_measure

    def fit(self, all_signals):
        #Choose best agregate function
        all_target = np.array([i.get_class() for i in all_signals])
        #features = [i.get_feature() for i in all_signals]
        mean_signal = np.array([i.evaluate_ext([], np.mean) for i in all_signals])
        best_score = self.qual_measure.basic_quality(all_target, mean_signal, 'NWP')
        best_agg_fun = np.mean
        for new_agg in self.agg_functions:
            new_featute =  np.array([i.evaluate_ext([], new_agg) for i in all_signals])
            new_score = self.qual_measure.basic_quality(all_target, new_featute)
            print (new_score, new_agg)
            if new_score < best_score:
                best_score = new_score
                best_agg_fun = new_agg
        print ('Initial best score is ', best_score, best_agg_fun)
        all_transformations = []
        for k in range(0, self.max_size):
            founded = False
            for new_trans in self.trans_functions:
                new_feature = np.array([i.evaluate_ext(all_transformations + [new_trans], best_agg_fun) for i in all_signals])
                new_score = self.qual_measure.basic_quality(all_target, new_feature, 'NWP')
                print ('Try ', new_trans, ' score is ', new_score)
                if new_score < best_score:
                    print ('Founded better transform  ', new_trans)
                    print ('Better score is ', new_score)
                    founded = True
                    best_score = new_score
                    best_trans = new_trans
            if not founded:
                print ("Greedy alg has stopped on iteration ", k)
                break
            all_transformations += [best_trans]
        return all_transformations, best_agg_fun, best_score


if __name__ == "__main__":
    test_opt = SimpleGreedyOptimizer([np.abs, np.diff, np.sort, medfilt], [np.mean, np.max, np.min, np.std], 6)
    print (medfilt(np.array([1,2,3,1,2,3,0,4,3,-5,1,2,4,0])))
