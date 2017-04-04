import numpy as np

class Signal:
    """Class for signals"""

    def __init__(self, signal_id, class_id, data, sample_rate, patient_id=None):
        self.signal_id = signal_id
        self.class_id = class_id
        self.sample_rate = sample_rate
        self.data = data
        self.patient_id = patient_id
        self.transforms = []
        self.agg = np.mean

    def __str__(self):
        return "Signal with id {} and from class {}".format(self.signal_id, self.class_id)

    def get_feature(self):
        return {'data' : self.data, 'anything' : self.sample_rate}

    def get_class(self):
        return self.class_id

    def create_new_signal(self, transform):
        return Signal(self.signal_id, self.class_id, transform(self.data), self.sample_rate)

    def evaluate_ext(self, trans_func, agg_func):
        result = np.copy(self.data)
        for new_transform in trans_func:
            result = new_transform(result)
        return agg_func(result)

    def change_sginal(self, func):
        self.data = func(self.data)
        return self

if __name__ == "__main__":
    pass
