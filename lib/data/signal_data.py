import numpy as np

class Signal:
    """Class for signals"""

    def __init__(self, signal_id, class_id, data, sample_rate, patient_id=None, max_trans_size=10):
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

    def change_transforms(self, transforms):
        self.transforms = transforms

    def evaluate_inn(self):
        result = self.data
        for new_transform in self.transforms:
            result = new_transform(result)
        return self.agg(result)

    def evaluate_ext(self, trans_func, agg_func):
        result = self.data
        for new_transform in trans_func:
            result = new_transform(result)
        return agg_func(result)


if __name__ == "__main__":
    print ("ya tyt")
    test_signal = Signal(100, 1, 1000, None)
    print (test_signal)
