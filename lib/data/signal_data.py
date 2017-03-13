class Signal:
    """Class for signals"""

    def __init__(self, signal_id, class_id, data, sample_rate, patient_id=None, max_trans_size=10):
        self.signal_id = signal_id
        self.class_id = class_id
        self.sample_rate = sample_rate
        self.data = data
        self.patient_id = patient_id

    def __str__(self):
        return "Signal with id {} and from class {}".format(self.signal_id, self.class_id)


if __name__ == "__main__":
    print ("ya tyt")
    test_signal = Signal(100, 1, 1000, None)
    print (test_signal)
