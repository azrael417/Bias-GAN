import numpy as np
import torch

class moving_average(object):
    def __init__(self, window_size):
        self.window_size = window_size
        values=[]

    def append(self, value):
        values.append(value)
        if len(values) > self.window_size:
            values.pop(0)

    def mean(self):
        return mean(values)
        

def accuracy(prediction, label, threshold = 0.5):
    # prediction
    pred = torch.squeeze(prediction)
    pred = (pred > threshold).float()

    # label
    lab = torch.squeeze(label)
    
    # correct predictions
    correct = (pred == lab).sum()

    # accuracy
    accuracy = correct / float(prediction.shape[0])
    
    return accuracy
