import os
import numpy as np

class GPSROPostprocessor(object):

    def __init__(self, statsfile, channels, tag="label", normalization_type="MinMax"):
        # load min max from the data
        stats = np.load(statsfile)
        # label norm
        if normalization_type == "MinMax":
            self.min = stats[tag + "_minval"][channels]
            self.max = stats[tag + "_maxval"][channels]
            self.scale = self.max - self.min
            self.offset = self.min
        else:
            self.offset = stats[tag + "_mean"][channels]
            self.scale = 1. / np.sqrt( stats[tag + "_sqmean"][channels] - np.square(self.offset) )
        # reshape
        self.offset = np.reshape(self.offset, (self.offset.shape[0], 1, 1))
        self.scale = np.reshape(self.scale, (self.scale.shape[0], 1, 1))

    def process(self, input):            
        return (input * self.scale) + self.offset
