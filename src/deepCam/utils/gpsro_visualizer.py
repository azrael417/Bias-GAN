import numpy as np

#plotting
import matplotlib
matplotlib.use( "agg" )
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
import matplotlib.patches as mlines
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import matplotlib.backends.backend_pdf
import matplotlib.font_manager as font_manager
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

SMALL_SIZE = 30
MEDIUM_SIZE = 40
BIGGER_SIZE = 50

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#histogram class which supports updating
class Histogram(object):
    def __init__(self, nbins, value_range, num_bootstrap=100, seed=12345):
        self.nbins = nbins
        self.value_range = value_range
        self.bin_edges = None
        self.counts = None
        self.total_count = 0
        self.num_bootstrap = num_bootstrap
        self.rng = np.random.RandomState(seed)
	
    def update(self, arr):
        if self.bin_edges is not None:
            new_counts, _ = np.histogram(arr, bins=self.bin_edges)
            self.counts	= np.concatenate([self.counts, np.expand_dims(new_counts, axis=0)], axis=0)
            self.total_count += np.sum(new_counts)
        else:
            self.counts, self.bin_edges = np.histogram(arr, bins=self.nbins, range=self.value_range)
            self.counts = np.expand_dims(self.counts, axis=0)
            self.total_count = np.sum(counts)
            
    def get_hist(self, normalize = True):
        # get normalization
        norm = 1./float(self.total_count) if normalize else 1.
        # normalize the counts
        counts_norm = self.counts * norm
        # get mean
        self.counts_mean = np.sum(counts_norm, axis=0)
        # do bootstrap resampling of rows
        idx = self.rng.choice(counts_norm.shape[0], (self.num_bootstrap, counts_norm.shape[0]))
        counts_bs = np.sum(counts_norm[idx, :], axis=1)
        self.counts_std = np.std(counts_bs, axis=0)
        
        return self.counts_mean, self.bin_edges, self.counts_std

    
#plot the stuff
class GPSROVisualizer(object):

        
    def __init__(self, statsfile, channels, normalize=False, normalization_type="MinMax"):
        # load min max from the data
        stats = np.load(statsfile)
        # data norm
        if normalization_type == "MinMax":
            self.data_min = stats["data_minval"][channels]
            self.data_max = stats["data_maxval"][channels]
            self.data_scale = self.data_max - self.data_min
            self.data_offset = self.data_min
        else:
            self.data_offset = stats["data_mean"][channels]
            self.data_scale = 1. / np.sqrt( stats["data_sqmean"][channels] - np.square(self.data_offset) )
            
        # label norm
        if normalization_type == "MinMax":
            self.label_min = stats["label_minval"][channels]
            self.label_max = stats["label_maxval"][channels]
            self.label_scale = self.label_max - self.label_min
            self.label_offset = self.label_min
        else:
            self.label_offset = stats["label_mean"][channels]
            self.label_scale = 1. / np.sqrt( stats["label_sqmean"][channels] - np.square(self.label_offset) )
        
        #do we normalize?
        self.normalize = normalize
        #self.data_range = (0., 1.)
        
    
    def visualize_prediction(self, filename, prediction, groundtruth, mask=None, dpi=50):

        #plotrows = list(range(iprediction.shape[0]))
        plotrows = [5, 25]
        
        #rescale inputs by 256
        iprediction = np.array(prediction)
        igroundtruth = np.array(groundtruth)

        #create plot grid
        numcols = 3
        numrows = len(plotrows)

        #create figure
        fig, axvec = plt.subplots(figsize=(15*numcols, 12*numrows), nrows=numrows, ncols=numcols)

        for idx,idr in enumerate(plotrows):
            # store images for normalization per row:
            images = []

            # get binary mask
            if mask is not None:
                bmask = (mask[idr, ...] == 1.)
            # we set the masked pixel to -1. Since everything else is minmax normed
            # -1. should be a good choice
            mask_default = 2.
            
            # plot prediction
            pred = iprediction[idr, ...]
            if self.normalize:
                pred = (pred * self.label_scale[idr]) + self.label_offset[idr]
            if mask is not None:
                pred[ ~bmask ] = mask_default
            axvec[idx, 0].set_xlabel("Longitude")
            axvec[idx, 0].set_ylabel("Latitude")
            axvec[idx, 0].set_title("Predicted model error (level="+str(idr)+")")
            images.append(axvec[idx, 0].imshow(pred, extent=[0,360,-90,90], cmap='bwr', aspect='auto', origin="lower")) #, vmin=self.data_range[0], vmax=self.data_range[1]))

            # plot ground truth
            gt = igroundtruth[idr, ...]
            if self.normalize:
                gt = (gt * self.label_scale[idr]) + self.label_offset[idr]
            if mask is not None:
                gt[ ~bmask ] = mask_default
            axvec[idx, 1].set_xlabel("Longitude")
            axvec[idx, 1].set_title("Calculated model error (level="+str(idr)+")")
            images.append(axvec[idx, 1].imshow(gt, extent=[0,360,-90,90], cmap='bwr', aspect='auto', origin="lower")) #, vmin=self.data_range[0], vmax=self.data_range[1]))
            

            # normalize colors
            if self.normalize:
                vmin = -2
                vmax = 2
            else:
                # Find the min and max of all colors for use in setting the color scale.
                vmin = min(image.get_array().min() for image in images)
                vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin = vmin, vmax = vmax)
            for im in images:
                im.set_norm(norm)

            # set colorbars
            divider = make_axes_locatable(axvec[idx, 0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap='bwr'), ax=axvec[idx, 0], cax=cax)
            divider = make_axes_locatable(axvec[idx, 1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap='bwr'), ax=axvec[idx, 1], cax=cax)
                
            # plot CDF
            axvec[idx, 2].set_xlim(vmin, vmax)
            if mask is not None:
                fpred = pred[bmask].flatten()
            else:
                fpred = pred.flatten()
            num_samples = fpred.shape[0]
            y = np.array(range(num_samples))/float(num_samples)
            # prediction
            x = np.sort(fpred)
            axvec[idx, 2].plot(x, y, color="crimson", lw = 4, label='prediction')
            # groundtruth
            if mask is not None:
                fgt = gt[bmask].flatten()
            else:
                fgt = gt.flatten()
            x = np.sort(fgt)
            axvec[idx, 2].plot(x, y, color="dodgerblue", lw = 4, label='target')
            axvec[idx, 2].legend(loc='upper left', frameon = False)
            axvec[idx, 2].set_title("Pixel intensity CDF (level="+str(idr)+")")
            axvec[idx, 2].set_xlabel("Bias")
    
        plt.tight_layout()

        #save the stuff into buffer
        plt.savefig(filename, format='png', dpi=dpi)
        #close figure
        plt.close(fig)
