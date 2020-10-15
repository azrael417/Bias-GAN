import numpy as np

#plotting
import matplotlib
matplotlib.use( "agg" )
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.patches as mlines
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import matplotlib.backends.backend_pdf
import matplotlib.font_manager as font_manager
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages


#plot the stuff
def visualize_prediction(filename, prediction, groundtruth, dpi=50):

    #rescale inputs by 256
    iprediction = np.array(prediction * 256).astype('uint8')
    igroundtruth = np.array(groundtruth * 256).astype('uint8')
    idiff = np.array((prediction - groundtruth) * 256).astype('uint8')

    #create plot grid
    numcols = 3
    numrows = prediction.shape[0]

    #create figure
    fig, axvec = plt.subplots(figsize=(18*numcols, 10*numrows), nrows=numrows, ncols=numcols)

    for idr in range(numrows):
        #store images for normalization per row:
        images = []
        
        #plot prediction
        images.append(axvec[idr, 0].imshow(iprediction[idr,:,:], cmap='nipy_spectral'))

        #plot ground truth
        images.append(axvec[idr, 1].imshow(igroundtruth[idr,:,:], cmap='nipy_spectral'))

        #plot difference
        images.append(axvec[idr, 2].imshow(idiff[idr,:,:], cmap='nipy_spectral'))

        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)
    
    plt.tight_layout()

    #save the stuff into buffer
    plt.savefig(filename, format='png', dpi=dpi)
    #close figure
    plt.close(fig)
