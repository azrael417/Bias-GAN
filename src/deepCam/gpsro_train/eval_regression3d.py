# Basics
import os
import sys
import numpy as np
import argparse as ap
import datetime as dt
import subprocess as sp

# Torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from utils import losses
from utils import parsing_helpers as ph
from data import gpsro_dataset as gpsro
from architecture.gpsro import deeplab3d as dxc
from utils import gpsro_visualizer as gp
from utils import gpsro_postprocessor as pp

#vis stuff
from PIL import Image


#dict helper for argparse
class StoreDictKeyPair(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

#main function
def main(pargs):

    #set seed
    seed = 333
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("Using GPUs")
        device = torch.device("cuda", 0)
        torch.cuda.manual_seed(seed)
    else:
        print("Using CPUs")
        device = torch.device("cpu")

    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # we need to convert the loss weights to float
    for key in pargs.loss_weights:
        pargs.loss_weights[key] = float(pargs.loss_weights[key])
            
    # Determine normalizer
    normalizer = None
    if pargs.layer_normalization == "batch_norm":
        normalizer = nn.BatchNorm3d
    elif pargs.layer_normalization == "instance_norm":
        normalizer = nn.InstanceNorm3d
    else:
        raise NotImplementedError("Error, " + pargs.layer_normalization + " not supported")

    # Define architecture
    n_input_channels = 1 + pargs.noise_dimensions
    n_output_channels = 1
    net = dxc.DeepLab3d(n_input = n_input_channels, n_output = n_output_channels, os = 16, 
                        upsampler_type = pargs.upsampler_type, pretrained = False, normalizer = normalizer)
    net.to(device)

    #select loss
    criterion = None
    if pargs.loss_type == "l1":
        if pargs.enable_masks:
            criterion = losses.L1LossWeighted(normalize=False)
        else:
            criterion = torch.nn.L1Loss()
    elif pargs.loss_type == "smooth_l1":
        if pargs.enable_masks:
            criterion = losses.L1LossWeighted(normalize=False, smooth=True)
        else:
            criterion = torch.nn.SmoothL1Loss()
    elif pargs.loss_type == "l2":
        if pargs.enable_masks:
            criterion = losses.L2LossWeighted(normalize=False)
        else:
            criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError("Error, {} loss not supported".format(pargs.loss_type))

    #noise vector
    dist = None
    if pargs.noise_dimensions > 0:
        if pargs.noise_type == "Uniform":
            dist = torch.distributions.uniform.Uniform(0., 1.)
        elif pargs.noise_type == "Normal":
            dist = torch.distributions.normal.Normal(0., 1.)
        else:
            raise NotImplementedError("Error, noise type {} not supported.".format(noise_type))
        
    #load net and optimizer states
    # reload model
    checkpoint = torch.load(pargs.checkpoint, map_location = device)
    model_dict = {}
    for k in checkpoint['model']:
        model_dict[k.replace("module.","")] = checkpoint['model'][k]
    net.load_state_dict(model_dict)
            
    # Set up the data feeder
    # validation
    validation_dir = os.path.join(root_dir, pargs.data_set)
    validation_set = gpsro.GPSRODataset(validation_dir,
                                        statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                        channels = pargs.channels,
                                        normalization_type = "MinMax" if pargs.noise_type == "Uniform" else "MeanVariance",
                                        shuffle = False,
                                        masks = pargs.enable_masks,
                                        shard_idx = 0, shard_num = 1,
                                        num_intra_threads = pargs.max_intra_threads,
                                        read_device = torch.device("cpu") if pargs.disable_gds else device,
                                        send_device = device)
    validation_loader = DataLoader(validation_set, pargs.local_batch_size, drop_last=True)

    # visualizer
    gpviz = gp.GPSROVisualizer(statsfile = os.path.join(root_dir, 'stats3d.npz'),
                               channels = pargs.channels,
                               normalize = True,
                               normalization_type = "MinMax" if pargs.noise_type == "Uniform" else "MeanVariance")
    
    # postprocessing
    pproc = pp.GPSROPostprocessor(statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                  channels = pargs.channels,
                                  normalization_type = "MinMax" if pargs.noise_type == "Uniform" else "MeanVariance")
        
    # eval network
    print('{:14.4f} REPORT: starting evaluation'.format(dt.datetime.now().timestamp()))
    net.eval()
        
    # vali loss
    mse_list_val = []

    # these we need for the R2-Value
    val_mean = 0.
    val_var = 0.
    pred_var = 0.
    
    # disable gradients
    with torch.no_grad():
        
        # iterate over validation sample
        for step_val, token_val in enumerate(validation_loader):
            
            if pargs.enable_masks:
                inputs_raw_val, label_val, masks_val, filename = token_val
                masks_val = torch.unsqueeze(masks_val, dim=1)
            else:
                inputs_raw_val, label_val, filename = token_val

            #unsqueeze
            inputs_raw_val = torch.unsqueeze(inputs_raw_val, dim=1)
            label_val = torch.unsqueeze(label_val, dim=1)

            # generate noise vector and concat with inputs
            if dist is not None:
                inputs_noise_val = dist.rsample( (inputs_raw_val.shape[0], pargs.noise_dimensions, inputs_raw_val.shape[1], inputs_raw_val.shape[2], inputs_raw_val.shape[3]) ).to(device)
                inputs_val = torch.cat((inputs_raw_val, inputs_noise_val), dim = 1)
            else:
                inputs_val = inputs_raw_val
                
            # forward pass
            outputs_val = net(inputs_val)
            
            # Compute loss and average across nodes
            if pargs.enable_masks:
                loss_dict = {"valid": criterion(outputs_val, label_val, masks_val),
                             "hole": criterion(outputs_val, label_val, 1. - masks_val)}
                loss_val = 0.
                for key in loss_dict:
                    loss_val += loss_dict[key] * pargs.loss_weights[key]
            else:
                loss_val = criterion(outputs_val, label_val)

            # append to list
            mse_list_val.append(loss_val)

            # compute quantities relevant for R2
            valarr = pproc.process(torch.squeeze(label_val.detach().cpu(), dim = 0).numpy())
            outarr = pproc.process(torch.squeeze(outputs_val.detach().cpu(), dim = 0).numpy())
            val_mean += np.mean(valarr)
            val_var += np.mean(np.square(valarr))
            pred_var += np.mean(np.square(outarr - valarr))

            for sample_idx in range(len(filename)):
                # get datestamp 
                datestamp = filename[sample_idx].replace(".npy","")

                # save numpy array
                outfilename = os.path.join(pargs.output_dir, pargs.output_prefix + "_" + datestamp + '.npy')
                np.save(outfilename, outarr[sample_idx, 0, ...])

                # visualize
                plotname = os.path.join(output_dir, "plot_evaluation_{}.png".format(datestamp))
                prediction_val = outputs_val.detach()[sample_idx, 0, ...].cpu().numpy()
                groundtruth_val = label_val.detach()[sample_idx, 0, ...].cpu().numpy()
                gpviz.visualize_prediction(plotname, prediction_val, groundtruth_val)
                
        # average the validation loss
        count_val_global = float(len(mse_list_val))
        loss_val_global = sum(mse_list_val)
        loss_val_avg = loss_val_global / count_val_global
        # computing R2 Value
        # divide by sample size:
        val_mean /= float(step_val + 1)
        val_var /= float(step_val + 1)
        pred_var /= float(step_val + 1)
        # finalize the variance
        val_var -= np.square(val_mean)
        # take the ratio
        val_r2 = 1. - pred_var / val_var
        
        # print results
        print('{:14.4f} REPORT evaluation: loss {}'.format(dt.datetime.now().timestamp(), loss_val_avg))
        print('{:14.4f} REPORT evaluation: R2 {}'.format(dt.datetime.now().timestamp(), val_r2))

            
    print('{:14.4f} REPORT: finishing evaluation'.format(dt.datetime.now().timestamp()))
    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable.", required=True)
    AP.add_argument("--output_prefix", type=str, help="Filename output for the predictions.", required=True)
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--checkpoint", type=str, help="Checkpoint file to use for evaluation.", required=True)
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--data_set", type=str, default='validation', choices=["validation", "test"], help="which set to pick")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_intra_threads", type=int, default=8, help="Maximum degree of parallelism within reader")
    AP.add_argument("--validation_visualization_frequency", type=int, default = 5, help="Frequency with which a random sample is visualized during validation")
    AP.add_argument("--channels", type=int, nargs='+', default=list(range(0,45)), help="Channels used in input")
    AP.add_argument("--enable_masks", action='store_true')
    AP.add_argument("--upsampler_type", type=str, default="Deconv", choices=["Interpolate", "Deconv", "Deconv1x"], help="Which upsampler to use")
    AP.add_argument("--layer_normalization", type=str, default="batch_norm", choices=["batch_norm", "instance_norm"], help="Layer normalization type")
    AP.add_argument("--noise_dimensions", type=int, default=1, help="Number of additional noise dimensions")
    AP.add_argument("--noise_type", type=str, default="Uniform", choices=["Uniform", "Normal"], help="Noise type")
    AP.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2", "smooth_l1"], help="Loss type")
    AP.add_argument("--loss_weights", action=StoreDictKeyPair)
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--disable_gds", action='store_true')
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
