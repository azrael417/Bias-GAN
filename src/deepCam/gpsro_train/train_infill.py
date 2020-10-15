# Basics
import sys
import os
import wandb
import numpy as np
import argparse as ap
import datetime as dt
import subprocess as sp

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from utils import losses
from utils import parsing_helpers as ph
from data import gpsro_dataset as gpsro
from architecture.gpsro import infill as dxi
from utils import gpsro_visualizer as gp

#vis stuff
from PIL import Image

#apex
from apex import amp
import apex.optimizers as aoptim

#horovod
#import horovod.torch as hvd
from comm.distributed import comm as distcomm


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

    
        
    # Train network
    if (pargs.logging_frequency > 0) and (comm.rank() == 0):
        wandb.watch(net)
    comm.printr('{:14.4f} REPORT: starting training'.format(dt.datetime.now().timestamp()), 0)
    step = start_step
    epoch = start_epoch
    current_lr = pargs.start_lr if not pargs.lr_schedule else scheduler.get_last_lr()[0]
    net.train()
    while True:
        
        comm.printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
        loss_list = []
        
        #for inputs_raw, labels, source in train_loader:
        for inputs_raw, label, masks, filename in train_loader:

            if dist is not None:
                # generate noise vector and concat with inputs
                inputs_noise = dist.rsample( (inputs_raw.shape[0], pargs.noise_dimensions, inputs_raw.shape[2], inputs_raw.shape[3]) ).to(device)
                inputs = torch.cat((inputs_raw, inputs_noise), dim = 1)
            else:
                inputs = inputs_raw
            
            # forward pass
            outputs, _ = net(inputs, masks)
            
            # Compute loss and average across nodes
            loss_dict = criterion(inputs_raw, outputs, label, masks)
            loss_dict_avg = {}
            loss = 0.
            for key in loss_dict:
                loss += loss_dict[key] * pargs.loss_weights[key]
                loss_dict_avg[key] = comm.metric_average(loss_dict[key], "train_loss_" + key, device=device)
            loss_avg = comm.metric_average(loss, "train_loss", device=device)
            loss_list.append(loss_avg)
            
            # Backprop
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            #step counter
            step += 1

            if pargs.lr_schedule:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

            #print some metrics
            comm.printr('{:14.4f} REPORT training: step {} loss {} LR {}'.format(dt.datetime.now().timestamp(), step, loss_avg, current_lr), 0)

            #visualize if requested
            if (step % pargs.training_visualization_frequency == 0) and (comm.rank() == 0):
                sample_idx = np.random.randint(low=0, high=label.shape[0])
                plotname = os.path.join(output_dir, "plot_train_step{}_sampleid{}.png".format(step,sample_idx))
                prediction = outputs.detach()[sample_idx,...].cpu().numpy()
                groundtruth = label.detach()[sample_idx,...].cpu().numpy()
                gpviz.visualize_prediction(plotname, prediction, groundtruth)
                
                #log if requested
                if pargs.logging_frequency > 0:
                    img = Image.open(plotname)
                    wandb.log({"Train Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
            
            #log if requested
            if (pargs.logging_frequency > 0) and (step % pargs.logging_frequency == 0) and (comm.rank() == 0):
                wandb.log({"Training Loss total": loss_avg}, step = step)
                for key in loss_dict_avg:
                    wandb.log({"Training Loss " + key : loss_dict_avg[key]}, step = step)
                wandb.log({"Current Learning Rate": current_lr}, step = step)
                
            # validation step if desired
            if (step % pargs.validation_frequency == 0):
                
                #eval
                net.eval()
                
                # vali loss
                loss_list_val = []

                # disable gradients
                with torch.no_grad():
                
                    # iterate over validation sample
                    step_val = 0
                    for inputs_raw_val, label_val, masks_val, filename in validation_loader:

                        # generate noise vector and concat with inputs
                        if pargs.noise_dimensions > 0:
                            inputs_noise_val = dist.rsample( (inputs_raw_val.shape[0], pargs.noise_dimensions, inputs_raw_val.shape[2], inputs_raw_val.shape[3]) ).to(device)
                            inputs_val = torch.cat((inputs_raw_val, inputs_noise_val), dim = 1)
                        else:
                            inputs_val = inputs_raw_val
                        
                        # forward pass
                        outputs_val, _ = net(inputs_val, masks_val)

                        # Compute loss and average across nodes
                        loss_dict = criterion(inputs_raw_val, outputs_val, label_val, masks_val)
                        loss_val = 0.
                        for key in loss_dict:
                            loss_val += loss_dict[key] * pargs.loss_weights[key]

                        # append to list
                        loss_list_val.append(loss_val)
                        
                        # visualize the last sample if requested
                        if (step_val % pargs.validation_visualization_frequency == 0) and (comm.rank() == 0):
                            sample_idx = np.random.randint(low=0, high=label_val.shape[0])
                            plotname = os.path.join(output_dir, "plot_validation_step{}_valstep{}_sampleid{}.png".format(step,step_val,sample_idx))
                            prediction_val = outputs_val.detach()[sample_idx,...].cpu().numpy()
                            groundtruth_val = label_val.detach()[sample_idx,...].cpu().numpy()
                            gpviz.visualize_prediction(plotname, prediction_val, groundtruth_val)
                            
                            #log if requested
                            if pargs.logging_frequency > 0:
                                img = Image.open(plotname)
                                wandb.log({"Validation Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
                
                # average the validation loss
                count_val = float(len(loss_list_val))
                count_val_global = comm.metric_average(count_val, "val_count", op_name="sum", device=device)
                loss_val = sum(loss_list_val)
                loss_val_global = comm.metric_average(loss_val, "val_loss", op_name="sum", device=device)
                loss_val_avg = loss_val_global / count_val_global
                
                # print results
                comm.printr('{:14.4f} REPORT validation: step {} loss {}'.format(dt.datetime.now().timestamp(), step, loss_val_avg), 0)

                # log in wandb
                if (pargs.logging_frequency > 0) and (comm.rank() == 0):
                    wandb.log({"Validation Loss Total": loss_val_avg}, step=step)

                # set to train
                net.train()
            
            #save model if desired
            if (step % pargs.save_frequency == 0) and (comm.rank() == 0):
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()}
                torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_step_" + str(step) + ".cpt") )

            #are we done?
            if step >= pargs.max_steps:
                break
                
        #do some after-epoch prep, just for the books
        epoch += 1
        if comm.rank()==0:
          
            # Save the model
            checkpoint = {
                'step': step,
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, os.path.join(output_dir, pargs.model_prefix + "_epoch_" + str(epoch) + ".cpt") )

        #are we done?
        if step >= pargs.max_steps:
            break

    comm.printr('{:14.4f} REPORT: finishing training'.format(dt.datetime.now().timestamp()), 0)
    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default="", help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_intra_threads", type=int, default=8, help="Maximum degree of parallelism within reader")
    AP.add_argument("--max_steps", type=int, default=25000, help="Maximum number of steps to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--validation_frequency", type=int, default=100, help="Frequency with which the model is validated")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--training_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during training")
    AP.add_argument("--validation_visualization_frequency", type=int, default = 5, help="Frequency with which a random sample is visualized during validation")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--channels", type=int, nargs='+', default=list(range(0,45)), help="Channels used in input")
    #AP.add_argument("--upsampler_type", type=str, default="Deconv", choices=["Interpolate", "Deconv", "Deconv1x"], help="Which upsampler to use")
    AP.add_argument("--noise_dimensions", type=int, default=0, help="Number of additional noise dimensions")
    AP.add_argument("--noise_type", type=str, default="Uniform", choices=["Uniform", "Normal"], help="Noise type")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"], help="Optimizer to use")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_weights", action=StoreDictKeyPair)
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--lr_decay_patience", type=int, default=3, help="Minimum number of steps used to wait before decreasing LR")
    AP.add_argument("--lr_decay_rate", type=float, default=0.25, help="LR decay factor")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--disable_gds", action='store_true')
    AP.add_argument("--layer_normalization", type=str, default="batch_norm", choices=["batch_norm", "instance_norm"], help="Layer normalization type")
    AP.add_argument("--amp_opt_level", type=str, default="O0", help="AMP optimization level")
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
