# Basics
import os
import sys
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
from architecture.gpsro import deeplab3d as dxc
from utils import gpsro_visualizer as gp
from utils import gpsro_postprocessor as pp

#vis stuff
from PIL import Image

#apex
import torch.cuda.amp as amp
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

    # init communicator
    comm = distcomm()

    #set seed
    seed = 333 + 7 * comm.rank()
    
    # Some setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        comm.printr("Using GPUs",0)
        device = torch.device("cuda", comm.local_rank())
        torch.cuda.manual_seed(seed)
    else:
        comm.printr("Using CPUs",0)
        device = torch.device("cpu")

    # set benchmark mode
    torch.backends.cudnn.benchmark = True
        
    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    
    if comm.rank() == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    # we need to convert the loss weights to float
    for key in pargs.loss_weights:
        pargs.loss_weights[key] = float(pargs.loss_weights[key])
            
    # Setup WandB
    if (pargs.logging_frequency > 0) and (comm.rank() == 0):
        # get wandb api token
        with open("/certs/.wandbirc_gpsro") as f:
            wbtoken = f.readlines()[0].replace("\n","")
        # log in: that call can be blocking, it should be quick
        sp.call(["wandb", "login", wbtoken])
        
        #init db and get config
        resume_flag = pargs.run_tag if pargs.resume_logging else False
        wandb.init(project = 'GPSRO bias correction', name = pargs.run_tag, id = pargs.run_tag, resume = resume_flag)
        config = wandb.config
    
        #set general parameters
        config.root_dir = root_dir
        config.output_dir = pargs.output_dir
        config.max_steps = pargs.max_steps
        config.local_batch_size = pargs.local_batch_size
        config.num_workers = comm.size()
        config.channels = pargs.channels
        config.noise_dimensions = pargs.noise_dimensions
        config.upsampler_type = pargs.upsampler_type
        config.noise_type = pargs.noise_type
        config.optimizer = pargs.optimizer
        config.start_lr = pargs.start_lr
        config.adam_eps = pargs.adam_eps
        config.weight_decay = pargs.weight_decay
        config.loss_type = pargs.loss_type
        config.model_prefix = pargs.model_prefix
        config.amp_opt_level = pargs.amp_opt_level
        config.use_batchnorm = False if pargs.disable_batchnorm else True
        config.enable_masks = pargs.enable_masks

        # loss weights
        for key in pargs.loss_weights:
            config.update({"loss_weights_"+key: pargs.loss_weights[key]}, allow_val_change = True)
        
        # lr schedule if applicable
        for key in pargs.lr_schedule:
            config.update({"lr_schedule_"+key: pargs.lr_schedule[key]}, allow_val_change = True)

    # Determine normalizer
    normalizer = dxc.Identity if pargs.disable_batchnorm else nn.BatchNorm3d

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
            criterion = nn.L1Loss()
    elif config["loss_type"] == "smooth_l1":
        if pargs.enable_masks:
            criterion = losses.L1LossWeighted(normalize=False, smooth=True)
        else:
            criterion = nn.SmoothL1Loss()
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
    
    #select optimizer
    optimizer = ph.get_optimizer(net, pargs.optimizer, pargs.start_lr, pargs.adam_eps, pargs.weight_decay)

    # grad scaler
    gscaler = amp.GradScaler(enabled = pargs.enable_amp)
    
    #load net and optimizer states
    start_step, start_epoch = comm.init_training_state(net, optimizer, pargs.checkpoint, device)
    
    #make model distributed
    net = comm.DistributedModel(net)

    #select scheduler
    if pargs.lr_schedule:
        scheduler = ph.get_lr_schedule(pargs.start_lr, pargs.lr_schedule, optimizer, last_step = start_step)
        
    #wrap the optimizer
    optimizer = comm.DistributedOptimizer(optimizer,
                                          named_parameters=net.named_parameters(),
                                          compression_name = None,
                                          op_name = "average")

    # Set up the data feeder
    # train
    train_dir = os.path.join(root_dir, "train")
    train_set = gpsro.GPSRODataset(train_dir,
                                   statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                   channels = pargs.channels,
                                   normalization_type = "MinMax" if pargs.noise_type == "Uniform" else "MeanVariance",
                                   shuffle = True,
                                   masks = pargs.enable_masks,
                                   shard_idx = comm.rank(), shard_num = comm.size(),
                                   num_intra_threads = pargs.max_intra_threads,
                                   read_device = torch.device("cpu") if pargs.disable_gds else device,
                                   send_device = device)
    train_loader = DataLoader(train_set, pargs.local_batch_size, drop_last=True)
    
    # validation
    validation_dir = os.path.join(root_dir, "validation")
    validation_set = gpsro.GPSRODataset(validation_dir,
                                        statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                        channels = pargs.channels,
                                        normalization_type = "MinMax" if pargs.noise_type == "Uniform" else "MeanVariance",
                                        shuffle = True,
                                        masks = pargs.enable_masks,
                                        shard_idx = comm.rank(), shard_num = comm.size(),
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
        mse_list = []
        
        #for inputs_raw, labels, source in train_loader:
        for token in train_loader:
            
            if pargs.enable_masks:
                inputs_raw, label, masks, filename = token
                masks = torch.unsqueeze(masks, dim=1)
            else:
                inputs_raw, label, filename = token
                
            #unsqueeze
            inputs_raw = torch.unsqueeze(inputs_raw, dim=1)
            label = torch.unsqueeze(label, dim=1)

            if dist is not None:
                # generate noise vector and concat with inputs
                inputs_noise = dist.rsample( (inputs_raw.shape[0], pargs.noise_dimensions, inputs_raw.shape[1], inputs_raw.shape[2], inputs_raw.shape[3]) ).to(device)
                inputs = torch.cat((inputs_raw, inputs_noise), dim = 1)
            else:
                inputs = inputs_raw
            
            # forward pass
            with amp.autocast(enabled = pargs.enable_amp):
                outputs = net(inputs)
            
                # Compute loss and average across nodes
                if pargs.enable_masks:
                    loss_dict = {"valid": criterion(outputs, label, masks), "hole": criterion(outputs, label, 1. - masks)}
                    loss = 0.
                    for key in loss_dict:
                        loss += loss_dict[key] * pargs.loss_weights[key]
                else:
                    loss = criterion(outputs, label)

            # average loss    
            loss_avg = comm.metric_average(loss, "train_loss", device=device)
            mse_list.append(loss_avg)
            
            # Backprop
            optimizer.zero_grad()
            gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()

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
                prediction = outputs.detach()[sample_idx, 0, ...].cpu().numpy()
                groundtruth = label.detach()[sample_idx, 0, ...].cpu().numpy()
                gpviz.visualize_prediction(plotname, prediction, groundtruth)
                
                #log if requested
                if pargs.logging_frequency > 0:
                    img = Image.open(plotname)
                    wandb.log({"Train Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
            
            #log if requested
            if (pargs.logging_frequency > 0) and (step % pargs.logging_frequency == 0) and (comm.rank() == 0):
                wandb.log({"Training Loss": loss_avg}, step = step)
                wandb.log({"Current Learning Rate": current_lr}, step = step)
                
            # validation step if desired
            if (step % pargs.validation_frequency == 0):
                
                #eval
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
                        with amp.autocast(enabled = pargs.enable_amp):
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
                        outarr = pproc.process(torch.squeeze(outputs_val.cpu(), dim = 0).numpy())
                        valarr = pproc.process(torch.squeeze(label_val.cpu(), dim = 0).numpy())
                        val_mean += np.mean(valarr)
                        val_var += np.mean(np.square(valarr))
                        pred_var += np.mean(np.square(outarr - valarr))
                        
                        # visualize the last sample if requested
                        if (step_val % pargs.validation_visualization_frequency == 0) and (comm.rank() == 0):
                            sample_idx = np.random.randint(low=0, high=label_val.shape[0])
                            plotname = os.path.join(output_dir, "plot_validation_step{}_valstep{}_sampleid{}.png".format(step,step_val,sample_idx))
                            prediction_val = outputs_val.detach()[sample_idx, 0, ...].cpu().numpy()
                            groundtruth_val = label_val.detach()[sample_idx, 0, ...].cpu().numpy()
                            gpviz.visualize_prediction(plotname, prediction_val, groundtruth_val)
                            
                            #log if requested
                            if pargs.logging_frequency > 0:
                                img = Image.open(plotname)
                                wandb.log({"Validation Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
                
                # average the validation loss
                count_val = float(len(mse_list_val))
                count_val_global = comm.metric_average(count_val, "val_count", op_name="sum", device=device)
                loss_val = sum(mse_list_val)
                loss_val_global = comm.metric_average(loss_val, "val_loss", op_name="sum", device=device)
                loss_val_avg = loss_val_global / count_val_global

                # computing R2 Value
                # divide by sample size:
                val_mean_global = comm.metric_average(val_mean, "val_mean", op_name="sum", device=device)
                val_mean_avg = val_mean_global / count_val_global
                val_var_global = comm.metric_average(val_var, "val_var", op_name="sum", device=device)
                val_var_avg = val_var_global / count_val_global
                pred_var_global = comm.metric_average(pred_var, "pred_var", op_name="sum", device=device)
                pred_var_avg = pred_var_global / count_val_global
                # finalize the variance
                val_var_avg -= np.square(val_mean_avg)
                # take the ratio
                val_r2_avg = 1. - pred_var_avg / val_var_avg
                
                # print results
                comm.printr('{:14.4f} REPORT validation: step {} loss {}'.format(dt.datetime.now().timestamp(), step, loss_val_avg), 0)
                comm.printr('{:14.4f} REPORT validation: step {} R2 {}'.format(dt.datetime.now().timestamp(), step, val_r2_avg), 0)

                # log in wandb
                if (pargs.logging_frequency > 0) and (comm.rank() == 0):
                    wandb.log({"Validation Loss": loss_val_avg}, step=step)
                    wandb.log({"Validation R2": val_r2_avg}, step=step)
                    
                # set to train
                net.train()
            
            #save model if desired
            if (step % pargs.save_frequency == 0) and (comm.rank() == 0):
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
		}
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
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
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
    AP.add_argument("--enable_masks", action='store_true')
    AP.add_argument("--upsampler_type", type=str, default="Deconv", choices=["Interpolate", "Deconv", "Deconv1x"], help="Which upsampler to use")
    AP.add_argument("--noise_dimensions", type=int, default=1, help="Number of additional noise dimensions")
    AP.add_argument("--noise_type", type=str, default="Uniform", choices=["Uniform", "Normal"], help="Noise type")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"], help="Optimizer to use")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"], help="Loss type")
    AP.add_argument("--loss_weights", action=StoreDictKeyPair)
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--lr_decay_patience", type=int, default=3, help="Minimum number of steps used to wait before decreasing LR")
    AP.add_argument("--lr_decay_rate", type=float, default=0.25, help="LR decay factor")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--disable_gds", action='store_true')
    AP.add_argument("--disable_batchnorm", action='store_true')
    AP.add_argument("--enable_amp", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
