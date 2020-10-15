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
from utils import yparams as yp
from utils import parsing_helpers as ph
from data import gpsro_dataset as gpsro
from architecture.gpsro import deeplab as dxc
from utils import gpsro_visualizer as gp
from utils import gpsro_postprocessor as pp

#vis stuff
from PIL import Image

#apex
from apex import amp
import apex.optimizers as aoptim

#horovod
#import horovod.torch as hvd
from comm.distributed import comm as distcomm

hyperparameter_defaults = dict(
    max_intra_threads = 1,
    num_workers = 1,
    model_prefix = "regressor",
    amp_opt_level = "O1",
    checkpoint = "",
    enable_gds = False,
    save_frequency = 1000,
    logging_frequency = 50,
    validation_frequency = 100,
    training_visualization_frequency = 200,
    validation_visualization_frequency = 50,
    loss_weights = {"valid": 1., "hole": 0.8},
    channels = list(range(0,45)),
)

#main function
def main(pargs):

    # init communicator
    comm = distcomm("dummy")

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
        
    # get wandb api token
    with open("/certs/.wandbirc_gpsro") as f:
        wbtoken = f.readlines()[0].replace("\n","")
        # log in: that call can be blocking, it should be quick
        sp.call(["wandb", "login", wbtoken])

    # init wandb
    if pargs.tag_run:
        wandb.init(project = 'GPSRO bias correction', config = hyperparameter_defaults, name = pargs.run_tag, id = pargs.run_tag)
    else:
        wandb.init(project = 'GPSRO bias correction', config = hyperparameter_defaults)
    if pargs.config_file is None:
        config = wandb.config
    else:
        config = yp.yparams(pargs.config_file, "default").to_dict()
        if "channels" not in config:
            config["channels"] = list(range(0,45))
        
        # update wandb config
        wandb.config.update(config, allow_val_change=True)

    
    #set up directories
    root_dir = config["root_dir"]
    output_dir = config["output_dir"].format(run_tag=pargs.run_tag)
    
    if comm.rank() == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            
    # Determine normalizer
    normalizer = None
    if config["layer_normalization"] == "batch_norm":
        normalizer = torch.nn.BatchNorm2d
    elif config["layer_normalization"] == "instance_norm":
        normalizer = torch.nn.InstanceNorm2d
    else:
        raise NotImplementedError("Error, " + config["layer_normalization"] + " not supported")

    # Define architecture
    n_input_channels = len(config["channels"]) + config["noise_dimensions"]
    n_output_channels = len(config["channels"])
    net = dxc.DeepLabv3_plus(n_input = n_input_channels, n_output = n_output_channels, os = 16, 
                             upsampler_type = config["upsampler_type"], pretrained = False, normalizer = normalizer)
    net.to(device)

    #select loss
    criterion = None
    if config["loss_type"] == "l1":
        if config["enable_masks"]:
            criterion = losses.L1LossWeighted(normalize=False)
        else:
            criterion = torch.nn.L1Loss()
    elif config["loss_type"] == "smooth_l1":
        if config["enable_masks"]:
            criterion = losses.L1LossWeighted(normalize=False, smooth=True)
        else:
            criterion = torch.nn.SmoothL1Loss()
    elif config["loss_type"] == "l2":
        if config["enable_masks"]:
            criterion = losses.L2LossWeighted(normalize=False)
        else:
            criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError("Error, {} loss not supported".format(config["loss_type"]))

    #get loss weights dict
    loss_weights = None
    if "loss_weights.valid" in config.keys():
        loss_weights = {x.split(".")[1]: config[x] for x in config.keys() if x.startswith("loss_weights")}
    elif "loss_weights" in config.keys():
        loss_weights = config["loss_weights"]

    #noise vector
    dist = None
    if config["noise_dimensions"] > 0:
        if config["noise_type"] == "Uniform":
            dist = torch.distributions.uniform.Uniform(0., 1.)
        elif config["noise_type"] == "Normal":
            dist = torch.distributions.normal.Normal(0., 1.)
        else:
            raise NotImplementedError("Error, noise type {} not supported.".format(noise_type))
    
    #select optimizer
    optimizer = ph.get_optimizer(net, config["optimizer"], config["start_lr"], config["adam_eps"], config["weight_decay"])

    #wrap net and optimizer in amp
    net, optimizer = amp.initialize(net, optimizer, opt_level = config["amp_opt_level"])
    
    #load net and optimizer states
    start_step, start_epoch = comm.init_training_state(net, optimizer, config["checkpoint"], device)
    
    #make model distributed
    net = comm.DistributedModel(net)

    #select scheduler
    scheduler = None
    if "lr_schedule" in config.keys():
        scheduler = ph.get_lr_schedule(config["start_lr"], config["lr_schedule"], optimizer, last_step = start_step)
    elif "lr_schedule.type" in config.keys():
        scheddict = {x.split(".")[1]: config[x] for x in config.keys() if x.startswith("lr_schedule")}
        scheduler = ph.get_lr_schedule(config["start_lr"], scheddict, optimizer, last_step = start_step)
        
    #wrap the optimizer
    optimizer = comm.DistributedOptimizer(optimizer,
                                          named_parameters=net.named_parameters(),
                                          compression_name = None,
                                          op_name = "average")

    # Set up the data feeder
    # train
    train_dir = os.path.join(root_dir, "train")
    train_set = gpsro.GPSRODataset(train_dir,
                                   statsfile = os.path.join(root_dir, 'stats.npz'),
                                   channels = config["channels"],
                                   normalization_type = "MinMax" if config["noise_type"] == "Uniform" else "MeanVariance",
                                   shuffle = True,
                                   masks = config["enable_masks"],
                                   shard_idx = comm.rank(), shard_num = comm.size(),
                                   num_intra_threads = config["max_intra_threads"],
                                   read_device = torch.device("cpu") if not config["enable_gds"] else device,
                                   send_device = device)
    train_loader = DataLoader(train_set, config["local_batch_size"], drop_last=True)
    
    # validation
    validation_dir = os.path.join(root_dir, "validation")
    validation_set = gpsro.GPSRODataset(validation_dir,
                                        statsfile = os.path.join(root_dir, 'stats.npz'),
                                        channels = config["channels"],
                                        normalization_type = "MinMax" if config["noise_type"] == "Uniform" else "MeanVariance",
                                        shuffle = True,
                                        masks = config["enable_masks"],
                                        shard_idx = comm.rank(), shard_num = comm.size(),
                                        num_intra_threads = config["max_intra_threads"],
                                        read_device = torch.device("cpu") if not config["enable_gds"] else device,
                                        send_device = device)
    validation_loader = DataLoader(validation_set, config["local_batch_size"], drop_last=True)

    # visualizer
    gpviz = gp.GPSROVisualizer(statsfile = os.path.join(root_dir, 'stats.npz'),
                               channels = config["channels"],
                               normalize = True,
                               normalization_type = "MinMax" if config["noise_type"] == "Uniform" else "MeanVariance")

    # postprocessing
    pproc = pp.GPSROPostprocessor(statsfile = os.path.join(root_dir, 'stats.npz'),
                                  channels = config["channels"],
                                  normalization_type = "MinMax" if config["noise_type"] == "Uniform" else "MeanVariance")
        
    # Train network
    if (config["logging_frequency"] > 0) and (comm.rank() == 0):
        wandb.watch(net)
    comm.printr('{:14.4f} REPORT: starting training'.format(dt.datetime.now().timestamp()), 0)
    step = start_step
    epoch = start_epoch
    current_lr = config["start_lr"] if (scheduler is None) else scheduler.get_last_lr()[0]
    net.train()
    while True:
        
        comm.printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
        mse_list = []
        
        #for inputs_raw, labels, source in train_loader:
        for token in train_loader:
            
            if config["enable_masks"]:
                inputs_raw, label, masks, filename = token
            else:
                inputs_raw, label, filename = token

            if dist is not None:
                # generate noise vector and concat with inputs
                inputs_noise = dist.rsample( (inputs_raw.shape[0], config["noise_dimensions"], inputs_raw.shape[2], inputs_raw.shape[3]) ).to(device)
                inputs = torch.cat((inputs_raw, inputs_noise), dim = 1)
            else:
                inputs = inputs_raw
            
            # forward pass
            outputs = net(inputs)
            
            # Compute loss and average across nodes
            if config["enable_masks"]:
                loss_dict = {"valid": criterion(outputs, label, masks), "hole": criterion(outputs, label, 1. - masks)}
                loss = 0.
                loss_norm = 0.
                for key in loss_dict:
                    loss += loss_dict[key] * loss_weights[key]
                    loss_norm += loss_weights[key]
            else:
                loss = criterion(outputs, label) #/ loss_norm
            loss_avg = comm.metric_average(loss, "train_loss", device=device)
            mse_list.append(loss_avg)
            
            # Backprop
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            #step counter
            step += 1

            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

            #print some metrics
            comm.printr('{:14.4f} REPORT training: step {} loss {} LR {}'.format(dt.datetime.now().timestamp(), step, loss_avg, current_lr), 0)

            #visualize if requested
            if (step % config["training_visualization_frequency"] == 0) and (comm.rank() == 0):
                sample_idx = np.random.randint(low=0, high=label.shape[0])
                plotname = os.path.join(output_dir, "plot_train_step{}_sampleid{}.png".format(step,sample_idx))
                prediction = outputs.detach()[sample_idx,...].cpu().numpy()
                groundtruth = label.detach()[sample_idx,...].cpu().numpy()
                gpviz.visualize_prediction(plotname, prediction, groundtruth)
                
                #log if requested
                if config["logging_frequency"] > 0:
                    img = Image.open(plotname)
                    wandb.log({"Train Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
            
            #log if requested
            if (config["logging_frequency"] > 0) and (step % config["logging_frequency"] == 0) and (comm.rank() == 0):
                wandb.log({"Training Loss": loss_avg}, step = step)
                wandb.log({"Current Learning Rate": current_lr}, step = step)
                
            # validation step if desired
            if (step % config["validation_frequency"] == 0):
                
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
                        
                        if config["enable_masks"]:
                            inputs_raw_val, label_val, masks_val, filename = token_val
                        else:
                            inputs_raw_val, label_val, filename = token_val

                        # generate noise vector and concat with inputs
                        if dist is not None:
                            inputs_noise_val = dist.rsample( (inputs_raw_val.shape[0], config["noise_dimensions"], inputs_raw_val.shape[2], inputs_raw_val.shape[3]) ).to(device)
                            inputs_val = torch.cat((inputs_raw_val, inputs_noise_val), dim = 1)
                        else:
                            inputs_val = inputs_raw_val
                        
                        # forward pass
                        outputs_val = net(inputs_val)

                        # Compute loss and average across nodes
                        if config["enable_masks"]:
                            loss_dict = {"valid": criterion(outputs_val, label_val, masks_val),
                                         "hole": criterion(outputs_val, label_val, 1. - masks_val)}
                            loss_val = 0.
                            loss_norm_val = 0.
                            for key in loss_dict:
                                loss_val += loss_dict[key] * loss_weights[key]
                                loss_norm_val += loss_weights[key]
                        else:
                            loss_val = criterion(outputs_val, label_val) #/ loss_norm_val

                        # append to list
                        mse_list_val.append(loss_val)

                        # compute quantities relevant for R2
                        outarr = pproc.process(torch.squeeze(outputs_val.cpu(), dim = 0).numpy())
                        valarr = pproc.process(torch.squeeze(label_val.cpu(), dim = 0).numpy())
                        val_mean += np.mean(valarr)
                        val_var += np.mean(np.square(valarr))
                        pred_var += np.mean(np.square(outarr - valarr))
                        
                        # visualize the last sample if requested
                        if (step_val % config["validation_visualization_frequency"] == 0) and (comm.rank() == 0):
                            sample_idx = np.random.randint(low=0, high=label_val.shape[0])
                            plotname = os.path.join(output_dir, "plot_validation_step{}_valstep{}_sampleid{}.png".format(step,step_val,sample_idx))
                            prediction_val = outputs_val.detach()[sample_idx,...].cpu().numpy()
                            groundtruth_val = label_val.detach()[sample_idx,...].cpu().numpy()
                            gpviz.visualize_prediction(plotname, prediction_val, groundtruth_val)
                            
                            #log if requested
                            if config["logging_frequency"] > 0:
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
                if (config["logging_frequency"] > 0) and (comm.rank() == 0):
                    wandb.log({"Validation Loss": loss_val_avg}, step=step)
                    wandb.log({"Validation R2": val_r2_avg}, step=step)

                # set to train
                net.train()
            
            #save model if desired
            if (step % config["save_frequency"] == 0) and (comm.rank() == 0):
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
		}
                torch.save(checkpoint, os.path.join(output_dir, config["model_prefix"] + "_step_" + str(step) + ".cpt") )

            #are we done?
            if step >= config["max_steps"]:
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
            torch.save(checkpoint, os.path.join(output_dir, config["model_prefix"] + "_epoch_" + str(epoch) + ".cpt") )

        #are we done?
        if step >= config["max_steps"]:
            break

    comm.printr('{:14.4f} REPORT: finishing training'.format(dt.datetime.now().timestamp()), 0)
    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--run_tag", type=str, default="hpo", help="Unique run tag, to allow for better identification")
    AP.add_argument("--config_file", type=str, default=None, help="YAML file to read config data from. If none specified, use WandB")
    AP.add_argument("--tag_run", action='store_true', help="Tag the WandB run with the config tag. Do not use in HPO mode.")
    pargs, _ = AP.parse_known_args()
    
    #run the stuff
    main(pargs)
