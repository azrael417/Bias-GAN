# Basics
import os
import sys
import wandb
import numpy as np
import argparse as ap
import datetime as dt
import time
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
from architecture.gpsro import infill3d as dxi
from utils import gpsro_visualizer as gp
from utils import gpsro_postprocessor as pp

#vis stuff
from PIL import Image

#apex and AMP
import torch.cuda.amp as amp
import apex.optimizers as aoptim

#horovod
#import horovod.torch as hvd
from comm.distributed import comm as distcomm


class Infill3d(object):
    
    def __init__(self, config):
        # init communicator
        self.comm = distcomm(mode="dummy")

        #set seed
        seed = 333 + 7 * self.comm.rank()
    
        # Some setup
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            self.comm.printr("Using GPUs",0)
            self.device = torch.device("cuda", self.comm.local_rank())
            torch.cuda.manual_seed(seed)
            torch.cuda.set_device(self.device.index)
        else:
            self.comm.printr("Using CPUs",0)
            self.device = torch.device("cpu")
            
        # set benchmark mode
        torch.backends.cudnn.benchmark = True
        
        # get wandb api token
        with open("/certs/.wandbirc_gpsro") as f:
            wbtoken = f.readlines()[0].replace("\n","")
            # log in: that call can be blocking, it should be quick
            sp.call(["wandb", "login", wbtoken])

        # init config and wandb:
        if "run_tag" in config.keys():
            wandb.init(project = 'GPSRO bias correction', config = config, name = config["run_tag"], id = config["run_tag"])
        else:
            wandb.init(project = 'GPSRO bias correction', config = config)
        
        # extract config
        self.config = wandb.config
        
        #set up directories
        root_dir = self.config["root_dir"]
        
        # create run tag if not existing
        if "run_tag" not in self.config.keys():
            run_tag = time.strftime(f'run_%Y%m%d-%H%M%S')
            self.config["run_tag"] = run_tag
        
        self.output_dir = self.config["output_dir"].format(run_tag=self.config["run_tag"])
    
        if self.comm.rank() == 0:
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

        # Determine normalizer
        normalizer = None
        if self.config["layer_normalization"] == "batch_norm":
            normalizer = torch.nn.BatchNorm3d
        elif self.config["layer_normalization"] == "instance_norm":
            normalizer = torch.nn.InstanceNorm3d
        else:
            raise NotImplementedError("Error, " + self.config["layer_normalization"] + " not supported")

        # Define architecture
        n_input_channels = 1 + self.config["noise_dimensions"]
        n_output_channels = 1
        self.net = dxi.PConvUNet3d(input_channels = n_input_channels, output_channels = n_output_channels, 
                                   normalizer = normalizer, layer_size = 6)
        #self.net = dxi.PartialEncoderDecoderDirectPadSmall(n_input_channels, n_output_channels, factor=1, use_oracle_design=False)
        self.net.to(self.device)

        #select loss
        self.criterion = losses.InpaintingLoss(loss_type = self.config["loss_type"]).to(self.device)

        self.loss_weights = None
        if "loss_weights.valid" in self.config.keys():
            self.loss_weights = {x.split(".")[1]: float(self.config[x]) for x in self.config.keys() if x.startswith("loss_weights")}
        elif "loss_weights" in self.config.keys():
            self.loss_weights = self.config["loss_weights"]
            self.loss_weights = {x: float(self.loss_weights[x]) for x in self.loss_weights.keys()}
        
        #noise vector
        self.dist = None
        if self.config["noise_dimensions"] > 0:
            if self.config["noise_type"] == "Uniform":
                self.dist = torch.distributions.uniform.Uniform(0., 1.)
            elif pargs.noise_type == "Normal":
                self.dist = torch.distributions.normal.Normal(0., 1.)
            else:
                raise NotImplementedError("Error, noise type {} not supported.".format(self.config["noise_dimensions"]))
    
        #select optimizer
        self.optimizer = ph.get_optimizer(self.net, self.config["optimizer"], 
                                          self.config["start_lr"], 
                                          self.config["adam_eps"], 
                                          self.config["weight_decay"])

        # grad scaler
        self.gscaler = amp.GradScaler(enabled = self.config["enable_amp"])
    
        # load net and optimizer states
        self.start_step, self.start_epoch = self.comm.init_training_state(self.net, self.optimizer, self.config["checkpoint"], self.device)
    
        # make model distributed
        self.net = self.comm.DistributedModel(self.net)

        #select scheduler
        self.scheduler = None
        if "lr_schedule" in self.config.keys():
            self.scheduler = ph.get_lr_schedule(self.config["start_lr"], self.config["lr_schedule"], self.optimizer, last_step = self.start_step)
        elif "lr_schedule.type" in self.config.keys():
            scheddict = {x.split(".")[1]: self.config[x] for x in self.config.keys() if x.startswith("lr_schedule")}
            self.scheduler = ph.get_lr_schedule(self.config["start_lr"], scheddict, self.optimizer, last_step = self.start_step)
            
        #wrap the optimizer
        self.optimizer = self.comm.DistributedOptimizer(self.optimizer,
                                                        named_parameters=self.net.named_parameters(),
                                                        compression_name = None,
                                                        op_name = "average")

        # Set up the data feeder
        # train
        train_dir = os.path.join(root_dir, "train")
        train_set = gpsro.GPSRODataset(train_dir,
                                       statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                       channels = self.config["channels"],
                                       normalization_type = "MinMax" if self.config["noise_type"] == "Uniform" else "MeanVariance",
                                       shuffle = True,
                                       masks = True,
                                       shard_idx = self.comm.rank(), shard_num = self.comm.size(),
                                       num_intra_threads = self.config["max_intra_threads"],
                                       read_device = torch.device("cpu") if not self.config["enable_gds"] else self.device,
                                       send_device = self.device)
        self.train_loader = DataLoader(train_set, self.config["local_batch_size"], drop_last=True)
    
        # validation
        validation_dir = os.path.join(root_dir, "validation")
        validation_set = gpsro.GPSRODataset(validation_dir,
                                            statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                            channels = self.config["channels"],
                                            normalization_type = "MinMax" if self.config["noise_type"] == "Uniform" else "MeanVariance",
                                            shuffle = True,
                                            masks = True,
                                            shard_idx = self.comm.rank(), shard_num = self.comm.size(),
                                            num_intra_threads = self.config["max_intra_threads"],
                                            read_device = torch.device("cpu") if not self.config["enable_gds"] else self.device,
                                            send_device = self.device)
        self.validation_loader = DataLoader(validation_set, self.config["local_batch_size"], drop_last=True)
                                   
        # visualizer
        self.gpviz = gp.GPSROVisualizer(statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                       channels = self.config["channels"],
                                       normalize = True,
                                       normalization_type = "MinMax" if self.config["noise_type"] == "Uniform" else "MeanVariance")

        # postprocessing
        self.pproc = pp.GPSROPostprocessor(statsfile = os.path.join(root_dir, 'stats3d.npz'),
                                          channels = self.config["channels"],
                                          normalization_type = "MinMax" if self.config["noise_type"] == "Uniform" else "MeanVariance")

        # some metrics we want to keep track of
        self.validation_losses = []
                                          
    def train(self):
        
        # Train network
        if (self.config["logging_frequency"] > 0) and (self.comm.rank() == 0):
            wandb.watch(self.net)
    
        # report training started
        self.comm.printr('{:14.4f} REPORT: starting training'.format(dt.datetime.now().timestamp()), 0)
    
        # get start info
        step = self.start_step
        epoch = self.start_epoch
        current_lr = self.config["start_lr"] if (self.scheduler is None) else self.scheduler.get_last_lr()[0]
    
        self.net.train()
        while True:
            self.comm.printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
            loss_list = []
            
            #for inputs_raw, labels, source in train_loader:
            for inputs_raw, label, masks, filename in self.train_loader:
                
                #unsqueeze
                inputs_raw = torch.unsqueeze(inputs_raw, dim=1)
                label = torch.unsqueeze(label, dim=1)
                masks = torch.unsqueeze(masks, dim=1)
                
                if self.dist is not None:
                    # generate noise vector and concat with inputs
                    inputs_noise = self.dist.rsample( (inputs_raw.shape[0], \
                                                        self.config["noise_dimensions"], \
                                                        inputs_raw.shape[1], \
                                                        inputs_raw.shape[2], \
                                                        inputs_raw.shape[3]) ).to(self.device)
                    inputs = torch.cat((inputs_raw, inputs_noise), dim = 1)
                else:
                    inputs = inputs_raw
            
                # forward pass and loss
                with amp.autocast(enabled = self.config["enable_amp"]):
                    outputs, _ = self.net(inputs, masks)
                    loss_dict = self.criterion(inputs_raw, outputs, label, masks)
                    
                # reduce loss across ranks
                loss_dict_avg = {}
                loss = 0.
                for key in loss_dict:
                    loss += loss_dict[key] * self.loss_weights[key]
                    loss_dict_avg[key] = self.comm.metric_average(loss_dict[key], "train_loss_" + key, device = self.device)
                
                loss_avg = self.comm.metric_average(loss, "train_loss", device = self.device)
                loss_list.append(loss_avg)
            
                # Backprop
                self.optimizer.zero_grad()
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
                self.gscaler.update()

                #step counter
                step += 1

                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.scheduler.step()

                #print some metrics
                self.comm.printr('{:14.4f} REPORT training: step {} loss {} LR {}'.format(dt.datetime.now().timestamp(), step, loss_avg, current_lr), 0)

                #visualize if requested
                if (step % self.config["training_visualization_frequency"] == 0) and (self.comm.rank() == 0):
                    sample_idx = np.random.randint(low=0, high=label.shape[0])
                    plotname = os.path.join(self.output_dir, "plot_train_step{}_sampleid{}.png".format(step, sample_idx))
                    prediction = outputs.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                    groundtruth = label.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                    self.gpviz.visualize_prediction(plotname, prediction, groundtruth)
                
                    #log if requested
                    if self.config["logging_frequency"] > 0:
                        img = Image.open(plotname)
                        wandb.log({"Train Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
            
                #log if requested
                if (self.config["logging_frequency"] > 0) and (step % self.config["logging_frequency"] == 0) and (self.comm.rank() == 0):
                    wandb.log({"Training Loss total": loss_avg}, step = step)
                    for key in loss_dict_avg:
                        wandb.log({"Training Loss " + key : loss_dict_avg[key]}, step = step)
                    wandb.log({"Current Learning Rate": current_lr}, step = step)
                
                # validation step if desired
                if (step % self.config["validation_frequency"] == 0):
                    self.validate(step, epoch)
                    self.net.train()
            
                #save model if desired
                if (step % self.config["save_frequency"] == 0) and (self.comm.rank() == 0):
                    checkpoint = {
                        'step': step,
                        'epoch': epoch,
                        'model': self.net.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(self.output_dir, self.config["model_prefix"] + "_step_" + str(step) + ".cpt") )

                #are we done?
                if step >= self.config["max_steps"]:
                    break
                
            #do some after-epoch prep, just for the books
            epoch += 1
          
            # save model
            if self.comm.rank()==0:
          
                # Save the model
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(self.output_dir, self.config["model_prefix"] + "_epoch_" + str(epoch) + ".cpt") )

            #are we done?
            if step >= self.config["max_steps"]:
                break
                
        self.comm.printr('{:14.4f} REPORT: finishing training'.format(dt.datetime.now().timestamp()), 0)

    
    def validate(self, step, epoch):
        #eval
        self.net.eval()

        # vali loss
        loss_list_val = []
        
        # these we need for the R2-Value
        val_mean = 0.
        val_var = 0.
        pred_var = 0.
        
        # disable gradients
        with torch.no_grad():

            # iterate over validation sample
            for step_val, (inputs_raw_val, label_val, masks_val, filename) in enumerate(self.validation_loader):
                
                #unsqueeze
                inputs_raw_val = torch.unsqueeze(inputs_raw_val, dim=1)
                label_val = torch.unsqueeze(label_val, dim=1)
                masks_val = torch.unsqueeze(masks_val, dim=1)
                
                # generate noise vector and concat with inputs
                if self.dist is not None:
                    inputs_noise_val = self.dist.rsample( (inputs_raw_val.shape[0], \
                                                        self.config["noise_dimensions"], \
                                                        inputs_raw_val.shape[1], \
                                                        inputs_raw_val.shape[2], \
                                                        inputs_raw_val.shape[3]) ).to(self.device)
                    inputs_val = torch.cat((inputs_raw_val, inputs_noise_val), dim = 1)
                else:
                    inputs_val = inputs_raw_val
        
                # forward pass
                outputs_val, _ = self.net(inputs_val, masks_val)

                # Compute loss and average across nodes
                loss_dict = self.criterion(inputs_raw_val, outputs_val, label_val, masks_val)
                loss_val = 0.
                for key in loss_dict:
                    loss_val += loss_dict[key] * self.loss_weights[key]

                # append to list
                loss_list_val.append(loss_val)
                
                # compute quantities relevant for R2
                outarr = self.pproc.process(torch.squeeze(outputs_val.cpu(), dim = 0).numpy())
                valarr = self.pproc.process(torch.squeeze(label_val.cpu(), dim = 0).numpy())
                val_mean += np.mean(valarr)
                val_var += np.mean(np.square(valarr))
                pred_var += np.mean(np.square(outarr - valarr))
                
                # visualize the last sample if requested
                if (step_val % self.config["validation_visualization_frequency"] == 0) and (self.comm.rank() == 0):
                    sample_idx = np.random.randint(low=0, high=label_val.shape[0])
                    plotname = os.path.join(self.output_dir, "plot_validation_step{}_valstep{}_sampleid{}.png".format(step,step_val,sample_idx))
                    prediction_val = outputs_val.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                    groundtruth_val = label_val.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                    self.gpviz.visualize_prediction(plotname, prediction_val, groundtruth_val)
                
                    #log if requested
                    if self.config["logging_frequency"] > 0:
                        img = Image.open(plotname)
                        wandb.log({"Validation Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)

        # average the validation loss
        count_val = float(len(loss_list_val))
        count_val_global = self.comm.metric_average(count_val, "val_count", op_name="sum", device=self.device)
        loss_val = sum(loss_list_val)
        loss_val_global = self.comm.metric_average(loss_val, "val_loss", op_name="sum", device=self.device)
        loss_val_avg = loss_val_global / count_val_global

        # computing R2 Value
        # divide by sample size:
        val_mean_global = self.comm.metric_average(val_mean, "val_mean", op_name="sum", device=self.device)
        val_mean_avg = val_mean_global / count_val_global
        val_var_global = self.comm.metric_average(val_var, "val_var", op_name="sum", device=self.device)
        val_var_avg = val_var_global / count_val_global
        pred_var_global = self.comm.metric_average(pred_var, "pred_var", op_name="sum", device=self.device)
        pred_var_avg = pred_var_global / count_val_global
        # finalize the variance
        val_var_avg -= np.square(val_mean_avg)
        # take the ratio
        val_r2_avg = 1. - pred_var_avg / val_var_avg
    
        # print results
        self.comm.printr('{:14.4f} REPORT validation: step {} loss {}'.format(dt.datetime.now().timestamp(), step, loss_val_avg), 0)
        self.comm.printr('{:14.4f} REPORT validation: step {} R2 {}'.format(dt.datetime.now().timestamp(), step, val_r2_avg), 0)

        # append to loss list
        self.validation_losses.append(loss_val_avg)
        
        # log in wandb
        if (self.config["logging_frequency"] > 0) and (self.comm.rank() == 0):
            wandb.log({"Validation Loss Total": loss_val_avg}, step=step)
            wandb.log({"Validation R2": val_r2_avg}, step=step)
            
            
