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
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from utils import losses
from utils import metrics
from utils import yparams as yp
from utils import parsing_helpers as ph
from data import gpsro_dataset as gpsro
from architecture.gpsro import infill3d_gan as dxi
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


class Infill3dGAN(object):
    
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

        # Determine normalizers
        gen_normalizer = None
        if self.config["gen_layer_normalization"] == "batch_norm":
            gen_normalizer = torch.nn.BatchNorm3d
        elif self.config["gen_layer_normalization"] == "instance_norm":
            gen_normalizer = torch.nn.InstanceNorm3d
        else:
            raise NotImplementedError("Error, " + self.config["gen_layer_normalization"] + " not supported")
            
        disc_normalizer = None
        if self.config["disc_layer_normalization"] == "batch_norm":
            disc_normalizer = torch.nn.BatchNorm3d
        elif self.config["disc_layer_normalization"] == "instance_norm":
            disc_normalizer = torch.nn.InstanceNorm3d
        else:
            raise NotImplementedError("Error, " + self.config["disc_layer_normalization"] + " not supported")

        # Define architecture
        n_input_channels = 1 + self.config["noise_dimensions"]
        n_output_channels = 1
        self.net = dxi.GAN(input_channels = n_input_channels, output_channels = n_output_channels, 
                           gen_normalizer = gen_normalizer, disc_normalizer = disc_normalizer,
                           gen_layer_size = 6, disc_layer_size = 6)
        
        # extract handles
        self.generator = self.net.generator.to(self.device)
        self.discriminator = self.net.discriminator.to(self.device)

        #select loss
        self.reconst_criterion = losses.InpaintingLoss(loss_type = self.config["loss_type"]).to(self.device)
        self.gan_criterion = losses.GANLoss("ModifiedMinMax", self.config["local_batch_size"], self.device)

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
            elif self.config["noise_type"] == "Normal":
                self.dist = torch.distributions.normal.Normal(0., 1.)
            else:
                raise NotImplementedError("Error, noise type {} not supported.".format(self.config["noise_type"]))
        else:
            raise NotImplementedError("Error, please use at least one noise dimension.")
    
        #select optimizers
        self.gen_optimizer = ph.get_optimizer(self.generator.parameters(), 
                                              self.config["gen_optimizer"], 
                                              self.config["gen_start_lr"], 
                                              self.config["gen_adam_eps"], 
                                              self.config["gen_weight_decay"])
                                              
        self.disc_optimizer = ph.get_optimizer(self.discriminator.parameters(), 
                                              self.config["disc_optimizer"], 
                                              self.config["disc_start_lr"], 
                                              self.config["disc_adam_eps"], 
                                              self.config["disc_weight_decay"])

        # grad scalers
        self.gen_gscaler = amp.GradScaler(enabled = self.config["enable_amp"])
        self.disc_gscaler = amp.GradScaler(enabled = self.config["enable_amp"])
    
        # load net and optimizer states
        self.start_step, self.start_epoch = self.comm.init_gan_training_state(self.generator, self.discriminator, 
                                                                              self.gen_optimizer, self.disc_optimizer,
                                                                              self.config["checkpoint"], self.device)
    
        # make model distributed
        self.generator = self.comm.DistributedModel(self.generator)
        self.discriminator = self.comm.DistributedModel(self.discriminator)

        #select scheduler
        self.gen_scheduler = None
        if "gen_lr_schedule" in self.config.keys():
            self.gen_scheduler = ph.get_lr_schedule(self.config["gen_start_lr"], self.config["gen_lr_schedule"], 
                                                    self.gen_optimizer, last_step = self.start_step)
        elif "gen_lr_schedule.type" in self.config.keys():
            scheddict = {x.split(".")[1]: self.config[x] for x in self.config.keys() if x.startswith("gen_lr_schedule")}
            self.gen_scheduler = ph.get_lr_schedule(self.config["gen_start_lr"], scheddict, self.gen_optimizer, last_step = self.start_step)
        
        self.disc_scheduler = None
        if "disc_lr_schedule" in self.config.keys():
            self.disc_scheduler = ph.get_lr_schedule(self.config["disc_start_lr"], self.config["disc_lr_schedule"], 
                                                     self.disc_optimizer, last_step = self.start_step)
        elif "disc_lr_schedule.type" in self.config.keys():
            scheddict = {x.split(".")[1]: self.config[x] for x in self.config.keys() if x.startswith("disc_lr_schedule")}
            self.disc_scheduler = ph.get_lr_schedule(self.config["disc_start_lr"], scheddict, self.disc_optimizer, last_step = self.start_step)
            
        #wrap the optimizer
        self.gen_optimizer = self.comm.DistributedOptimizer(self.gen_optimizer,
                                                            named_parameters=self.generator.named_parameters(),
                                                            compression_name = None,
                                                            op_name = "average")
        self.disc_optimizer = self.comm.DistributedOptimizer(self.disc_optimizer,
                                                             named_parameters=self.discriminator.named_parameters(),
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
            wandb.watch(self.generator)
            wandb.watch(self.discriminator)
    
        # report training started
        self.comm.printr('{:14.4f} REPORT: starting training'.format(dt.datetime.now().timestamp()), 0)
    
        # get start info
        step = self.start_step
        epoch = self.start_epoch

        # get LR
        if self.disc_scheduler is not None:
            d_current_lr = self.disc_scheduler.get_last_lr()[0]
        if self.gen_scheduler is not None:
            g_current_lr = self.gen_scheduler.get_last_lr()[0]
        
        self.generator.train()
        self.discriminator.train()
        d_acc_avg = 0.5
        while True:
            self.comm.printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
            d_loss_list = []
            g_loss_list = []
            
            #for inputs_raw, labels, source in train_loader:
            for inputs_raw, outputs_real, masks_raw, filename in self.train_loader:
                
                #unsqueeze
                inputs_raw = torch.unsqueeze(inputs_raw, dim = 1).to(self.device)
                outputs_real = torch.unsqueeze(outputs_real, dim = 1).to(self.device)
                masks_raw = torch.unsqueeze(masks_raw, dim = 1).to(self.device)
                
                # generate noise vector and concat with inputs, also do same with masks
                inputs_noise = self.dist.rsample( (inputs_raw.shape[0], \
                                                    self.config["noise_dimensions"], \
                                                    inputs_raw.shape[2], \
                                                    inputs_raw.shape[3], \
                                                    inputs_raw.shape[4]) ).to(self.device)
                masks_noise = torch.ones( (masks_raw.shape[0], \
                                           self.config["noise_dimensions"], \
                                           masks_raw.shape[2], \
                                           masks_raw.shape[3], \
                                           masks_raw.shape[4]) ).to(self.device)
                inputs = torch.cat((inputs_raw, inputs_noise), dim = 1)
                masks = torch.cat((masks_raw, masks_noise), dim = 1)
                
                # see what we need to train
                train_generator = True
                train_discriminator = True
                if step < self.config["gen_warmup_steps"]:
                    train_generator = True
                    train_discriminator = False
                else:
                    if (d_acc_avg > self.config["disc_acc_max"]):
                        # discriminator is too good:
                        train_generator = True
                        train_discriminator = False
                    elif (d_acc_avg < self.config["disc_acc_min"]):
                        # discriminator is too bad
                        train_generator = False
                        train_discriminator = True
                    else:
                        # all good, update both
                        train_generator = True
                        train_discriminator = True
                
                
                # Discriminator part
                with amp.autocast(enabled = self.config["enable_amp"]):
                    outputs_fake, _ = self.generator(inputs, masks)
                    logits_real, prediction_real = self.discriminator(outputs_real, masks)
                    logits_fake, prediction_fake = self.discriminator(outputs_fake, masks)
                    # losses
                    d_loss = self.gan_criterion.d_loss(logits_real, logits_fake)
                    d_loss *= self.loss_weights["adv"]
                        
                # average the loss
                d_loss_avg = self.comm.metric_average(d_loss, "train_loss_d", device = self.device)
                d_loss_list.append(d_loss_avg)
                
                # accuracy calculation
                d_acc = 0.5 * (metrics.accuracy(prediction_real, self.gan_criterion.label_real) 
                                + metrics.accuracy(prediction_fake, self.gan_criterion.label_fake))
                d_acc_avg = self.comm.metric_average(d_acc, "train_accuracy_d", device = self.device)
                
                # train disco
                if train_discriminator:
                    # Backprop
                    self.disc_optimizer.zero_grad()
                    self.disc_gscaler.scale(d_loss).backward()
                    self.disc_gscaler.step(self.disc_optimizer)
                    self.disc_gscaler.update()
                    
                    if self.disc_scheduler is not None:
                        d_current_lr = self.disc_scheduler.get_last_lr()[0]
                        self.disc_scheduler.step()
                
                
                # Generator part
                with amp.autocast(enabled = self.config["enable_amp"]):
                    outputs_fake, _ = self.generator(inputs, masks)
                    logits_fake, _ = self.discriminator(outputs_fake, masks)
                    # reconstruction loss
                    g_loss_dict = self.reconst_criterion(inputs_raw, outputs_fake, outputs_real, masks_raw)
                    # adversarial loss
                    if step >= self.config["gen_warmup_steps"]:
                        g_loss_dict["adv"] = self.gan_criterion.g_loss(logits_fake)
                    
                # reduce across ranks
                g_loss_dict_avg = {}
                g_loss = 0.
                for key in g_loss_dict:
                    g_loss += g_loss_dict[key] * self.loss_weights[key]
                    g_loss_dict_avg[key] = self.comm.metric_average(g_loss_dict[key], "train_loss_g_" + key, device = self.device)
                
                # combine losses and average
                g_loss_avg = self.comm.metric_average(g_loss, "train_loss_g", device = self.device)
                g_loss_list.append(g_loss_avg)
            
                if train_generator:
                    self.gen_optimizer.zero_grad()
                    self.gen_gscaler.scale(g_loss).backward()
                    self.gen_gscaler.step(self.gen_optimizer)
                    self.gen_gscaler.update()
                    
                    if self.gen_scheduler is not None:
                        g_current_lr = self.gen_scheduler.get_last_lr()[0]
                        self.gen_scheduler.step()

                #step counter
                step += 1

                #print some metrics
                self.comm.printr('{:14.4f} REPORT training: step {} d_loss {} g_loss {} d_LR {} g_LR {}'.format(dt.datetime.now().timestamp(), 
                                                                                                                step, d_loss_avg, g_loss_avg,
                                                                                                                d_current_lr, g_current_lr), 0)

                #visualize if requested
                if (step % self.config["training_visualization_frequency"] == 0) and (self.comm.rank() == 0):
                    sample_idx = np.random.randint(low=0, high=outputs_real.shape[0])
                    plotname = os.path.join(self.output_dir, "plot_train_step{}_sampleid{}.png".format(step, sample_idx))
                    prediction = outputs_fake.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                    groundtruth = outputs_real.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                    self.gpviz.visualize_prediction(plotname, prediction, groundtruth)
                
                    #log if requested
                    if self.config["logging_frequency"] > 0:
                        img = Image.open(plotname)
                        wandb.log({"Train Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
            
                #log if requested
                if (self.config["logging_frequency"] > 0) and (step % self.config["logging_frequency"] == 0) and (self.comm.rank() == 0):
                    wandb.log({"Training Loss Generator total": g_loss_avg}, step = step)
                    wandb.log({"Training Loss Discriminator total": d_loss_avg}, step = step)
                    wandb.log({"Training Accuracy Discriminator total": d_acc_avg}, step = step)
                    for key in g_loss_dict_avg:
                        wandb.log({"Training Loss Generator " + key : g_loss_dict_avg[key]}, step = step)
                    wandb.log({"Current Learning Rate Generator": g_current_lr}, step = step)
                    wandb.log({"Current Learning Rate Discriminator": d_current_lr}, step = step)
                
                # validation step if desired
                if (step % self.config["validation_frequency"] == 0):
                    self.validate(step, epoch)
                    self.generator.train()
                    self.discriminator.train()
            
                #save model if desired
                if (step % self.config["save_frequency"] == 0) and (self.comm.rank() == 0):
                    checkpoint = {
                        'step': step,
                        'epoch': epoch,
                        'generator': self.generator.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'g_opt': self.gen_optimizer.state_dict(),
                        'd_opt': self.disc_optimizer.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(self.output_dir, self.config["model_prefix"] + "_step_" + str(step) + ".cpt") )

                #are we done?
                if step >= self.config["max_steps"]:
                    break
                
            #do some after-epoch prep, just for the books
            epoch += 1

            #are we done?
            if step >= self.config["max_steps"]:
                break
                
        self.comm.printr('{:14.4f} REPORT: finishing training'.format(dt.datetime.now().timestamp()), 0)

    
    def validate(self, step, epoch):
        #eval
        self.generator.eval()
        self.discriminator.eval()

        # vali loss
        loss_list_val = []
        
        # these we need for the R2-Value
        val_mean = 0.
        val_var = 0.
        pred_var = 0.
        
        # disable gradients
        with torch.no_grad():

            # iterate over validation sample
            for step_val, (inputs_raw_val, outputs_real_val, masks_raw_val, filename) in enumerate(self.validation_loader):
                
                #unsqueeze
                inputs_raw_val = torch.unsqueeze(inputs_raw_val, dim=1).to(self.device)
                outputs_real_val = torch.unsqueeze(outputs_real_val, dim=1).to(self.device)
                masks_raw_val = torch.unsqueeze(masks_raw_val, dim=1).to(self.device)
                
                # generate noise vector and concat with inputs
                inputs_noise_val = self.dist.rsample( (inputs_raw_val.shape[0], \
                                                       self.config["noise_dimensions"], \
                                                       inputs_raw_val.shape[2], \
                                                       inputs_raw_val.shape[3], \
                                                       inputs_raw_val.shape[4]) ).to(self.device)
                masks_noise_val = torch.ones( (masks_raw_val.shape[0], \
                                               self.config["noise_dimensions"], \
                                               masks_raw_val.shape[2], \
                                               masks_raw_val.shape[3], \
                                               masks_raw_val.shape[4]) ).to(self.device)
                inputs_val = torch.cat((inputs_raw_val, inputs_noise_val), dim = 1)
                masks_val = torch.cat((masks_raw_val, masks_noise_val), dim = 1)
        
                # forward pass
                with amp.autocast(enabled = self.config["enable_amp"]):
                    outputs_fake_val, _ = self.generator(inputs_val, masks_val)

                # Compute loss and average across nodes
                loss_dict = self.reconst_criterion(inputs_raw_val, outputs_fake_val, outputs_real_val, masks_val)
                loss_val = 0.
                for key in loss_dict:
                    loss_val += loss_dict[key] * self.loss_weights[key]

                # append to list
                loss_list_val.append(loss_val)
                
                # compute quantities relevant for R2
                outarr = self.pproc.process(torch.squeeze(outputs_fake_val.cpu(), dim = 0).numpy())
                valarr = self.pproc.process(torch.squeeze(outputs_real_val.cpu(), dim = 0).numpy())
                val_mean += np.mean(valarr)
                val_var += np.mean(np.square(valarr))
                pred_var += np.mean(np.square(outarr - valarr))
                
                # visualize the last sample if requested
                if (step_val % self.config["validation_visualization_frequency"] == 0) and (self.comm.rank() == 0):
                    sample_idx = np.random.randint(low=0, high=outputs_real_val.shape[0])
                    plotname = os.path.join(self.output_dir, "plot_validation_step{}_valstep{}_sampleid{}.png".format(step,step_val,sample_idx))
                    prediction_val = outputs_fake_val.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                    groundtruth_val = outputs_real_val.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
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
        
