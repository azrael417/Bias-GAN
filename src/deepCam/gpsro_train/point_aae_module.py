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
from architecture.gpsro import point_aae as aae

#vis stuff
from PIL import Image

#apex and AMP
import torch.cuda.amp as amp
import apex.optimizers as aoptim

#horovod
from comm.distributed import comm as distcomm


class PointAAE3d(object):
    
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
            # check if we have a group tag
            if "group_tag" in config.keys():
                wandb.init(project = 'GPSRO bias correction',
                           group = config["group_tag"], config = config,
                           name = config["run_tag"], id = config["run_tag"])
            else:
                wandb.init(project = 'GPSRO bias correction', config = config,
                           name = config["run_tag"], id = config["run_tag"])
        else:
            # check if we have a group tag
            if "group_tag" in config.keys():
                wandb.init(project = 'GPSRO bias correction', group = config["group_tag"], config = config)
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

        # noise params
        self.noise_mu = self.config["noise_mu"]
        self.noise_std = self.config["noise_std"]
        self.noise = torch.FloatTensor(self.config["batch_size"], self.config["latent_dim"]).to(self.device)

        # Define architecture
        self.net = aae.AAE3dModel(self.config["num_points", 
                                  self.config["num_features"], 
                                  self.config["latent_dim"],
                                  self.config["encoder_filters"], 
                                  self.config["use_encoder_bias"], 
                                  self.config["encoder_relu_slope"],
                                  self.config["generator_filters"], 
                                  self.config["use_generator_bias"], 
                                  self.config["generator_relu_slope"],
                                  self.config["discriminator_filters", 
                                  self.config["use_discriminator_bias"], 
                                  self.config["discriminator_relu_slope"])
        
        # extract handles
        self.encoder = self.net.encoder.to(self.device)
        self.generator = self.net.generator.to(self.device)
        self.discriminator = self.net.discriminator.to(self.device)

        #select losses
        self.lambda_gp = self.config["loss_weights.gp"]
        self.reconst_criterion = losses.ChamferLoss()
        self.lambda_rec = self.config["loss_weights.rec"]
    
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
        self.start_step, self.start_epoch = self.comm.init_aae_training_state(self.encoder, self.generator, self.discriminator, 
                                                                              self.gen_optimizer, self.disc_optimizer,
                                                                              self.config["checkpoint"], self.device)
    
        # make model distributed
        self.encoder = self.comm.DistributedModel(self.encoder)
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

        ## Set up the data feeder
        ## train
        #train_dir = os.path.join(root_dir, "train")
        #train_set = gpsro.GPSRODataset(train_dir,
        #                               statsfile = os.path.join(root_dir, 'stats3d.npz'),
        #                               channels = self.config["channels"],
        #                               normalization_type = "MinMax" if self.config["noise_type"] == "Uniform" else "MeanVariance",
        #                               shuffle = True,
        #                               masks = True,
        #                               shard_idx = self.comm.rank(), shard_num = self.comm.size(),
        #                               num_intra_threads = self.config["max_intra_threads"],
        #                               read_device = torch.device("cpu") if not self.config["enable_gds"] else self.device,
        #                               send_device = self.device)
        #self.train_loader = DataLoader(train_set, self.config["local_batch_size"], drop_last=True)
        #
        ## validation
        #validation_dir = os.path.join(root_dir, "validation")
        #validation_set = gpsro.GPSRODataset(validation_dir,
        #                                    statsfile = os.path.join(root_dir, 'stats3d.npz'),
        #                                    channels = self.config["channels"],
        #                                    normalization_type = "MinMax" if self.config["noise_type"] == "Uniform" else "MeanVariance",
        #                                    shuffle = True,
        #                                    masks = True,
        #                                    shard_idx = self.comm.rank(), shard_num = self.comm.size(),
        #                                    num_intra_threads = self.config["max_intra_threads"],
        #                                    read_device = torch.device("cpu") if not self.config["enable_gds"] else self.device,
        #                                    send_device = self.device)
        #self.validation_loader = DataLoader(validation_set, self.config["local_batch_size"], drop_last=True)
        #
        ## some metrics we want to keep track of
        #self.validation_losses = []
    
    def _loss_fnc_d(self, noise, real_logits, codes, fake_logits):

        with amp.autocast(enabled = self.config["enable_amp"])
            # classification loss (critic)
            loss = torch.mean(fake_logits) - torch.mean(real_logits)
        
            # gradient penalty
            alpha = torch.rand(self.batch_size, 1).to(self.devices[2])
            interpolates = noise + alpha * (codes - noise)
            disc_interpolates = self.discriminator.discriminate(interpolates)
        
        scaled_disc_interpolates = self.disc_gscaler.scale(disc_interpolates)
        scaled_gradients = torch.autograd.grad(outputs=scaled_disc_interpolates,
                                            inputs=interpolates,
                                            grad_outputs=torch.ones_like(disc_interpolates).to(self.device),
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]
        inv_scale = 1./self.disc_gscaler.get_scale()
        gradients = [p * inv_scale for p in scaled_gradients]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = ((slopes - 1) ** 2).mean()
        loss += self.lambda_gp * gradient_penalty
        
        return loss


    def _loss_fnc_eg(self, data, rec_batch, fake_logit):
        # reconstruction loss: here we need input shape (batch_size, num_points, points_dim)
        loss = self.lambda_rec * torch.mean(self.reconst_criterion(rec_batch.permute(0, 2, 1), data.permute(0, 2, 1)))

        # add generator loss if requested
        if fake_logit is not None:
            loss += -torch.mean(fake_logit)
        
        return loss
    
        
    def train(self):
        
        # Train network
        if (self.config["logging_frequency"] > 0) and (self.comm.rank() == 0):
            wandb.watch(self.encoder)
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
        
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()

        while True:
            self.comm.printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
            d_loss_list = []
            g_loss_list = []
            
            #for inputs_raw, labels, source in train_loader:
            for inputs, outputs_real, filename in self.train_loader:
                
                inputs = inputs.to(self.device)
                outputs_real = outputs_real.to(self.device)
                
                with amp.autocast(enabled = self.config["enable_amp"]):
                    # get reconstruction
                    codes, mu, logvar = self.encoder.encode(inputs)
            
                    # get noise
                    self.noise.normal_(mean = self.noise_mu, std = self.noise_std)
            
                    # get logits
                    real_logits = self.discriminator.discriminate(self.noise)
                    fake_logits = self.discriminator.discriminate(codes)
                
                    # get loss
                    d_loss = self._loss_fnc_d(self.noise, real_logits, codes, fake_logits)
                
                # average the loss
                d_loss_avg = self.comm.metric_average(d_loss, "train_loss_d", device = self.device)
                d_loss_list.append(d_loss_avg)
                
                # backward
                self.disc_optimizer.zero_grad(set_to_none = True)
                self.disc_gscaler.scale(d_loss).backward(retain_graph = True)
                self.disc_gscaler.step(self.disc_optimizer)
                self.disc_gscaler.update()
                
                if self.disc_scheduler is not None:
                    d_current_lr = self.disc_scheduler.get_last_lr()[0]
                    self.disc_scheduler.step()
                
                with amp.autocast(enabled = self.config["enable_amp"]):
                    # eg step
                    rec_batch = self.generator.generate(codes)
                    fake_logit = self.discriminator.discriminate(codes)
            
                    # get loss
                    g_loss = self._loss_fnc_eg(outputs_real, rec_batch, fake_logit)
                
                # combine losses and average
                g_loss_avg = self.comm.metric_average(g_loss, "train_loss_g", device = self.device)
                g_loss_list.append(g_loss_avg)
            
                # backward
                self.gen_optimizer.zero_grad(set_to_none = True)
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
            
                #log if requested
                if (self.config["logging_frequency"] > 0) and (step % self.config["logging_frequency"] == 0) and (self.comm.rank() == 0):
                    wandb.log({"Training Loss Generator total": g_loss_avg}, step = step)
                    wandb.log({"Training Loss Discriminator total": d_loss_avg}, step = step)
                    wandb.log({"Current Learning Rate Generator": g_current_lr}, step = step)
                    wandb.log({"Current Learning Rate Discriminator": d_current_lr}, step = step)
                
                # validation step if desired
                if (step % self.config["validation_frequency"] == 0):
                    self.validate(step, epoch)
                    self.encoder.train()
                    self.generator.train()
                    self.discriminator.train()
            
                #save model if desired
                if (step % self.config["save_frequency"] == 0) and (self.comm.rank() == 0):
                    checkpoint = {
                        'step': step,
                        'epoch': epoch,
                        'encoder': self.encoder.state_dict(),
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
        self.encoder.eval()
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
            for step_val, (inputs_val, outputs_real_val, filename) in enumerate(self.validation_loader):
                
                #unsqueeze
                inputs_val = inputs_val.to(self.device)
                outputs_real_val = outputs_real_val.to(self.device)
        
                # forward pass
                with amp.autocast(enabled = self.config["enable_amp"]):
                    codes_val, mu_val, logvar_val = self.encoder.encode(inputs_val)
                    # just reconstruction loss is important here
                    outputs_fake_val = self.generator.generate(codes_val)
                
                # get loss
                g_loss_val = self._loss_fnc_eg(outputs_real_val, rec_batch, fake_logit)
                loss_val = g_loss_val
                
                # append to list
                loss_list_val.append(g_loss_val)
                
                ## compute quantities relevant for R2
                #outarr = self.pproc.process(torch.squeeze(outputs_fake_val.cpu(), dim = 0).numpy())
                #valarr = self.pproc.process(torch.squeeze(outputs_real_val.cpu(), dim = 0).numpy())
                #val_mean += np.mean(valarr)
                #val_var += np.mean(np.square(valarr))
                #pred_var += np.mean(np.square(outarr - valarr))
                #
                ## visualize the last sample if requested
                #if (step_val % self.config["validation_visualization_frequency"] == 0) and (self.comm.rank() == 0):
                #    sample_idx = np.random.randint(low=0, high=outputs_real_val.shape[0])
                #    plotname = os.path.join(self.output_dir, "plot_validation_step{}_valstep{}_sampleid{}.png".format(step,step_val,sample_idx))
                #    prediction_val = outputs_fake_val.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                #    groundtruth_val = outputs_real_val.detach()[sample_idx, 0, ...].cpu().numpy().astype(np.float32)
                #    self.gpviz.visualize_prediction(plotname, prediction_val, groundtruth_val)
                #
                #    #log if requested
                #    if self.config["logging_frequency"] > 0:
                #        img = Image.open(plotname)
                #        wandb.log({"Validation Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)

        ## average the validation loss
        #count_val = float(len(loss_list_val))
        #count_val_global = self.comm.metric_average(count_val, "val_count", op_name="sum", device=self.device)
        loss_val = sum(loss_list_val)
        loss_val_global = self.comm.metric_average(loss_val, "val_loss", op_name="sum", device=self.device)
        loss_val_avg = loss_val_global / count_val_global

        ## computing R2 Value
        ## divide by sample size:
        #val_mean_global = self.comm.metric_average(val_mean, "val_mean", op_name="sum", device=self.device)
        #val_mean_avg = val_mean_global / count_val_global
        #val_var_global = self.comm.metric_average(val_var, "val_var", op_name="sum", device=self.device)
        #val_var_avg = val_var_global / count_val_global
        #pred_var_global = self.comm.metric_average(pred_var, "pred_var", op_name="sum", device=self.device)
        #pred_var_avg = pred_var_global / count_val_global
        ## finalize the variance
        #val_var_avg -= np.square(val_mean_avg)
        ## take the ratio
        #val_r2_avg = 1. - pred_var_avg / val_var_avg
        #
        # print results
        self.comm.printr('{:14.4f} REPORT validation: step {} loss {}'.format(dt.datetime.now().timestamp(), step, loss_val_avg), 0)
        #self.comm.printr('{:14.4f} REPORT validation: step {} R2 {}'.format(dt.datetime.now().timestamp(), step, val_r2_avg), 0)

        # append to loss list
        self.validation_losses.append(loss_val_avg)
        
        # log in wandb
        if (self.config["logging_frequency"] > 0) and (self.comm.rank() == 0):
            wandb.log({"Validation Loss Total": loss_val_avg}, step=step)
            #wandb.log({"Validation R2": val_r2_avg}, step=step)
