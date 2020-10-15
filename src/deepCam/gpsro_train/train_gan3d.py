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
from utils import metrics
from utils import parsing_helpers as ph
from data import gpsro_dataset as gpsro
from architecture.gpsro import deeplab3d_gan as dxg
from utils import gpsro_visualizer as gp
from utils.losses import GANLoss

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

    # init communicator
    comm = distcomm("openmpi-nccl")

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

    #set up directories
    root_dir = os.path.join(pargs.data_dir_prefix)
    output_dir = pargs.output_dir
    
    if comm.rank() == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    
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
        config.optimizer_generator = pargs.optimizer_generator
        config.optimizer_discriminator = pargs.optimizer_discriminator
        config.start_lr_generator = pargs.start_lr_generator
        config.start_lr_discriminator = pargs.start_lr_discriminator
        config.adam_eps = pargs.adam_eps
        config.weight_decay = pargs.weight_decay
        config.loss_type_gan = pargs.loss_type_gan
        config.loss_type_regression = pargs.loss_type_regression
        config.loss_weight_gan = pargs.loss_weight_gan
        config.loss_weight_regression = pargs.loss_weight_regression
        config.loss_weight_gp = pargs.loss_weight_gp
        config.generator_warmup_steps = pargs.generator_warmup_steps
        config.model_prefix = pargs.model_prefix
        config.amp_opt_level = pargs.amp_opt_level
        config.use_batchnorm = False if pargs.disable_batchnorm else True
        config.enable_masks = pargs.enable_masks

        # lr schedule if applicable
        for key in pargs.lr_schedule_generator:
            config.update({"lr_schedule_generator_"+key: pargs.lr_schedule_generator[key]}, allow_val_change = True)
        for key in pargs.lr_schedule_discriminator:
            config.update({"lr_schedule_discriminator_"+key: pargs.lr_schedule_discriminator[key]}, allow_val_change = True)
        for key in pargs.relative_update_schedule:
            config.update({"pargs.relative_update_schedule_"+key: pargs.relative_update_schedule[key]}, allow_val_change = True)

    # Define architecture: we have only one channel now
    n_input_channels = 1
    n_output_channels = 1

    # generator
    generator_normalizer = dxc.Identity if pargs.disable_batchnorm else nn.BatchNorm3d
    generator = dxg.Generator(n_input_channels, n_output_channels,
                              pargs.upsampler_type,
                              pargs.noise_type,
                              pargs.noise_dimensions,
                              os=16, pretrained=False,
                              normalizer=generator_normalizer)
    generator.to(device)

    # discriminator
    discriminator_normalizer = dxc.Identity if pargs.disable_batchnorm else nn.InstanceNorm3d if pargs.loss_type_gan == "Wasserstein" else nn.BatchNorm3d
    discriminator = dxg.Discriminator(n_input = n_input_channels,
                                      os=16, pretrained=False,
                                      normalizer=discriminator_normalizer)
    discriminator.to(device)

    #select losses
    criterion_gan = GANLoss(pargs.loss_type_gan, pargs.local_batch_size, device)
    criterion_regression = None
    if pargs.loss_type_regression == "l1":
        if pargs.enable_masks:
            criterion_regression = losses.L1LossWeighted()
        else:
            criterion_regression = nn.L1Loss()
    elif pargs.loss_type_regression == "smooth_l1":
        if pargs.enable_masks:
            criterion = losses.L1LossWeighted(normalize=False, smooth=True)
        else:
            criterion = nn.SmoothL1Loss()
    elif pargs.loss_type_regression == "l2":
        criterion_regression = nn.MSELoss()
    else:
        raise NotImplementedError("Error, loss {} not implemented.".format(pargs.loss_type_regression))

    #select optimizer
    g_opt = ph.get_optimizer(generator, pargs.optimizer_generator, pargs.start_lr_generator, pargs.adam_eps, pargs.weight_decay)
    d_opt = ph.get_optimizer(discriminator, pargs.optimizer_discriminator, pargs.start_lr_discriminator, pargs.adam_eps, pargs.weight_decay)

    #wrap net and optimizer in amp
    generator, g_opt = amp.initialize(generator, g_opt, opt_level = pargs.amp_opt_level)
    discriminator, d_opt = amp.initialize(discriminator, d_opt, opt_level = pargs.amp_opt_level)
    
    #load net and optimizer states
    start_step, start_epoch = comm.init_gan_training_state(generator, discriminator, g_opt, d_opt, pargs.checkpoint, device)
    
    #make models distributed
    generator = comm.DistributedModel(generator)
    discriminator = comm.DistributedModel(discriminator)

    #select scheduler
    if pargs.lr_schedule_generator:
        g_scheduler = ph.get_lr_schedule(pargs.start_lr_generator, pargs.lr_schedule_generator, g_opt, last_step = start_step)
    if pargs.lr_schedule_discriminator:
        d_scheduler = ph.get_lr_schedule(pargs.start_lr_discriminator, pargs.lr_schedule_discriminator, d_opt, last_step = start_step)
    
    #wrap the optimizer
    g_opt = comm.DistributedOptimizer(g_opt,
                                      named_parameters=generator.named_parameters(),
                                      compression_name = None,
                                      op_name = "average")
                                      
    d_opt = comm.DistributedOptimizer(d_opt,
                                      named_parameters=discriminator.named_parameters(),
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
    
    # set up label for real and fake data
    label_real = torch.ones ( (pargs.local_batch_size, 1) ).to(device)
    label_fake = torch.zeros( (pargs.local_batch_size, 1) ).to(device)
        
    # Train network
    if (pargs.logging_frequency > 0) and (comm.rank() == 0):
        wandb.watch(generator)
    comm.printr('{:14.4f} REPORT: starting training'.format(dt.datetime.now().timestamp()), 0)
    step = start_step
    epoch = start_epoch
    g_lr = pargs.start_lr_generator if not pargs.lr_schedule_generator else g_scheduler.get_last_lr()[0]
    d_lr = pargs.start_lr_discriminator if not pargs.lr_schedule_discriminator else d_scheduler.get_last_lr()[0]

    #init losses
    d_loss_avg = 0.
    g_loss_avg = 0.
    
    # prep for training
    generator.train()
    discriminator.train()
    while True:
        
        comm.printr('{:14.4f} REPORT: starting epoch {}'.format(dt.datetime.now().timestamp(), epoch), 0)
        g_loss_list = []
        d_loss_list = []

        for token in train_loader:

            if pargs.enable_masks:
                inputs, outputs_real, masks, filename = token
                masks = torch.unsqueeze(masks, dim=1)
            else:
                inputs, outputs_real, filename = token
                masks = None

            # un-squeeze
            inputs = torch.unsqueeze(inputs, dim=1)
            outputs_real = torch.unsqueeze(outputs_real, dim=1)

            
            # check what needs to be trained
            train_generator = True
            train_discriminator = True
            if pargs.relative_update_schedule == "static":
                train_generator = (step < pargs.generator_warmup_steps) or (step % pargs.relative_update_schedule["update_frequency_generator"] == 0)
                train_discriminator = (step >= pargs.generator_warmup_steps) and (step % pargs.relative_update_schedule["update_frequency_discriminator"] == 0)
            elif (pargs.relative_update_schedule == "adaptive") and (pargs.loss_type_gan != "Wasserstein"):
                if step < pargs.generator_warmup_steps:
                    train_generator = True
                    train_discriminator = False
                else:
                    if (d_acc_avg > pargs.relative_update_schedule["acc_max"]):
                        # discriminator is too good:
                        train_generator = True
                        train_discriminator = False
                    elif (d_acc_avg < pargs.relative_update_schedule["acc_min"]):
                        # discriminator is too bad
                        train_generator = False
                        train_discriminator = True
                    else:
                        # all good, update both
                        train_generator = True
                        train_discriminator = True
                

            # compute discriminator loss
            # Compute outputs and logits
            outputs_fake = generator(inputs)
            logits_real, prediction_real = discriminator(outputs_real)
            logits_fake, prediction_fake = discriminator(outputs_fake)
            d_loss = criterion_gan.d_loss(logits_real, logits_fake)
            # In the Wasserstein case, we need to compute the gradient penalty as well
            if pargs.loss_type_gan == "Wasserstein":
                gp_loss = dxg.gradient_penalty(discriminator, outputs_fake, outputs_real)
                gp_loss_avg = comm.metric_average(gp_loss, "train_loss_discriminator_gp", device=device)
                d_loss += pargs.loss_weight_gp * gp_loss
            # rescale with the gan loss weight to maintain correlation between generator and discriminator
            d_loss *= pargs.loss_weight_gan
            # average the loss
            d_loss_avg = comm.metric_average(d_loss, "train_loss_discriminator_total", device=device)
            d_loss_list.append(d_loss_avg)

            # in the Min Max case, compute accuracy
            if pargs.loss_type_gan == "ModifiedMinMax":
                d_acc = 0.5 * (metrics.accuracy(prediction_real, label_real) + metrics.accuracy(prediction_fake, label_fake))
                d_acc_avg = comm.metric_average(d_acc, "train_accuracy_discriminator", device=device)

            if train_discriminator:
                # Backprop Discriminator
                d_opt.zero_grad()
                with amp.scale_loss(d_loss, d_opt) as scaled_loss:
                    scaled_loss.backward()
                d_opt.step()

                # do lr scheduler step 
                if pargs.lr_schedule_discriminator:
                    d_lr = d_scheduler.get_last_lr()[0]
                    d_scheduler.step()
                    

            #compute generator loss
            # Compute outputs and logits
            outputs_fake = generator(inputs)
            logits_fake, _ = discriminator(outputs_fake)
            gan_loss = criterion_gan.g_loss(logits_fake)
            if pargs.enable_masks:
                regression_loss = criterion_regression(outputs_fake, outputs_real, masks)
            else:
                regression_loss = criterion_regression(outputs_fake, outputs_real)
            if step < pargs.generator_warmup_steps:
                g_loss = regression_loss
            else:
                g_loss = pargs.loss_weight_gan * gan_loss + pargs.loss_weight_regression * regression_loss

            # average losses
            g_loss_avg = comm.metric_average(g_loss, "train_loss_generator_total", device=device)
            gan_loss_avg = comm.metric_average(gan_loss, "train_loss_generator_adversarial", device=device)
            reg_loss_avg = comm.metric_average(regression_loss, "train_loss_generator_regression", device=device)

            if train_generator:
                # Backprop Generator
                g_opt.zero_grad()
                with amp.scale_loss(g_loss, g_opt) as scaled_loss:
                    scaled_loss.backward()
                g_opt.step()

                # do lr scheduler step
                if pargs.lr_schedule_generator:
                    g_lr = g_scheduler.get_last_lr()[0]
                    g_scheduler.step()

            #step counter
            step += 1

            #print some metrics
            comm.printr('{:14.4f} REPORT training: step {} d_loss {} d_lr {} g_loss {} g_lr {}'.format(dt.datetime.now().timestamp(),
                                                                                                       step, d_loss_avg, d_lr,
                                                                                                       g_loss_avg, g_lr), 0)
           
            #visualize if requested
            if (step % pargs.training_visualization_frequency == 0) and (comm.rank() == 0):
                outputs_fake = generator(inputs)
                sample_idx = np.random.randint(low=0, high=outputs_real.shape[0])
                plotname = os.path.join(output_dir, "plot_train_step{}_sampleid{}.png".format(step,sample_idx))
                prediction = outputs_fake.detach()[sample_idx, 0, ...].cpu().numpy()
                groundtruth = outputs_real.detach()[sample_idx, 0, ...].cpu().numpy()
                gpviz.visualize_prediction(plotname, prediction, groundtruth)
                
                #log if requested
                if pargs.logging_frequency > 0:
                    img = Image.open(plotname)
                    wandb.log({"Train Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. CDF")]}, step = step)
            
            #log if requested
            if (pargs.logging_frequency > 0) and (step % pargs.logging_frequency == 0) and (comm.rank() == 0):
                wandb.log({"Training Loss Generator": g_loss_avg}, step = step)
                wandb.log({"Training Loss Discriminator": d_loss_avg}, step = step)
                wandb.log({"Training Loss Generator: Regression": reg_loss_avg}, step = step)
                wandb.log({"Training Loss Generator: Adversarial": gan_loss_avg}, step = step)
                wandb.log({"Current Learning Rate Generator": g_lr}, step = step)
                wandb.log({"Current Learning Rate Discriminator": d_lr}, step = step)
                if pargs.loss_type_gan == "ModifiedMinMax":
                    wandb.log({"Training Accuracy Discriminator": d_acc_avg}, step = step)
                elif pargs.loss_type_gan == "Wasserstein":
                    wandb.log({"Training Loss Discriminator: Gradient Penalty": gp_loss_avg}, step = step)
                
            # validation step if desired
            if (step % pargs.validation_frequency == 0):
                
                #eval
                generator.eval()
                discriminator.eval()
                # vali loss
                d_loss_list_val = []
                g_loss_list_val = []
                gan_loss_list_val = []
                reg_loss_list_val = []
                d_acc_list_val = []
            
                # disable gradients
                with torch.no_grad():
                
                    # iterate over validation sample
                    step_val = 0
                    for token in validation_loader:

                        if pargs.enable_masks:
                            inputs_val, outputs_real_val, masks_val, filename =	token
                        else:
                            inputs_val, outputs_real_val, filename = token
                            masks_val = None
                        
                        # un-squeeze
                        inputs_val = torch.unsqueeze(inputs_val, dim=1)
                        outputs_real_val = torch.unsqueeze(outputs_real_val, dim=1)
                        if masks_val is	not None:
                            masks_val =	torch.unsqueeze(masks_val, dim=1)
                        
                        # forward pass
                        outputs_fake_val = generator(inputs_val)
                        logits_fake_val, predictions_fake_val = discriminator(outputs_fake_val)
                        logits_real_val, predictions_real_val = discriminator(outputs_real_val)
            
                        # Compute loss and average across nodes
                        # Discriminator
                        d_loss_val = criterion_gan.d_loss(logits_real_val, logits_fake_val)
                        d_loss_val *= pargs.loss_weight_gan
                        d_loss_list_val.append(d_loss_val)

                        # in the Min Max case, compute accuracy
                        if pargs.loss_type_gan == "ModifiedMinMax":
                            d_acc_val = 0.5 * (metrics.accuracy(predictions_real_val, label_real) + metrics.accuracy(predictions_fake_val, label_fake))
                            d_acc_list_val.append(d_acc_val)
                                                
                        # Generator
                        gan_loss_val = criterion_gan.g_loss(logits_fake_val)
                        if pargs.enable_masks:
                            regression_loss_val = criterion_regression(outputs_fake_val, outputs_real_val, masks_val)
                        else:
                            regression_loss_val = criterion_regression(outputs_fake_val, outputs_real_val)
                        if step < pargs.generator_warmup_steps:
                            g_loss_val = regression_loss_val
                        else:
                            g_loss_val = pargs.loss_weight_gan * gan_loss_val + pargs.loss_weight_regression * regression_loss_val
                        g_loss_list_val.append(g_loss_val)
                        gan_loss_list_val.append(gan_loss_val)
                        reg_loss_list_val.append(regression_loss_val)
                        
                        # visualize the last sample if requested
                        if (step_val % pargs.validation_visualization_frequency == 0) and (comm.rank() == 0):
                            outputs_fake_val = generator(inputs_val)
                            sample_idx = np.random.randint(low=0, high=outputs_real_val.shape[0])
                            plotname = os.path.join(output_dir, "plot_validation_step{}_valstep{}_sampleid{}.png".format(step,step_val,sample_idx))
                            prediction_val = outputs_fake_val.detach()[sample_idx, 0, ...].cpu().numpy()
                            groundtruth_val = outputs_real_val.detach()[sample_idx, 0, ...].cpu().numpy()
                            gpviz.visualize_prediction(plotname, prediction_val, groundtruth_val)
                            
                            #log if requested
                            if pargs.logging_frequency > 0:
                                img = Image.open(plotname)
                                wandb.log({"Validation Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. CDF")]}, step = step)
                
                # average the validation loss
                count_val = float(len(d_loss_list_val))
                count_val_global = comm.metric_average(count_val, "val_count", op_name="sum", device=device)
                # Discriminator
                # total
                d_loss_val = sum(d_loss_list_val)
                d_loss_val_global = comm.metric_average(d_loss_val, "d_val_loss", op_name="sum", device=device)
                d_loss_val_avg = d_loss_val_global / count_val_global
                # accuracy
                if pargs.loss_type_gan == "ModifiedMinMax":
                    d_acc_val = sum(d_acc_list_val)
                    d_acc_val_global = comm.metric_average(d_acc_val, "d_val_acc", op_name="sum", device=device)
                    d_acc_val_avg = d_acc_val_global / count_val_global
                # generator
                #total
                g_loss_val = sum(g_loss_list_val)
                g_loss_val_global = comm.metric_average(g_loss_val, "g_val_loss", op_name="sum", device=device)
                g_loss_val_avg = g_loss_val_global / count_val_global
                #regression
                reg_loss_val = sum(reg_loss_list_val)
                reg_loss_val_global = comm.metric_average(reg_loss_val, "reg_val_loss", op_name="sum", device=device)
                reg_loss_val_avg = reg_loss_val_global / count_val_global
                #adversarial
                gan_loss_val = sum(gan_loss_list_val)
                gan_loss_val_global = comm.metric_average(gan_loss_val, "gan_val_loss", op_name="sum", device=device)
                gan_loss_val_avg = gan_loss_val_global / count_val_global
                
                # print results
                comm.printr('{:14.4f} REPORT validation: step {} d_loss {} g_loss {}'.format(dt.datetime.now().timestamp(), step, d_loss_val_avg, g_loss_val_avg), 0)
            
                # log in wandb
                if (pargs.logging_frequency > 0) and (comm.rank() == 0):
                    wandb.log({"Validation Loss Discriminator": d_loss_val_avg}, step=step)
                    wandb.log({"Validation Loss Generator": g_loss_val_avg}, step=step)
                    wandb.log({"Validation Loss Generator: Regression": reg_loss_val_avg}, step=step)
                    wandb.log({"Validation Loss Generator: Adversarial": gan_loss_val_avg}, step=step)
                    if pargs.loss_type_gan == "ModifiedMinMax":
                        wandb.log({"Validation Accuracy Discriminator": d_acc_val_avg}, step=step)
            
                # set back to train
                generator.train()
                discriminator.train()
            
            #save model if desired
            if (step % pargs.save_frequency == 0) and (comm.rank() == 0):
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_opt': g_opt.state_dict(),
                    'd_opt': d_opt.state_dict(),
                    'amp': amp.state_dict()
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
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict(),
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
    AP.add_argument("--optimizer_generator", type=str, default="Adam", choices=["Adam", "AdamW"], help="Optimizer to use for updating the generator")
    AP.add_argument("--optimizer_discriminator", type=str, default="Adam", choices=["Adam", "AdamW"], help="Optimizer to use for updating the discriminator")
    AP.add_argument("--start_lr_generator", type=float, default=1e-3, help="Start LR for generator")
    AP.add_argument("--start_lr_discriminator", type=float, default=1e-3, help="Start LR for discriminator")
    AP.add_argument("--generator_warmup_steps", type=int, default=0, help="Number of steps to just train the generator without discriminator as regressor")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    AP.add_argument("--loss_type_gan", type=str, default="ModifiedMinMax", choices=["ModifiedMinMax", "Wasserstein"], help="Loss type for adversarial part")
    AP.add_argument("--loss_type_regression", type=str, default="l1", choices=["l1", "smooth_l1", "l2"], help="Loss type for regression part")
    AP.add_argument("--loss_weight_gan", type=float, default=1., help="Weight for adversarial component for generator loss")
    AP.add_argument("--loss_weight_regression", type=float, default=1., help="Weight for regression component for generator loss")
    AP.add_argument("--loss_weight_gp", type=float, default=10., help="Weight for gradient penalty (GP) component for critic loss (denoted by lambda in the literature)")
    AP.add_argument("--lr_schedule_generator", action=StoreDictKeyPair)
    AP.add_argument("--lr_schedule_discriminator", action=StoreDictKeyPair)
    AP.add_argument("--relative_update_schedule", action=StoreDictKeyPair)
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--disable_gds", action='store_true')
    AP.add_argument("--disable_batchnorm", action='store_true')
    AP.add_argument("--amp_opt_level", type=str, default="O0", help="AMP optimization level")
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
