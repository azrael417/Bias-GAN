# Basics
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
from utils import utils
from utils import losses
from utils import parsing_helpers as ph
from data import era_dataset as era
#from architecture import deeplab_xception as dxc
from architecture import deeplab_debug as dxc
from utils import era_visualizer as ev

#vis stuff
from PIL import Image

#horovod
#import horovod.torch as hvd
from comm.horovod import comm as hvdcomm


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

    # init horovod
    comm = hvdcomm()

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
    fac = 2 if pargs.num_raid == 4 else 1
    mod = 4 if pargs.num_raid == 4 else 2
    root_dir = os.path.join(pargs.data_dir_prefix, "data{}".format( fac * (comm.local_rank() // mod) + 1 ), \
                            "ecmwf_data", "gpu{}".format( comm.local_rank() ))
    output_dir = pargs.output_dir
    if comm.rank() == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    
    # Setup WandB
    if (pargs.logging_frequency > 0) and (comm.rank() == 0):
        # get wandb api token
        with open("/root/.wandbirc") as f:
            wbtoken = f.readlines()[0].replace("\n","")
        # log in: that call can be blocking, it should be quick
        sp.call(["wandb", "login", wbtoken])
        
        #init db and get config
        resume_flag = pargs.run_tag if pargs.resume_logging else False
        wandb.init(project = 'ERA5 prediction', name = pargs.run_tag, id = pargs.run_tag, resume = resume_flag)
        config = wandb.config
    
        #set general parameters
        config.root_dir = root_dir
        config.output_dir = pargs.output_dir
        config.max_steps = pargs.max_steps
        config.local_batch_size = pargs.local_batch_size
        config.num_workers = comm.size()
        config.channels = pargs.channels
        config.noise_dimensions = pargs.noise_dimensions
        config.noise_type = pargs.noise_type
        config.optimizer = pargs.optimizer
        config.start_lr = pargs.start_lr
        config.adam_eps = pargs.adam_eps
        config.weight_decay = pargs.weight_decay
        config.loss_type = pargs.loss_type
        config.model_prefix = pargs.model_prefix
        config.precision = "fp16" if  pargs.enable_fp16 else "fp32"
        config.use_batchnorm = False if pargs.disable_batchnorm else True

        # lr schedule if applicable
        for key in pargs.lr_schedule:
            config.update({"lr_schedule_"+key: pargs.lr_schedule[key]}, allow_val_change = True)

    # Determine normalizer
    normalizer = dxc.Identity if pargs.disable_batchnorm else torch.nn.BatchNorm2d

    # Define architecture
    n_input_channels = len(pargs.channels) + pargs.noise_dimensions
    n_output_channels = len(pargs.channels)
    net = dxc.DeepLabv3_plus(nInputChannels = n_input_channels, n_output = n_output_channels, os=16, pretrained=False, normalizer=normalizer)

    #select loss
    criterion = None
    if pargs.loss_type == "l1":
        criterion = torch.nn.L1Loss()
    elif pargs.loss_type == "l2":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError("Error, {} loss not supported".format(pargs.loss_type))

    #select optimizer
    optimizer = ph.get_optimizer(net, "Adam", pargs.start_lr, pargs.adam_eps, pargs.weight_decay)

    #load net and optimizer states
    start_step, start_epoch = comm.init_training_state(net, optimizer, pargs.checkpoint, device)
    
    #upload the network to the device
    if pargs.enable_fp16:
        net.half()
    net.to(device)

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
    train_loader = era.ERADataLoader(train_dir,
                                     statsfile = os.path.join(train_dir, 'stats.npz'),
                                     channels = pargs.channels,
                                     batch_size = pargs.local_batch_size,
                                     normalization_type = "MinMax" if pargs.noise_type == "Uniform" else "MeanVariance",
                                     shuffle = True,
                                     num_inter_threads = min([pargs.local_batch_size, pargs.max_inter_threads]),
                                     num_intra_threads = pargs.max_intra_threads,
                                     read_device = torch.device("cpu") if pargs.disable_gds else device,
                                     send_device = device)
    
    # validation
    validation_dir = os.path.join(root_dir, "validation")
    validation_loader = era.ERADataLoader(validation_dir,
                                          statsfile = os.path.join(train_dir, 'stats.npz'),
                                          channels = pargs.channels,
                                          batch_size = pargs.local_batch_size,
                                          normalization_type = "MinMax" if pargs.noise_type == "Uniform" else "MeanVariance",
                                          shuffle = False,
                                          num_inter_threads = min([pargs.local_batch_size, pargs.max_inter_threads]),
                                          num_intra_threads = pargs.max_intra_threads,
                                          read_device = torch.device("cpu") if pargs.disable_gds else device,
                                          send_device = device)
    
    #Noise vector
    udist = None
    if pargs.noise_type == "Uniform":
        udist = torch.distributions.uniform.Uniform(0., 1.)
    elif pargs.noise_type == "Normal":
        udist = torch.distributions.normal.Normal(0., 1.)

        
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
        for inputs_raw, label, inputs_info, label_info in train_loader:

            #generate random sample: format of input data is NHWC 
            if pargs.noise_dimensions > 0:
                ishape = inputs_raw.shape
                inputs_noise = udist.rsample( (ishape[0], pargs.noise_dimensions, ishape[2], ishape[3]) ).to(device)  
                #concat tensors
                inputs = torch.cat([inputs_raw, inputs_noise], dim=1)
            else:
                inputs = inputs_raw
            
            # Maybe we need to use half precision
            if pargs.enable_fp16:
                inputs, label = inputs.half(), label.half()

            # forward pass
            outputs = net.forward(inputs)
            
            # Compute loss and average across nodes
            loss = criterion(outputs, label)
            loss_avg = comm.metric_average(loss, "train_loss")
            mse_list.append(loss_avg)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #step counter
            step += 1

            if pargs.lr_schedule:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

            #print some metrics
            comm.printr('{:14.4f} REPORT training: step {} loss {} LR {}'.format(dt.datetime.now().timestamp(), step, loss_avg, current_lr), 0)

            #visualize if requested
            if (step % pargs.visualization_frequency == 0) and (comm.rank() == 0):
                sample_idx = np.random.randint(low=0, high=label.shape[0])
                filename = os.path.join(output_dir, "plot_step{}_sampleid{}.png".format(step,sample_idx))
                prediction = outputs.detach()[sample_idx,...].cpu().numpy()
                groundtruth = label.detach()[sample_idx,...].cpu().numpy()
                ev.visualize_prediction(filename, prediction, groundtruth)
                
                #log if requested
                if pargs.logging_frequency > 0:
                    img = Image.open(filename)
                    wandb.log({"Examples": [wandb.Image(img, caption="Prediction vs. Ground Truth vs. Difference")]}, step = step)
            
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

                # disable gradients
                with torch.no_grad():
                
                    # iterate over validation sample
                    for inputs_raw_val, label_val, inputs_info_val, label_info_val in validation_loader:

                        # generate random sample: format of input data is NHWC 
                        if pargs.noise_dimensions > 0:
                            ishape_val = inputs_raw_val.shape
                            inputs_noise_val = udist.rsample( (ishape_val[0], pargs.noise_dimensions, ishape_val[2], ishape_val[3]) ).to(device)

                            # concat tensors
                            inputs_val = torch.cat([inputs_raw_val, inputs_noise_val], dim=1)
                        else:
                            inputs_val = inputs_raw_val
                    
                        # forward pass
                        outputs_val = net.forward(inputs_val)

                        # Compute loss and average across nodes
                        loss_val = criterion(outputs_val, label_val)

                        # append to list
                        mse_list_val.append(loss_val)

                # average the validation loss
                count_val = float(len(mse_list_val))
                count_val_global = comm.metric_average(count_val, "val_count", op_name="sum")
                loss_val = sum(mse_list_val)
                loss_val_global = comm.metric_average(loss_val, "val_loss", op_name="sum")
                loss_val_avg = loss_val_global / count_val_global
                
                # print results
                comm.printr('{:14.4f} REPORT validation: step {} loss {}'.format(dt.datetime.now().timestamp(), step, loss_val_avg), 0)

                # log in wandb
                if (pargs.logging_frequency > 0) and (comm.rank() == 0):
                    wandb.log({"Validation Loss": loss_val_avg}, step=step)

                # set to train
                net.train()
            
            #save model if desired
            if (step % pargs.save_frequency == 0) and (comm.rank() == 0):
                checkpoint = {
                    'step': step,
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict()
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
                'optimizer': optimizer.state_dict()
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
    AP.add_argument("--num_raid", type=int, default=4, choices=[4, 8], help="Number of available raid drives")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_intra_threads", type=int, default=8, help="Maximum degree of parallelism within reader")
    AP.add_argument("--max_steps", type=int, default=25000, help="Maximum number of steps to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--validation_frequency", type=int, default=100, help="Frequency with which the model is validated")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during training")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10], help="Channels used in input")
    AP.add_argument("--noise_dimensions", type=int, default=5, help="Number of additional noise dimensions")
    AP.add_argument("--noise_type", type=str, default="Uniform", choices=["Uniform", "Normal"], help="Noise type")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam"], help="Optimizer to use")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"], help="Loss type")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--lr_decay_patience", type=int, default=3, help="Minimum number of steps used to wait before decreasing LR")
    AP.add_argument("--lr_decay_rate", type=float, default=0.25, help="LR decay factor")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--disable_gds", action='store_true')
    AP.add_argument("--disable_batchnorm", action='store_true')
    AP.add_argument("--enable_fp16", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    pargs = AP.parse_args()

    #run the stuff
    main(pargs)
