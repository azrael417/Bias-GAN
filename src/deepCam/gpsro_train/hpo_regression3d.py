# Basics
import os
import sys
import wandb
import numpy as np
import argparse as ap
import datetime as dt
import subprocess as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import yparams as yp

# the training module
from regression3d_module import Regression3d

hyperparameter_defaults = dict(
    max_intra_threads = 1,
    num_workers = 1,
    model_prefix = "regressor",
    enable_amp = True,
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

# main function
def main(pargs):
    
    if pargs.config_file is not None:
        config = yp.yparams(pargs.config_file, "default").to_dict()
        if "channels" not in config:
            config["channels"] = list(range(0,45))
        config["checkpoint"] = pargs.checkpoint
    else:
        config = hyperparameter_defaults
        
    # initialize model
    r3d = Regression3d(config)
    
    # train
    r3d.train()    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--config_file", type=str, default=None, help="YAML file to read config data from. If none specified, use WandB")
    pargs, _ = AP.parse_known_args()
    
    #run the stuff
    main(pargs)
