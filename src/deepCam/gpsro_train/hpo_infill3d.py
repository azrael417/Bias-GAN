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
from infill3d_module import Infill3d

# hpo stuff
import ray
from ray import tune
from ray.tune import Trainable, run, sample_from
from ray.tune.schedulers import AsyncHyperBandScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.integration.wandb import WandbLogger

hyperparameter_defaults = dict(
    max_intra_threads = 1,
    num_workers = 1,
    model_prefix = "infill3d",
    enable_amp = True,
    checkpoint = "",
    enable_gds = False,
    save_frequency = 1000,
    logging_frequency = 50,
    validation_frequency = 100,
    training_visualization_frequency = 200,
    validation_visualization_frequency = 50,
    loss_type = "smooth-l1",
    loss_weights = {"valid": 1., "hole": 0.8, "tv": 0.0},
    channels = list(range(0,45)),
)


def train_wrapper(config, batch_size, start_lr, lr_schedule):
    # update config
    config["start_lr"] = start_lr
    config["batch_size"] = batch_size
    config["lr_schedule"] = lr_schedule
    
    # initialize model
    inf3d = Infill3d(config)
    
    # train
    inf3d.train()

    # return min loss:
    tune.report(validation_loss = min(inf3d.validation_losses))
    
    
#main function
def main(pargs):

    if pargs.config_file is not None:
        config = yp.yparams(pargs.config_file, "default").to_dict()
        if "channels" not in config:
            config["channels"] = list(range(0,45))
        config["checkpoint"] = pargs.checkpoint
    else:
        config = hyperparameter_defaults
    
    # add run tag if requested
    if pargs.run_tag is not None:
        config["run_tag"] = pargs.run_tag

    # override config
    tune_config = {'batch_size':  hp.choice('batch_size', [4, 8, 16, 32]),
                   'start_lr': hp.loguniform('start_lr', 1e-5, 1e-1),
                   'lr_schedule': hp.choice('lr_schedule', [
                       {"type": "multistep", "milestones": [5000], "decay_rate": 0.1},
                       {"type": "multistep", "milestones": [10000], "decay_rate": 0.1},
                       {"type": "cosine_annealing", "t_max": 500, "eta_min": 0},
                       {"type": "cosine_annealing", "t_max": 1000, "eta_min": 0}
                   ])
    }

    current_best_params={"batch_size": 2, "start_lr": 1e-4, 'lr_schedule': 2}
    
    # create scheduler and search
    scheduler = AsyncHyperBandScheduler()
    algo = HyperOptSearch(points_to_evaluate=current_best_params)
    
    # run the training
    tune.run(train_wrapper,
             search_alg=algo,
             scheduler=scheduler,
             metric="validation_loss",
             mode="min",
             **tune_config)
    
if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--config_file", type=str, default=None, help="YAML file to read config data from. If none specified, use WandB")
    AP.add_argument("--run_tag", type=str, default=None, help="A tag to identify the run")
    pargs, _ = AP.parse_known_args()
    
    #run the stuff
    main(pargs)
