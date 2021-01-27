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

# hpo stuff
import ray
from ray import tune
from ray.tune import Trainable, run, sample_from
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.integration.wandb import WandbLogger

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


def train_wrapper(config, checkpoint_dir=None):

    # initialize model
    r3d = Regression3d(config)

    # train
    r3d.train()

    # return min loss:
    tune.report(validation_loss = min(r3d.validation_losses))

    
# main function
def main(pargs):
    
    if pargs.config_file is not None:
        config = yp.yparams(pargs.config_file, "default").to_dict()
        if "channels" not in config:
            config["channels"] = list(range(0,45))
        config["checkpoint"] = pargs.checkpoint

        if "use_augmentation" not in config:
            config["use_augmentation"] = False
    else:
        config = hyperparameter_defaults

    # add run tag if requested
    if pargs.run_tag is not None:
        config["run_tag"] = pargs.run_tag

    if pargs.group_tag is not None:
        config["group_tag"] = pargs.group_tag

    # get wandb api token
    with open("/certs/.wandbirc_gpsro") as f:
        wbtoken = f.readlines()[0].replace("\n","")

    # add wandb info
    config["wandb"] = {"project": "GPSRO bias correction",
                       "api_key": wbtoken,
                       "monitor_gym": False}
    
    # init ray
    ray.init()

    # override config
    tune_config = {'local_batch_size':  hp.choice('local_batch_size', config['local_batch_size']),
                   'start_lr': hp.loguniform('start_lr', np.log(config['start_lr']['min']), np.log(config['start_lr']['max'])),
                   'weight_decay': hp.loguniform('weight_decay', np.log(config['weight_decay']['min']), np.log(config['weight_decay']['max'])),
                   'layer_normalization': hp.choice('layer_normalization', ["instance_norm", "batch_norm"]),
                   'lr_schedule': hp.choice('lr_schedule', config['lr_schedule'])
    }

    # update config:
    for key in tune_config.keys():
        config[key] = tune_config[key]

    tune_kwargs = {'num_samples': config["num_trials"],
                   'config': config}

    current_best_params = [{"local_batch_size": 1,
                            "start_lr": 0.00303,
                            'lr_schedule': 2,
                            "weight_decay": 0.01,
                            'layer_normalization': 1}]

    # create scheduler and search
    #scheduler = AsyncHyperBandScheduler()
    algo = HyperOptSearch(config, points_to_evaluate = current_best_params, metric='validation_loss', mode='min')
    algo = ConcurrencyLimiter(algo, max_concurrent=config["max_concurrent_trials"])

    # run the training
    tune.run(train_wrapper,
             #loggers=[WandbLogger],
             resources_per_trial={'gpu': 1},
             num_samples=config["num_trials"],
             search_alg=algo)

    # goodbye
    ray.shutdown()
    

if __name__ == "__main__":

    #arguments
    AP = ap.ArgumentParser()
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--config_file", type=str, default=None, help="YAML file to read config data from. If none specified, use WandB")
    AP.add_argument("--run_tag", type=str, default=None, help="A tag to identify the run")
    AP.add_argument("--group_tag", type=str, default=None, help="A tag to group runs")
    pargs, _ = AP.parse_known_args()
    
    #run the stuff
    main(pargs)
