import re
import numpy as np
import torch
import torch.optim as optim
import apex.optimizers as aoptim


def get_optimizer(parameters, optimizer_name, start_lr, adam_eps, weight_decay):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(parameters, lr=start_lr, eps=adam_eps, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(parameters, lr=start_lr, eps=adam_eps, weight_decay=weight_decay)
    elif optimizer_name == "LAMB":
        optimizer = aoptim.FusedLAMB(parameters, lr=start_lr, eps=adam_eps, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Error, optimizer {} not supported".format(optimizer_name))

    #add start LR to optimizer dict:
    optimizer.param_groups[0]["initial_lr"] = start_lr

    #return the optimizer
    return optimizer


def get_lr_schedule(start_lr, scheduler_arg, optimizer, last_step = -1):
    #the scheduler init is off by one, so step 0 is -1
    init_step = last_step if last_step > 0 else -1
    
    #now check
    if scheduler_arg["type"] == "multistep":
        if isinstance(scheduler_arg["milestones"], str):
            milestones = [ int(x) for x in scheduler_arg["milestones"].split() ]
        elif isinstance(scheduler_arg["milestones"], list):
            milestones = [int(x) for x in scheduler_arg["milestones"]]
        else:
            raise NotImplementedError("milestones variable has to be either a string or a list")
        gamma = float(scheduler_arg["decay_rate"])
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=gamma, last_epoch = init_step)
    elif scheduler_arg["type"] == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = scheduler_arg["t_max"],
                                                    eta_min = scheduler_arg["eta_min"], last_epoch = init_step)
    else:
        raise ValueError("Error, scheduler type {} not supported.".format(scheduler_arg["type"]))
