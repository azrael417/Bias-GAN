import os
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

#torch helpers
class comm(object):
    
    def metric_average(self, val, name, op_name=None, device=None):
        if dist.is_available() and dist.is_initialized():
            reduce_op = dist.ReduceOp.SUM
            fact = 1.
            if op_name == "average":
                fact = 1. / float(self.size())

            if isinstance(val, torch.Tensor):
                tensor = val.clone().detach().requires_grad_(False)
            else:
                tensor = torch.tensor(val)
            
            if device is not None:
                tensor = tensor.to(device)
            
            dist.all_reduce(tensor, op = reduce_op)
        
            return fact * tensor.item()
        else:
            if isinstance(val, torch.Tensor):
                return val.item()
            else:
                return val

    
    def printr(self, msg, rank=0):
        if self.rank() == rank:
            print(msg)

            
    def __init__(self, mode="openmpi"):
        #default pytorch port
        port = "29500"
        os.environ["MASTER_PORT"] = port

        if mode == "openmpi":
            dist.init_process_group(backend = "mpi")
            return
            
        elif (mode == "openmpi-nccl"):
            #use pmix server address: only works for single node
            addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
            address = addrport.split(":")[0]
            os.environ["MASTER_ADDR"] = address
            comm_rank = os.getenv('OMPI_COMM_WORLD_RANK',0)
            comm_size = os.getenv("OMPI_COMM_WORLD_SIZE",0)
        elif mode == "dummy":
            os.environ["MASTER_ADDR"] = "localhost"
            comm_rank = 0
            comm_size = 1

        if mode != "dummy":
            dist.init_process_group(backend = "nccl",
                                    rank = comm_rank,
                                    world_size = comm_size)
        
    def size(self):
        """
        Gets size of communicator
        """
        if dist.is_available() and dist.is_initialized():
            size = dist.get_world_size()
        else:
            size = 1
        return size

    
    def rank(self):
        """
        Gets distributed rank or returns zero if distributed is not initialized.
        """
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        return rank

    
    def local_rank(self):
        """
        Gets node local rank or returns zero if distributed is not initialized.
        """
        #number of GPUs per node
        if dist.is_available() and dist.is_initialized() and torch.cuda.is_available():
            local_rank = dist.get_rank() % torch.cuda.device_count()
        else:
            local_rank = 0

        return local_rank

    
    def broadcast(self, tensor, root_rank, name = None):
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(tensor, src = root_rank)
        return tensor

    
    def init_training_state(self, model, optimizer, checkpoint_name, device_id):
        #restart from checkpoint if desired
        if (checkpoint_name is not None) and (os.path.isfile(checkpoint_name)):
            checkpoint = torch.load(checkpoint_name, map_location = device_id)
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

            # we need to do some key hacking for the model dict
            model_dict = {}
            for k in checkpoint['model']:
                model_dict[k.replace("module.","")] = checkpoint['model'][k]
            model.load_state_dict(model_dict)
        else:
            start_step = 0
            start_epoch = 0
            
        #we need to return the steps because they are immutable
        return start_step, start_epoch
        
        
    def init_gan_training_state(self, gmodel, dmodel, gopt, dopt, checkpoint_name, device_id):
        #restart from checkpoint if desired
        if (checkpoint_name is not None) and (os.path.isfile(checkpoint_name)):
            checkpoint = torch.load(checkpoint_name, map_location = device_id)
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']

            # Optimizer
            gopt.load_state_dict(checkpoint['g_opt'])
            dopt.load_state_dict(checkpoint['d_opt'])
            
            # we need to tweak the keys a bit:
            # generator
            model = {}
            for k in checkpoint['generator']:
                model[k.replace("module.","")] = checkpoint['generator'][k]
            gmodel.load_state_dict(model)
            # discriminator
            model = {}
            for	k in checkpoint['discriminator']:
                model[k.replace("module.","")] = checkpoint['discriminator'][k]
            dmodel.load_state_dict(model)
        else:
            start_step = 0
            start_epoch = 0
            
        #we need to return the steps because they are immutable
        return start_step, start_epoch
    
    
    def init_aae_training_state(self, emodel, gmodel, dmodel, gopt, dopt, checkpoint_name, device_id):
        #restart from checkpoint if desired
        if (checkpoint_name is not None) and (os.path.isfile(checkpoint_name)):
            checkpoint = torch.load(checkpoint_name, map_location = device_id)
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']

            # Optimizer
            gopt.load_state_dict(checkpoint['g_opt'])
            dopt.load_state_dict(checkpoint['d_opt'])
            
            # we need to tweak the keys a bit:
            # encoder
            model = {}
            for k in checkpoint['encoder']:
                model[k.replace("module.","")] = checkpoint['encoder'][k]
            emodel.load_state_dict(model)
            # generator
            model = {}
            for k in checkpoint['generator']:
                model[k.replace("module.","")] = checkpoint['generator'][k]
            gmodel.load_state_dict(model)
            # discriminator
            model = {}
            for	k in checkpoint['discriminator']:
                model[k.replace("module.","")] = checkpoint['discriminator'][k]
            dmodel.load_state_dict(model)
        else:
            start_step = 0
            start_epoch = 0
            
        #we need to return the steps because they are immutable
        return start_step, start_epoch
    
    
    def DistributedModel(self, model):
        if dist.is_available() and dist.is_initialized():
            return DDP(model)
        else:
            return model

    
    #this is just a no-op for pytorch distributed
    def DistributedOptimizer(self, optimizer, named_parameters, compression_name, op_name):
        return optimizer
