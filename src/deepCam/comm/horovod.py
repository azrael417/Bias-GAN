import numpy as np
import torch
import torch.optim as optim

import horovod.torch as hvd

#torch helpers
class comm(object):
    
    def metric_average(self, val, name, op_name=None, device=None):
        op = None
        if op_name == "average":
            op = hvd.Average
        elif op_name == "sum":
            op = hvd.Sum

        if isinstance(val, torch.Tensor):
            tensor = val.clone().detach().requires_grad_(False)
        else:
            tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name, op=op)
        return avg_tensor.item()

    def printr(self, msg, rank=0):
        if hvd.rank() == rank:
            print(msg)

    def __init__(self):
        hvd.init()

    def size(self):
        return hvd.size()
    
    def rank(self):
        return hvd.rank()

    def local_rank(self):
        return hvd.local_rank()

    def broadcast(self, tensor, root_rank, name = None):
        return hvd.broadcast(tensor, root_rank = root_rank, name = name)

    def broadcast_parameters(self, state_dict, root_rank):
        hvd.broadcast_parameters(state_dict, root_rank = root_rank)

    def broadcast_optimizer_state(self, optimizer, root_rank):
        hvd.broadcast_optimizer_state(optimizer, root_rank = root_rank)

    def init_training_state(self, model, optimizer, checkpoint_name, device_id):
        #restart from checkpoint if desired
        if (self.rank() == 0) and checkpoint_name:
            checkpoint = torch.load(checkpoint_name, map_location = device_id)
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['model'])
            amp.load_state_dict(checkpoint['amp'])
        else:
            start_step = 0
            start_epoch = 0
            
        #broadcast model and optimizer state
        steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False)
        steptens = self.broadcast(steptens, root_rank = 0, name="step")
        self.broadcast_parameters(model.state_dict(), root_rank = 0)
        self.broadcast_optimizer_state(optimizer, root_rank = 0)
        
        #extract broadcasted steps
        start_step = int(steptens.numpy()[0])
        start_epoch = int(steptens.numpy()[1])

        #we need to return the steps because they are immutable
        return start_step, start_epoch
        
    def init_gan_training_state(self, gmodel, dmodel, gopt, dopt, checkpoint_name, device_id):
        #restart from checkpoint if desired
        if (self.rank() == 0) and checkpoint_name:
            checkpoint = torch.load(checkpoint_name, map_location = device_id)
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
            gopt.load_state_dict(checkpoint['g_opt'])
            gmodel.load_state_dict(checkpoint['generator'])
            dopt.load_state_dict(checkpoint['d_opt'])
            dmodel.load_state_dict(checkpoint['discriminator'])
            amp.load_state_dict(checkpoint['amp'])
        else:
            start_step = 0
            start_epoch = 0
            
        #broadcast model and optimizer state
        steptens = torch.tensor(np.array([start_step, start_epoch]), requires_grad=False)
        steptens = self.broadcast(steptens, root_rank = 0, name="step")
        self.broadcast_parameters(gmodel.state_dict(), root_rank = 0)
        self.broadcast_parameters(dmodel.state_dict(), root_rank = 0)
        self.broadcast_optimizer_state(gopt, root_rank = 0)
        self.broadcast_optimizer_state(dopt, root_rank = 0)
        
        #extract broadcasted steps
        start_step = int(steptens.numpy()[0])
        start_epoch = int(steptens.numpy()[1])

        #we need to return the steps because they are immutable
        return start_step, start_epoch

    
    #in case of horovod, this is just a no-op
    def DistributedModel(self, model):
        return model
    
        
    def DistributedOptimizer(self, optimizer, named_parameters, compression_name, op_name):
        op = None
        if op_name == "average":
            op = hvd.Average
        if op_name == "sum":
            op = hvd.Sum

        if compression_name is None:
            compression = hvd.Compression.none
        
        return hvd.DistributedOptimizer(optimizer,
                                        named_parameters = named_parameters,
                                        compression = compression,
                                        op = op)
