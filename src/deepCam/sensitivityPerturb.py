# Basics
import os
import numpy as np

# Torch
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom
from utils import utils
from utils import losses
from data import cam_dataset
from architecture import deeplab_xception
from utils import visualize
from sklearn.metrics import confusion_matrix

import wandb
import sys, getopt # for IO

NOISECHANNEL = 1
NOISE_MEAN = 0.0
ZERO_OUT = False
PERTURB = False

fullCmdArguments = sys.argv

# - further arguments
argumentList = fullCmdArguments[1:]

unixOptions = "zpc:"
gnuOptions = ["zero-out", "perturb", "channel="]

try:
    arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

for currentArgument, currentValue in arguments:
    if currentArgument in ("-z", "--zero-out"):
        ZERO_OUT = True
    elif currentArgument in ("-p", "--perturb"):
        PERTURB = True
    elif currentArgument in ("-c", "--channel"):
        NOISECHANNEL = int(currentValue)
    
if (ZERO_OUT):
    add_on = "ZERO-OUT"
if (PERTURB):
    add_on = "PERTURB_CH_" + str(NOISECHANNEL)
    
wandb.init(project="sensitivity", name="64k_40p_PERTURB_" + add_on)


# parameters fro prediction
# model_path = "/global/homes/e/ekarais/deepCam.git/models/allhist_16k_60p.pth"
model_path = "/global/homes/a/andregr/deepCam/models/allhist_64k_40p.pth"
# model_path = "/global/homes/a/andregr/deepCam/models/allhist_4k_80p.pth"
data_dir = "/global/cscratch1/sd/amahesh/gb_data/All-Hist/"
channels = [0,1,2,10]
# subset_length = 16000
# train_frac = 0.6
# val_frac = 0.2
# test_frac = 0.02
batch_size = 16
start = 64540
end = 67000
ith = 1


wandb.config.noisechannel = NOISECHANNEL
wandb.config.noisemean = NOISE_MEAN

# parameters for visualization
output_dir = "/global/cscratch1/sd/lukasks/deepcam/data/visualization/allhist_64k_40p_highResolution/"

    
# Initialize run
torch.manual_seed(333)

# Define architecture
net = torch.nn.DataParallel(deeplab_xception.DeepLabv3_plus(nInputChannels=4, n_classes=3, os=16, pretrained=False))
if torch.cuda.is_available():
    print("Using GPUs")
    device = torch.device("cuda")
    net.load_state_dict(torch.load(model_path))
else:
    print("Using CPUs")
    device = torch.device("cpu")
    net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
net.to(device)

for j in range(0,4):
    
    # Get data
    data = cam_dataset.subset(data_dir, channels, start, end, ith)
    data_loader = DataLoader(data, batch_size, num_workers=8, drop_last=True)

    i = 0
    sum_iou = 0
    sum_iou_pert = 0
    confMatrixSum = np.zeros((3,3))
    for inputs, labels, source in data_loader:

        # Push data on GPU and pass forward
        inputs, labels = Variable(inputs, requires_grad=True).to(device), Variable(labels).to(device)
        perturbedInputs = inputs.clone()
        noiseShape = inputs.shape
        noiseShape = noiseShape[1:]
        
        if(PERTURB):
            gaussian = torch.distributions.Normal(torch.tensor([NOISE_MEAN]), torch.tensor([float(10**j)]))
            randomNum = gaussian.sample(sample_shape=torch.Size([16, 1, 768, 1152]))
            randomNum2 = torch.squeeze(randomNum)
            perturbedInputs[:,NOISECHANNEL,:,:] = perturbedInputs[:, NOISECHANNEL, :,:] + randomNum2.to(device)
            
        if(ZERO_OUT):
            perturbedInputs[:,j,:,:] = 0

        with torch.no_grad():
            outputsZero = net.forward(perturbedInputs)

        # Calculate test IoU
        predictionsZero = torch.max(outputsZero, 1)[1]
        iouZero = utils.get_iou(predictionsZero, labels, n_classes=3) / batch_size
        sum_iou_pert += iouZero
        
        confMatrix = confusion_matrix(torch.flatten(labels.cpu()), torch.flatten(predictionsZero.cpu()))
        print("Confusion matrix sum: ")
        confMatrixSum = confMatrixSum + confMatrix
        print(confMatrixSum)
        print("batch IoU with channel " + str(NOISECHANNEL) + " perturbed: " + str(iouZero))
        
        print("length is " + str(len(source)))        
        i+=1


    avg_iou_pert = sum_iou_pert/i
    print(f"Step {j}")
    if(PERTURB):
        wandb.log({"Batch IoU with Noise": avg_iou_pert},step=j)
    if(ZERO_OUT):
        wandb.log({"Batch IoU with Zero-out": avg_iou_pert},step=j)
    
    
#    from sklearn.metrics import confusion_matrix
#   import seaborn as sn
#    import pandas as pd
#    import matplotlib.pyplot as plt

#    confusion
