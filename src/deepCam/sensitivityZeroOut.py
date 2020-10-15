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

# Get data
data = cam_dataset.subset(data_dir, channels, start, end, ith)
data_loader = DataLoader(data, batch_size, num_workers=8, drop_last=True)
# train, val, test = cam_dataset.split(data_dir, channels, subset_length, train_frac, val_frac, test_frac)
# data_loader = DataLoader(test, batch_size, num_workers=8, drop_last=True)
ZEROCHANNEL = 2
for inputs, labels, source in data_loader:
            
    # Push data on GPU and pass forward
    inputs, labels = Variable(inputs, requires_grad=True).to(device), Variable(labels).to(device)
    zeroedOutInputs = inputs.clone()
    zeroedOutInputs[ZEROCHANNEL,:,:,:] = 0
    
    with torch.no_grad():
        outputs = net.forward(inputs)
        outputsZero = net.forward(zeroedOutInputs)

    # Calculate test IoU
    predictions = torch.max(outputs, 1)[1]
    predictionsZero = torch.max(outputsZero, 1)[1]
    iou = utils.get_iou(predictions, labels, n_classes=3) / batch_size
    iouZero = utils.get_iou(predictionsZero, labels, n_classes=3) / batch_size
    print("batch IoU: " + str(iou))
    print("batch IoU with channel " + str(ZEROCHANNEL) + " zeroed out: " + str(iouZero))
    print("length is " + str(len(source)))
    """
    for i in range(0,len(source)):
        print("visualizing " + source[i])
        
        visualize.generate_PNGs(prediction=predictions[i].cpu(),
                                h5_file_path=source[i],
                                output_dir=output_dir+"predict/",
                                png_name=os.path.splitext(os.path.basename(source[i]))[0],
                                target="predict")
        visualize.generate_PNGs(prediction=predictions[i].cpu(),
                                h5_file_path=source[i],
                                output_dir=output_dir+"true/",
                                png_name=os.path.splitext(os.path.basename(source[i]))[0],
                                target="true")
        """
#    from sklearn.metrics import confusion_matrix
#   import seaborn as sn
#    import pandas as pd
#    import matplotlib.pyplot as plt

#    confusion
