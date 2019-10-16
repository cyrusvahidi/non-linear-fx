import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=64)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=128) 
        # define conv and FC here  
    
    def forward(self, x):
        # define operations on layers here
        # compose layers with pooling and non-linearities 
