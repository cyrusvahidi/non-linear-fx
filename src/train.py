from __future__ import print_function

import torch

import time
from tqdm import range

import Net from network

params = yaml.safe_load(open("config.yaml"))
learns_params = params['lr']

# net = Net()
# net = net.cuda()

opt =  torch.optim.Adam(filter( lambda p: p.requires_grad, net.parameters()), lr=learn_params['lr'])

