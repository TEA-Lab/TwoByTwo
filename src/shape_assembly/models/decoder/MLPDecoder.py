import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
import os
import sys
import copy
import math
import numpy as np


class MLPDecoder(nn.Module):
    def __init__(self, feat_dim, num_points):
        super().__init__()
        self.np = num_points
        self.fc_layers = nn.Sequential(
            nn.Linear(feat_dim * 2, num_points * 2),
            nn.BatchNorm1d(num_points * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(num_points * 2, num_points * 3),
        )

    def forward(self, x):
        # x.shape: (bs,1024)
        batch_size = x.shape[0]
        f = self.fc_layers(x)
        return f.reshape(batch_size, self.np, 3)