import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 1  # random seed,  for reproduction


__C.NET = 'HR_Net'
__C.PRE_HR_WEIGHTS = '../HRNet_shanghaiB.pth'