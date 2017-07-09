import numpy as np 
import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable
import utils as tu

# ---------- DICE LOSS
def dice_loss(input, target):
    smooth = 0.000001

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = torch.dot(iflat, tflat)
    
    return - ((2. * intersection) /
              (iflat.sum() + tflat.sum() + 2*smooth))

def dice_loss_modified(input, target):
    smooth = 1.0
    smooth = 1e-8
    iflat = input.view(-1)
    tflat = target.view(-1)

    indices= tflat>=0
    tflat_sub = tflat[indices] 
    iflat_sub = iflat[indices] 

    intersection = torch.dot(iflat_sub, tflat_sub)

    score  = - (2. * intersection + 2*smooth)
    score = score / (iflat_sub.sum() + (tflat_sub.sum()) + 2*smooth)
    #print score

    return score

LOSS_DICT = {"categorical_crossentropy":F.nll_loss,
             "binary_crossentropy": F.binary_cross_entropy,
             "dice_loss": dice_loss,
             "dice_loss_modified": dice_loss_modified,
             "mse":torch.nn.MSELoss(size_average=True),
             "L1Loss":torch.nn.L1Loss()}
