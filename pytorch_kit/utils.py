import torch
from torch import nn
import torch.utils.data as data_utils
import numpy as np 
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler

def get_target_format(y, problem_type="classification"):

    if y.ndim == 2 and problem_type == "classification":
        y = np.argmax(y, axis=1)
        y = torch.LongTensor(y)

    elif y.ndim == 1 and problem_type == "classification":
        #y = ut.to_categorical(y)

        y = torch.LongTensor(y)

    else:
        y  = torch.FloatTensor(y)

    return y

def isVar(x):
    return isinstance(x, Variable)

def isNumpy(x):
    return isinstance(x, np.ndarray)


def numpy2var(*xList):
    rList = []
    for x in xList:
        if isinstance(x, list):
            x = np.array(x)
            
        r = Variable(torch.FloatTensor(x))
        
        if torch.cuda.is_available():
            r = r.cuda()

        rList += [r]

    if len(rList) == 1:
        return rList[0]
        
    return tuple(rList)

def get_numpy(matrix):
    if torch.cuda.is_available():
        return matrix.data.cpu().numpy()
    else:
        return matrix.data.numpy()

def get_data_loader(X, y, batch_size, 
                    problem_type="classification"):
    batch_size = min(X.shape[0], batch_size)
    X = torch.FloatTensor(X)

    y = get_target_format(y, problem_type=problem_type)

    data = data_utils.TensorDataset(X, y)

    # SAMPLE WITHOUT REPLACEMENT AND KEEP ALL BATCH SIZES THE SAME
    data_loader = data_utils.DataLoader(data, 
                                        batch_size=batch_size,
                                        sampler=RandomSampler(data),
                                        drop_last=True)

    return data_loader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_operations(model, LayerList):
    """
    Calculate the number of flops for given a string information of layer.
    We extract only resonable numbers and use them.
    
    Args:
        layer (str) : example
            Linear (512 -> 1000)
            Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    """
    #print(layer)
    flops = 0
    for layer_name in LayerList.values():
        layer = model._modules[layer_name]

        flops = 1
        import pdb; pdb.set_trace()  # breakpoint 707d18b3 //
        
        if isinstance(layer, nn.Linear):
            C1 = int(params[0])
            C2 = int(params[1])
            flops = C1*C2
            
        elif isinstance(layer, nn.Conv2d):
            C1 = int(layer.in_channels)
            C2 = int(layer.out_channels)
            K1, K2 = layer.kernel_size
            
            # image size
            H = 32
            W = 32
            flops = C1*C2*K1*K2*H*W
        
    #     print(type_name, flops)
        return flops



def get_flattenSize(n_channels, n_rows, n_cols, n_pools):
    return np.prod([ n_rows / (n_pools*2), 
                     n_cols / (n_pools*2), 
                     n_channels])