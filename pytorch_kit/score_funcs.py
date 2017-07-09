import torch
import utils 


def accuracy(y_pred, y):    
    if utils.isNumpy(y):
        y = utils.numpy2var(y)

    if utils.isNumpy(y_pred):
        y_pred = utils.numpy2var(y_pred)

    _, y_pred = y_pred.topk(1)

    y_pred = torch.squeeze(y_pred).long()
    y = torch.squeeze(y).long()

    acc = (y_pred == y).sum().data[0]

    return acc

def mean_squared_error(y_pred, y):      
    if utils.isNumpy(y):
        y = utils.numpy2var(y)

    if utils.isNumpy(y_pred):
        y_pred = utils.numpy2var(y_pred)

    return torch.sum((y_pred - y) ** 2)


SCORE_DICT = {"acc": accuracy, "mse": mean_squared_error}
