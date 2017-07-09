from torch.optim.optimizer import Optimizer, required
import copy
from torch.autograd import Variable
import numpy as np
import torch
from torch import optim


class SVRG(Optimizer):
    def __init__(self, model, layer=None, lr=1e-6):
        defaults = dict(lr=lr)
        if layer is None:
            params = model.parameters()
        else:
            params = model._modules[layer].parameters()
            
        super(SVRG, self).__init__(params, defaults)
        self.model_outer = copy.deepcopy(model)
    
        for group in self.param_groups:
            if layer is not None:
                group['params_outer'] = list(self.model_outer._modules[layer].parameters())
            else:
                group['params_outer'] = list(self.model_outer.parameters())

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)
 
    def step(self, xb, yb, verbose=False, loss_fn=None, n_samples=None):
        """Performs a single optimization step.
 
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.model_outer.zero_grad()
        y_pred = self.model_outer(xb)
        loss = loss_fn(y_pred, yb)
        loss.backward()


        for group in self.param_groups:
 
            for p_inner, p_outer in zip(group['params'], group['params_outer']):
                if p_inner.grad is None:
                    continue
    
                state = self.state[p_outer]
                assert 'full_grad' in state # should be non-empty
                g_inner = p_inner.grad.data
                g_outer = p_outer.grad.data
                c = -1.

                factor = 1. / float(xb.size()[0])
                #factor = 1.
                v = g_inner * factor + c * (g_outer * factor - state['full_grad'])

                if True:
                    print('v.norm = %.3f, fg.norm = %.3f' % 
                          (v.norm(), state['full_grad'].norm()))
                    if state['full_grad'].norm() < 1e-12:
                        import ipdb; ipdb.set_trace()

                p_inner.data.add_(-group['lr'], v)

        

        return loss
 
    def update_model_outer(self, data_loader, loss_fn=None):
        """
                model.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        return loss
        """
        # COPY PARAMS
        NORM = 0.
        GNORM = 0.

        for group in self.param_groups:
            for p_inner, p_outer in zip(group['params'], group['params_outer']):
                #if p.grad is None:
                #    continue
                p_outer.data.copy_(p_inner.data)

        for p in self.model_outer.parameters():
            NORM += p.norm()

        num_samples = data_loader.sampler.num_samples
        

        for bi, (xb, yb) in enumerate(data_loader):
            if torch.cuda.is_available():
                xb = xb.cuda()
                yb = yb.cuda()

            xb, yb = Variable(xb), Variable(yb)
            # forward, backward -> grad

            self.model_outer.zero_grad()
            y_pred = self.model_outer(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
 
            for group in self.param_groups:
                for p in group['params_outer']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    if 'full_grad' not in state:
                        state['full_grad'] = grad.new().resize_as_(grad).zero_()
                    if bi == 0:
                        state['full_grad'].zero_()

                    state['full_grad'].add_(1/float(num_samples), grad)
                    #state['full_grad'] = grad / float(num_samples)

                    GNORM += state['full_grad'].norm()
            print "bi %d" %bi

        print "P Inner: %.3f" % p_inner.norm().data[0]
        print "WEIGHT NORM: %.3f" % NORM.data[0]

        print "GRAD NORM: %.3f" % GNORM
        print "LOSS: %.3f" % loss.data[0]


OPT_DICT = {"adadelta":lambda model, lr: optim.Adadelta(model.parameters(), lr=lr),
            "adam":lambda model, lr: optim.Adam(model.parameters(), lr=lr),
            "svrg":lambda model, lr: SVRG(model, lr=lr),
            "sgd":lambda model, lr: optim.SGD(model.parameters(), lr=lr),
            "rprop":lambda model, lr: optim.Rprop(model.parameters(), lr=lr)}

