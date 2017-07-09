import torch
import time
import utils as utt
from torch.autograd import Variable

import numpy as np
import torch.nn as nn
import loss_funcs as lf
import optimizers as opt
import score_funcs as sf

weight_dict = {0:"weight", 1:"bias"}

class BaseModel(nn.Module):
    """INSPIRED BY KERAS AND SCIKIT-LEARN API"""
    def __init__(self, 
                 problem_type="classification", 
                 loss_name="categorical_crossentropy",
                 opt_name="adadelta"):

        super(BaseModel, self).__init__()
        self.loss_name = loss_name
        self.problem_type = problem_type
        self.my_optimizer = None
        self.opt_name = opt_name
        self.gpu_started = False

    # ------------ TRAINING FUNCTIONS
    def fit(self, X, y, batch_size=20, epochs=1, 
            save_every=10, weights_name=None, loss_name=None, 
            until_loss=None, verbose=1, reset_optimizer=False,
            opt_name=None, svrg_inner=3, learning_rate=1.0):

        if loss_name is not None:
            self.loss_name = loss_name

        self.start_gpu()

        # SET MODEL TO TRAIN MODE FOR DROPOUT AND BN
        self.train()

        # INITIALIZE DATA
        n_samples = X.shape[0] 
        batch_size = min(batch_size, n_samples)
        data_loader = utt.get_data_loader(X, y, batch_size, 
                                          problem_type=self.problem_type)

        # INITIALIZE OPTIMIZER
        if (opt_name is not None) and (opt_name != self.opt_name):
            self.opt_name = opt_name
            self.my_optimizer =opt.OPT_DICT[self.opt_name](self, learning_rate)
     
        if (reset_optimizer or self.my_optimizer is None):
            self.my_optimizer =opt.OPT_DICT[self.opt_name](self, learning_rate)

        if weights_name is not None:
            try:
                self.load_weights(weights=weights_name)

                if verbose:
                    print "SUCCESS: %s loaded..." % weights_name
            except:
                print "FAILED: %s NOT loaded..." % weights_name

        for epoch in range(epochs):
            if type(self.my_optimizer).__name__.lower() == "svrg" and (epoch%svrg_inner == 0):
                self.my_optimizer.update_model_outer(data_loader,lf.LOSS_DICT[self.loss_name])

            # INITILIZE VERBOSE
            losses = utt.AverageMeter()
            accs = utt.AverageMeter()

            # INNER LOOP
            s = time.time()
            n_samples = data_loader.sampler.num_samples

            for bi, (xb, yb) in enumerate(data_loader):
                # ----- 1.GET DATA
                if torch.cuda.is_available():
                    xb = xb.cuda()
                    yb = yb.cuda()

                if "categorical_crossentropy" in self.loss_name:
                    xb, yb = Variable(xb), Variable(yb)
                else:
                    xb, yb = Variable(xb), Variable(yb).float()

                # ----- 2.OPTIMIZER
                self.my_optimizer.zero_grad()
                y_pred = self(xb)

                if loss_name is None:
                    loss =lf.LOSS_DICT[self.loss_name](y_pred, yb)
                else:
                    loss =lf.LOSS_DICT[loss_name](y_pred, yb)

                loss.backward()

                if type(self.my_optimizer).__name__.lower() == "svrg":
                    self.my_optimizer.step(xb, yb, loss_fn=lf.LOSS_DICT[self.loss_name], 
                                           n_samples=n_samples)
                else:
                    self.my_optimizer.step()

                # ----- 3.VERBOSE
                losses.update(loss.data[0], batch_size)
                stdout = ("%d/%d - [%d/%d] - Loss (%s): %f" % (epoch+1, 
                            epochs, (bi+1)*batch_size, 
                            n_samples, self.loss_name, losses.avg))

                if (self.problem_type == "classification" and 
                    self.loss_name == "categorical_crossentropy"):
                    
                    accs.update(sf.accuracy(y_pred, yb) / float(batch_size), 
                                batch_size)

                    stdout += " - Acc: %.3f" % accs.avg

                if verbose > 1:
                    print stdout

            # ------ 4. OUTER LOOP
            stdout += " - Time: %.3f sec" % (time.time() - s)
            stdout += " - Optimizer: %s" % type(self.my_optimizer).__name__
            
            # stdout += " - Operations: %d" % count_operations(self.LayerList)
            if verbose == 1:
                print ""
                print stdout
            
            # ------ 5. SAVE WEIGHTS
            if ((epoch % save_every == 0) and (epoch!=0) and 
                (weights_name is not None)):

                self.save_weights(weights_name, verbose=verbose)

            if epoch == (epochs - 1) and (weights_name is not None):
                self.save_weights(weights_name, verbose=verbose)

            # ------ 6. EARLY STOP
            if until_loss is not None:
              if until_loss > losses.avg: 
                return 

    def fit_batch(self, xb, yb, verbose=0):
        self.start_gpu()
        self.train()

        if self.my_optimizer is None:
            print "optimizer set..."
            self.set_optimizer()
        

        if not isinstance(xb, torch.FloatTensor):
            xb = torch.FloatTensor(xb)
            yb = utt.get_target_format(yb, problem_type=self.problem_type)

        # ----- 1.OPTIMIZER
        if torch.cuda.is_available():
                xb = xb.cuda()
                yb = yb.cuda()

        xb, yb = Variable(xb), Variable(yb)

        # ----- 2.OPTIMIZER
        self.my_optimizer.zero_grad()
        y_pred = self(xb)
        loss =lf.LOSS_DICT[self.loss_name](y_pred, yb)
        loss.backward()

        self.my_optimizer.step()

        loss_value = loss.data[0]
        if verbose:
            print("loss: %.3f" %  loss_value)

        return loss_value

    # ------------ PREDICTION FUNCTIONS
    def predict(self, X, batch_size=10):
        self.start_gpu()

        self.eval()
        bs = batch_size
        if X.ndim == 3 or X.ndim == 1:
            X = X[np.newaxis]
        
        if self.problem_type == "segmentation":
            y_pred = np.zeros((X.shape[0], self.n_outputs, 
                               X.shape[2], X.shape[3]))

        if self.problem_type == "regression":
            y_pred = np.zeros((X.shape[0], self.n_outputs))

        if self.problem_type == "classification":
            y_pred = np.zeros((X.shape[0], self.n_outputs))

        i = 0
        while True:
            s = i*bs
            e = min((i+1)*bs, X.shape[0])
            Xb = torch.FloatTensor(X[s:e])

            if torch.cuda.is_available():
                Xb = Xb.cuda() 

            Xb = Variable(Xb)

            y_pred[s:e] = utt.get_numpy(self(Xb))

            i += 1

            if e == X.shape[0]:
                break

        return y_pred

    def forward_pass(self, XList, numpy_result=True):
        self.start_gpu()
        self.train()

        if not isinstance(XList, list):
            XList = [XList]

        if isinstance(XList[0], np.ndarray):
            XList = utt.numpy2var(*XList)

        if not isinstance(XList, list):
            XList = [XList]   
        y_pred = self(*XList)

        if numpy_result:
            y_pred = utt.get_numpy(y_pred)
            
        return y_pred 

    # ------------ EVALUATION FUNCTIONS
    def score(self, X, y, batch_size=100, score_function=None):
        if score_function is None:
            if self.problem_type == "classification":
                # Returns accuracy
                score_function = "acc" 

            elif self.problem_type == "regression":
                # Returns mean square error
                score_function = "mse"

            elif self.problem_type == "segmentation":
                # Returns Jaccard
                score_function = "jacc"

        # DEFINE SCORE FUNCTION
        score_function = sf.SCORE_DICT[score_function]

        if utt.isNumpy(X):
            X = utt.numpy2var(X)

        n = X.size()[0]

        score = 0.
        for b in range(0, n, batch_size):
            score += score_function(self.forward_pass(X[b:b+batch_size]),
                                    y[b:b+batch_size])

        return score / float(n)

    def compute_loss(self, X, y, batch_size=1000):
        batch_size = min(batch_size, X.shape[0])
        data_loader = utt.get_data_loader(X, y, batch_size, 
                                          problem_type=self.problem_type)
        n = data_loader.sampler.num_samples

        # compute total loss
        total_loss = 0.
        for bi, (xb, yb) in enumerate(data_loader):
            if torch.cuda.is_available():
                    xb = xb.cuda()
                    yb = yb.cuda()

            xb, yb = Variable(xb), Variable(yb)

            y_pred = self(xb)
            loss =lf.LOSS_DICT[self.loss_name](y_pred, yb)

            total_loss += loss.data[0]

        avg_loss = total_loss / float(n)
        #print "Average loss: %.3f" % avg_loss

        return avg_loss

    # -------- NETWORK CONFIGURATIONS
    def start_gpu(self):
        if not self.gpu_started:
            if torch.cuda.is_available():
                print "pytorch running on GPU...."
                self.cuda()
            else:
                print "pytorch running on CPU...."

            self.gpu_started = True

    def load_weights(self, weights=None):
        self.load_state_dict(torch.load(weights+".pth"))

    def save_weights(self, weights, verbose=1):
        torch.save(self.state_dict(), '%s.pth' % weights)
        if verbose:
            print("weights %s.pth saved..." % weights)

    def reset_optimizer(self, opt_name="adadelta",learning_rate=1.0):
        self.opt_name = opt_name
        self.my_optimizer =opt.OPT_DICT[self.opt_name](self, learning_rate)

    def set_optimizer(self, opt_name="adadelta", learning_rate=1.0):
        # INITIALIZE OPTIMIZER
        self.opt_name = opt_name
        self.my_optimizer =opt.OPT_DICT[self.opt_name](self, learning_rate)

    def get_optimizer(self):
        # INITIALIZE OPTIMIZER
        return self.opt_name  

    # ----------- ACCESSING NETWORK'S INTERNALS
    def get_weights(self, layer=None, norm_only=False, verbose=0):
        weight_norms = {}
        for key_param in self._modules.keys():
            if layer is not None and layer != key_param:
                continue

            for i, param in enumerate(self._modules[key_param].parameters()):
                weight = utt.get_numpy(param)
               

                if not norm_only and verbose:
                    print("weight:", weight[:5])
                weight_norm = np.linalg.norm(weight)

                if verbose:
                    print("\nLAYER %s - WEIGHT %d" % (key_param, i+1))
                    print("\nweight norm: %.3f" % (weight_norm))
                    print("min: %.3f, mean: %.3f, max: %.3f" % (weight.min(), weight.mean(), weight.max()))

                    print("shape: %s" % (str(weight.shape)))

                weight_norms["%s_%s norm" % (key_param, weight_dict[i])] = weight_norm

        return weight_norms


    def get_gradients(self, X, y, batch_size=10, norm_only=True):
        self.start_gpu()

        data_loader = utt.get_data_loader(X, y, batch_size=10)

        for bi, (xb, yb) in enumerate(data_loader):
            print("Batch %d" % bi)

            if torch.cuda.is_available():
                xb = xb.cuda()
                yb = yb.cuda()    

            xb, yb = Variable(xb), Variable(yb)

        

            y_pred = self(xb)
            loss =lf.LOSS_DICT[self.loss_name](y_pred, yb)
            print("loss (%s): %.3f" % (self.loss_name, loss.data[0]))
          
            # Zero the gradients before running the backward pass.
            self.zero_grad()
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Variable, so
            # we can access its data and gradients like we did before.
            for key_param in self._modules.keys():

                for i, param in enumerate(self._modules[key_param].parameters()):                    
                    grad = utt.get_numpy(param.grad)
                    print("\nLAYER %s - WEIGHT %d" % (key_param, i+1))

                    if not norm_only:
                        print("grad:", grad)
                    print("gradient norm: %.3f" % (np.linalg.norm(grad)))
                    print("min: %.3f, mean: %.3f, max: %.3f" % (grad.min(), grad.mean(), grad.max()))

                    print("shape: %s" % (str(grad.shape)))

                    
                print("\n")
        self.zero_grad()

    # ------- MISC (MIGHT BE USEFUL)
    def forward_backward(self, XList, y, loss_function=None, verbose=0):
        y_pred = self.forward_pass(XList, numpy_result=False)

        if isinstance(y, (np.ndarray, list)):
            y = utt.numpy2var(y)
    
        if self.my_optimizer is None:
            print "optimizer set..."
            self.set_optimizer()


        # ----- 2.OPTIMIZER
        self.my_optimizer.zero_grad()

        if loss_function is None:
            loss =lf.LOSS_DICT[self.loss_name](y_pred, *y)
        else:
            loss = loss_function(y_pred, *y)

        loss.backward()

        self.my_optimizer.step()

        loss_value = loss.data[0]
        if verbose:
            print "loss: %.3f" %  loss_value

        return loss_value

