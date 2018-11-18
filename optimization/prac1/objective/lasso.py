import torch

from objective.base import Objective
from utils import assert_true


class Lasso(Objective):
    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2,
                    "Input w should be 2D")
        assert_true(w.size(1) == 1,
                    "Lasso regression can only perform regression (size 1 output)")
        assert_true(x.dim() == 2,
                    "Input datapoint should be 2D")
        assert_true(y.dim() == 1,
                    "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")


class Lasso_subGradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        error = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0]
        
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        a = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0]
        b = torch.norm(w,p=1).squeeze(-1)*(self.hparams.mu)/2
        obj = a + b.squeeze()

        #print(obj)
        # TODO: compute subgradient

        dummy = torch.zeros(w.size())
        dummy[w>0] = 1
        dummy[w<0] = -1
        
        err = torch.mm(x,w).squeeze()-y
        print(err.size())
        dw = (2./x.size(0))*torch.mv(x.t(),err) + (self.hparams.mu/2.)*dummy.squeeze()
        
        dw = dw.unsqueeze(1)
        return {'obj': obj, 'dw': dw}


class SmoothedLasso_Gradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        error = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0]
        
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        obj = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0] + (self.hparams.mu/2)*self.hparams.temp*torch.log(torch.exp(w/self.hparams.temp)+torch.exp(-w/self.hparams.temp)).sum()
        #print(obj)

        #TODO: compute gradient
        #print(torch.exp(w/self.hparams.temp)/torch.exp(-w/self.hparams.temp))
        err = torch.mm(x,w)-y.unsqueeze(1)
        first = (2./x.size(0))*torch.mm(x.t(),err)
        print(first)
        second = (self.hparams.mu/2)*(torch.exp(w/self.hparams.temp)-torch.exp(-w/self.hparams.temp))/(torch.exp(w/self.hparams.temp)+torch.exp(-w/self.hparams.temp))
        print(second)
        dw =   second+first
        print(dw)
        return {'obj': obj, 'dw': dw}
