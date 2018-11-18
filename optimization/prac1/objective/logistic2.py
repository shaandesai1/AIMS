import torch

from objective.base import Objective
from utils import assert_true


class Logistic_Gradient(Objective):
    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2,
                    "Input w should be 2D")
        assert_true(x.dim() == 2,
                    "Input datapoint should be 2D")
        assert_true(y.dim() == 1,
                    "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute cross entropy prediction error
        im1 = torch.exp(torch.mm(w.t(),x.t())).sum(dim=0)
        im2 = torch.mm(w.t(),x.t())
        yy = torch.zeros(y.size())
        #print(im2.size())
        for i in range(y.size(0)):
            yy[i] = im2[y[i],i] 
        #print(yy)
        #print(torch.log(im1)-y)
        err = (torch.log(im1).sum() - yy.sum())/x.size(0)
        #print(err)
        #print(w.size())
        #print(x.size())
        error = err
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        im1 = torch.exp(torch.mm(w.t(),x.t())).sum(dim=0)
        im2 = torch.mm(w.t(),x.t())
        yy = torch.zeros(y.size())
        #print(im2.size())
        for i in range(y.size(0)):
            yy[i] = im2[y[i],i] 
        #print(yy)
        #print(torch.log(im1)-y)
        err = (torch.log(im1).sum() - yy.sum())/x.size(0)
        b = (w**2).sum()*(self.hparams.mu)/2
        obj = err + b.squeeze()
        obj.backward()
        dw = obj.grad
        #obj = None
        # TODO: compute gradient
        #denom=torch.exp(torch.mm(w.t(),x.t())).sum(dim=0)
        #print(denom)
        #dw = torch.zeros(w.t().size())
        #print(dw[0,:])
        #print(torch.exp(torch.dot(w.t()[0,:],x.t()[:,0]))*x.t()[:,0].squeeze()/denom[0])
        #for i in range(w.t().size(0)):
         #   for j in range(x.size(0)):
        #        dw[i,:] += torch.exp(torch.dot(w.t()[i,:],x.t()[:,j]))*x.t()[:,j].squeeze()/denom[j]
        #        if y[j] == i:
        #            dw[i,:] -=  x.t()[:,j]
        #dw = dw.t()/x.size(0) + self.hparams.mu*w
        return {'obj': obj, 'dw': dw}
