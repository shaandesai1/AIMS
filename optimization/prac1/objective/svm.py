import torch

from objective.base import Objective
from utils import accuracy, assert_true


class SVM(Objective):
    def __init__(self, hparams):
        super(SVM, self).__init__(hparams)
        self._range = torch.arange(hparams.n_classes)[None, :]

    def _scores(self,w,x):
        x = x.view(x.size(0),-1)
        return torch.mm(x,w)
 
    def _hinge(self, w, x, y):
        # check input sizes
        self._validate_inputs(w, x, y)

        # compute scores (size (n_samples, n_classes))
        scores = self._scores(w, x)

        # delta: matrix of zero-one misclassification / margin (size (n_samples, n_classes))
        # `detach()`: detach from autograd computational graph (non-differentiable op)
        # `float()`: cast from binary to float
        delta = torch.ne(y[:, None], self._range).detach().float()
        
        # augmented scores: subtract ground truth score and augment with margin
        aug = scores + delta - scores.gather(1, y[:, None])

        # find index of maximal augmented score per sample
        y_star = aug.argmax(1)

        # hinge is obtained by averaging augmented scores selected at y_star
        hinge = aug.gather(1, y_star[:, None]).mean()

        # loss is obtained by averaging delta selected at y_star
        loss = delta.gather(1, y_star[:, None]).mean()
        return hinge, loss

    #def task_error(self, w, x, y):
    #    self._validate_inputs(w, x, y)
        # TODO: Compute mean misclassification
    #    misclass= torch.argmax(torch.mm(x,w),dim=1)
    #    ctr = 0
    #    for i in range(misclass.size(0)):
    #        if misclass[i] != y[i]:
    #            ctr+=1

        #print(misclass)
        #print(y)
     #   error = ctr/x.size(0)
     #   return error

    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2, "Input w should be 2D")
        assert_true(x.dim() == 2, "Input datapoint should be 2D")
        assert_true(y.dim() == 1, "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")


class SVM_SubGradient(SVM):
    def __init__(self, hparams):
        super(SVM_SubGradient, self).__init__(hparams)
        

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        hinge,loss = self._hinge(w, x, y)
        
        return loss
    def oracle(self, w, x, y):
        #self._validate_inputs(w, x, y)
        hinge,loss = self._hinge(w, x, y)
        hinge.backward()        
        dw = w.grad + self.hparams.mu*w
        #print(dw)
        #print(loss)
        #print(hinge)
        primal = hinge + self.hparams.mu/2 *(w**2).sum()
        #print(primal)        
# TODO: Compute objective value
        #obj = 0
        
        #ns_step = torch.zeros(x.size(0))
        #for i in range(x.size(0)):
        #    artmp = torch.ones(w.size(1))
        #    artmp[y[i]] = 0
        #    obj+= torch.max(torch.mv(w.t(),x.t()[:,i]) +artmp - torch.ones(w.size(1))*torch.dot(w.t()[y[i],:],x.t()[:,i]))
        #    ns_step[i] = torch.argmax(torch.mv(w.t(),x.t()[:,i]) +artmp - torch.ones(w.size(1))*torch.dot(w.t()[y[i],:],x.t()[:,i]))
        #print(obj)
        #primal = obj/x.size(0) + self.hparams.mu/2 *(w**2).sum()
        #print(primal)

        # TODO: compute subgradient
        #dw = torch.zeros(w.t().size())
        #for j in range(x.size(0)):
        #    for i in range(w.size(1)):
        #        #dw[i,:] += x.t()[:,j]
        #        if i == ns_step[j]:
        #            dw[i,:] += x.t()[:,j]
        #        if i == y[j]:
        #            dw[i,:] -= x.t()[:,j]
        #dw=dw.t()/x.size(0) + self.hparams.mu*w
        #print(dw)
        return {'obj': primal, 'dw': dw}


class SVM_ConditionalGradient(SVM):
    def __init__(self, hparams):
        super(SVM_ConditionalGradient, self).__init__(hparams)
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        hinge,loss = self._hinge(w, x, y)
        return loss
    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)

        # TODO: Compute primal objective value
        hinge,loss = self._hinge(w, x, y)
        primal = hinge + self.hparams.mu/2 *(w**2).sum()
                
        hinge.backward()        
        
        # TODO: Compute w_s
        w_s = -w.grad*x.size(0)/(self.hparams.n_samples*self.hparams.mu)
        
        # TODO: Compute l_s
        l_s = (x.size(0)/self.hparams.n_samples)*loss

        return {'obj': primal, 'w_s': w_s, 'l_s': l_s}
