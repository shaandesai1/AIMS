import torch

from objective.base import Objective
from utils import assert_true


class Ridge(Objective):
    def _validate_inputs(self, w, x, y):
        assert_true(w.dim() == 2,
                    "Input w should be 2D")
        assert_true(w.size(1) == 1,
                    "Ridge regression can only perform regression (size 1 output)")
        assert_true(x.dim() == 2,
                    "Input datapoint should be 2D")
        assert_true(y.dim() == 1,
                    "Input label should be 1D")
        assert_true(x.size(0) == y.size(0),
                    "Input datapoint and label should contain the same number of samples")


class Ridge_ClosedForm(Ridge):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        error = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0]
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)

        # TODO: Compute objective value
        a = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0]
        b = torch.mm(w.t(),w).squeeze(-1)*(self.hparams.mu)/2
        obj = a + b.squeeze()

        # TODO: compute close form solution
        atmp = torch.inverse(torch.mm(x.t(),x)*2/x.size()[0] + self.hparams.mu*torch.eye(w.size()[0],w.size()[0]))
        #print(torch.mm(x.t(),x)/x.size()[0] + self.hparams.mu*torch.eye(w.size()[0],w.size()[0]))
        
        #print(torch.mm(x.t(),y.unsqueeze(-1)))       
        sol = torch.mm(atmp,torch.mm(x.t(),y.unsqueeze(-1))*2/x.size()[0])
        print(sol)
        return {'obj': obj, 'sol': sol}


class Ridge_Gradient(Ridge):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        #print(torch.mm(x,w).squeeze(-1))
	# TODO: Compute mean squared error
        error = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0]
        #print(error)
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute objective value
        #print(torch.mm(w.t(),w))
        a = ((torch.mm(x,w).squeeze(-1) - y)**2).sum()/x.size()[0]
        b = torch.mm(w.t(),w).squeeze(-1)*(self.hparams.mu)/2
        #print(b)
        obj =  a +b.squeeze()
        # TODO: compute close form solution
        an = torch.mm((2/x.size()[0])*x.t(),(torch.mm(x,w)-y.unsqueeze(-1)))
        a2 = self.hparams.mu*w
        #print(an)
        #print(a2) 
        dw = (an + a2)
        

        

        return {'obj': obj, 'dw': dw}
