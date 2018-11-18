import torch
import random
import math

from optim.base import Optimizer, Variable, HParams


class HParamsBCFW(HParams):
    required = ('n_samples', 'batch_size', 'n_features', 'n_classes', 'mu')
    defaults = {'verbose': 1, 'eps': 1e-5}

    def __init__(self, **kwargs):
        super(HParamsBCFW, self).__init__(kwargs)
        self.n_blocks = int(math.ceil(self.n_samples / float(self.batch_size)))


class VariablesBCFW(Variable):
    def __init__(self, hparams):
        """
        Shared Variable.
        """
        super(VariablesBCFW, self).__init__(hparams)

    def init(self):
        self.w = torch.zeros((self.hparams.n_features, self.hparams.n_classes),
                             requires_grad=True)
        self.w_i = torch.zeros((self.hparams.n_blocks, self.hparams.n_features,
                                self.hparams.n_classes))
        self.ll = torch.tensor(0.)
        self.l_i = torch.zeros((self.hparams.n_blocks,))


class BCFW(Optimizer):
    def __init__(self, hparams):
        """
        Shared Variable.
        """
        super(BCFW, self).__init__(hparams)

    def create_vars(self):
        return VariablesBCFW(self.hparams)

    def _step(self, oracle_info):

        # TODO: compute optimal step size
        gamma = self._step_size(oracle_info['w_s'],oracle_info['l_s'],oracle_info['i'])

        # TODO: perform update

    def _step_size(self, w_s,l_s,i):
        # TODO: compute optimal step size
        #print(self.variables.l_i.size())
        #print(self.variables.w_s.size())
        gamma = (-self.hparams.mu*(w_s - self.variables.w_i[i,:,:]).t().mm(self.variables.w) + l_s - self.variables.l_i[i])/(self.hparams.mu*torch.norm(w_s-self.variables.w_i[i,:,:],p=2))
        #gamma = None
        return gamma

    def get_sampler(self, dataset):
        # this sampler shuffles the order of the mini-batches but not
        # which indices each mini-batch contains
        dataset_size = len(dataset)
        batch_size = self.hparams.batch_size
        all_indices = list(range(dataset_size))
        batch_indices = []
        for i in range(int(math.ceil(dataset_size / float(batch_size)))):
            batch_indices.append(
                (i, all_indices[i * batch_size: i * batch_size + batch_size]))
        random.shuffle(batch_indices)
        for i, batch_index in batch_indices:
            x, y = dataset[torch.LongTensor(batch_index)]
            yield (i, x, y.type_as(dataset.target_type))

    def get_sampler_len(self, dataset):
        batch_size = self.hparams.batch_size
        return int(math.ceil(len(dataset) / float(batch_size)))
