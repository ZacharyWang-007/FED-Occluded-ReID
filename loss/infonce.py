import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class InfoNCE(nn.Module):
    def __init__(self, num_samples, temp=0.05, momentum=0.2):
        super(InfoNCE, self).__init__()
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('labels', torch.arange(num_samples).long().cuda())
        # torch.zeros(num_samples).long())

    def forward(self, inputs, features, indexes):
        # inputs: B*2048, features: L*2048
  
        inputs = inputs.mm(features.t())

        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)