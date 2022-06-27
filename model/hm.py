import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_norm, indexes, features, features_norm, momentum):
        ctx.features = features
        ctx.features_norm = features_norm
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        # outputs = inputs.mm(ctx.features.t())
        
        outputs = inputs_norm.mm(ctx.features_norm.t())
        # inputs.mm(F.normalize(ctx.features, dim=-1).t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[1]:
            grad_inputs = grad_outputs.mm(ctx.features_norm)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            # ctx.features[y] /= ctx.features[y].norm()

        return None, grad_inputs, None, None, None, None


def hm(inputs, inputs_norm, indexes, features, features_norm, momentum=0.5):
    return HM.apply(inputs, inputs_norm, indexes, features, features_norm, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        # change the data type to float32 to float16
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # self.register_buffer('features', torch.zeros((num_samples, num_features), dtype=torch.float16))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes, self_momentum=None):
        # inputs: B*2048, features: L*2048
        inputs_norm = F.normalize(inputs, dim=1)
        feature_norm = F.normalize(self.features, dim=1)

        if self_momentum is None:
            inputs = hm(inputs, inputs_norm, indexes, self.features, feature_norm, self.momentum)
        else:
            inputs = hm(inputs, inputs_norm, indexes, self.features, feature_norm, self_momentum)
    
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        # change the data type to float32 to float16
        sim = torch.zeros(labels.max()+1, B).float().cuda()
        # sim = torch.zeros((labels.max()+1, B), dtype=torch.float16).cuda()

        # import pdb
        # pdb.set_trace()

        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets) 