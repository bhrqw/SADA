# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from torch.distributions import laplace
from torch.distributions import uniform

from attacks.tools import clamp
from attacks.tools import clamp_by_pnorm
from attacks.tools import batch_multiply
from attacks.tools import normalize_by_pnorm
from attacks.tools import predict_from_logits


def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    # TODO: Currently only considered one way of "uniform" sampling
    # for Linf, there are 3 ways:
    #   1) true uniform sampling by first calculate the rectangle then sample
    #   2) uniform in eps box then truncate using data domain (implemented)
    #   3) uniform sample in data domain then truncate with eps box
    # for L2, true uniform sampling is hard, since it requires uniform sampling
    #   inside a intersection of cube and ball, so there are 2 ways:
    #   1) uniform sample in the data domain, then truncate using the L2 ball
    #       (implemented)
    #   2) uniform sample in the L2 ball, then truncate using the data domain
    # for L1: uniform l1 ball init, then truncate using the data domain

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(0, 1)
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray
        delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data


def is_successful(y1, y2, targeted):
    if targeted is True:
        return y1 == y2
    else:
        return y1 != y2


class AttackConfig(object):
    # a convenient class for generate an attack/adversary instance

    def __init__(self):
        self.kwargs = {}

        for mro in reversed(self.__class__.__mro__):
            if mro in (AttackConfig, object):
                continue
            for kwarg in mro.__dict__:
                if kwarg in self.AttackClass.__init__.__code__.co_varnames:
                    self.kwargs[kwarg] = mro.__dict__[kwarg]
                else:
                    # make sure we don't specify wrong kwargs
                    assert kwarg in ["__module__", "AttackClass", "__doc__"]

    def __call__(self, *args):
        adversary = self.AttackClass(*args, **self.kwargs)
        print(self.AttackClass, args, self.kwargs)
        return adversary


def multiple_mini_batch_attack(adversary, loader, device="cuda"):
    lst_label = []
    lst_pred = []
    lst_advpred = []

    for data, label in loader:
        data, label = data.to(device), label.to(device)
        adv = adversary.perturb(data, label)
        advpred = predict_from_logits(adversary.predict(adv))
        pred = predict_from_logits(adversary.predict(data))
        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)

    return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred)


def max_index (data,back_rate):
    # import pdb;pdb.set_trace()
    dim_1 = data.view(data.size(0),-1)
    # k=2表示选择两个最大值
    max_n,max_l = dim_1.topk(int(dim_1.size(1)*back_rate),dim=1)
    # 要加上values，否则会得到一个包含values和indexs的对象
    a_min,_=torch.min(max_n,dim=-1)
    # repeat里的4和data的最后一维相同
    a_min=a_min.unsqueeze(-1).repeat(1,dim_1.size(-1))
    ge=torch.ge(dim_1,a_min).float()
    result = ge.view(data.size())
    return result
