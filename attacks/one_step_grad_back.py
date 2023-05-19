# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import random
import numpy as np
from attacks.tools import clamp
from attacks.tools import normalize_by_pnorm
from .utils import max_index

from .base import Attack
from .base import LabelMixin


# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
# mu = torch.tensor(cifar10_mean).view(3,1,1)
# std = torch.tensor(cifar10_std).view(3,1,1)
image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]
mu = torch.tensor(image_mean).view(3,1,1).cuda()
std = torch.tensor(image_std).view(3,1,1).cuda()
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class GradientSignAttack_b(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=10, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(GradientSignAttack_b, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
    
    def max_index(data):
        index = []  # 创建列表,存放最大值的索引
        # data = data.A  # 若data是矩阵，需先转为array,因为矩阵ravel后仍是二维的形式，影响if环节
        dim_1 = data.ravel()  # 转为一维数组
        import pdb;pdb.set_trace()
        while len(index) < (data.size//4):
            max_n = np.argmax(dim_1)  # 最大值max_n
            pos = np.unravel_index(max_n, data.shape, order='C')# 返回一维索引对应的多维索引，譬如4把转为(1,1),即一维中的第5个元素对应在二维里的位置是2行2列
            index.append(pos)  # 存入index
            dim_1[max_n] = 1
        # import pdb;pdb.set_trace()
        pdb.set_trace()
        dim_1=np.reshape(dim_1,data.shape)
        dim_1 = np.isin(dim_1, 1).astype(np.uint8)
        return dim_1


    def perturb(self,back_rate, x, y=None, sa_b=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, y)
        # print(y)
        # import pdb; pdb.set_trace()
        epsilon = (self.eps / 255.) / std
        delta = torch.zeros_like(x).cuda()
        # import pdb;pdb.set_trace()
        delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())
        delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(), epsilon[1][0][0].item())
        delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(), epsilon[2][0][0].item())
        delta.requires_grad = True

        max_out, _ = torch.max(delta, dim=1, keepdim=True)
        # import pdb;pdb.set_trace()

        #########xu yao xiu gai#######
        max_ten = max_index(max_out,back_rate)
        ##############################

        xadv = x.requires_grad_()
        # xadv = Variable(x.detach(), requires_grad=True)
        # print(xadv)
        outputs = self.predict(xadv+delta)
        


        if isinstance(outputs, tuple):
            logits = outputs[-2]
        else:
            logits = outputs

        loss = self.loss_fn(logits, y)
        if self.targeted:
            loss = -loss

        loss.backward()
        # (loss / 2).backward()
        # grad_sign = xadv.grad.detach().sign()

        grad = delta.grad.detach()
        delta.data = delta + epsilon * torch.sign(grad)
        delta = delta.detach()
        # xadv = xadv + self.eps * grad_sign

        #######################
        # delta = delta * grad_sign
        # delta = self.eps * grad
        # if simple:
            # alpha =4
            # a = torch.ones(delta.data.shape)
            # a[:,:,alpha:-alpha,alpha:-alpha] = a[:,:,alpha:-alpha,alpha:-alpha]*0
        #     # import pdb; pdb.set_trace()
        # pdb.set_trace()
        if type(sa_b) != type(None):
            a = sa_b
            delta.data = torch.mul(delta.data, a.cuda())
        # delta = clamp(delta, -torch.tensor(epsilon).cuda(), torch.tensor(epsilon).cuda())
        delta = clamp(delta, -epsilon.cuda(), epsilon.cuda())
        xadv = clamp(xadv + delta, lower_limit.cuda(), upper_limit.cuda())
        # import pdb; pdb.set_trace()
        # xadv = xadv + delta
        #######################
        # xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv,max_ten


FGSM_back = GradientSignAttack_b
