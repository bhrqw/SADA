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
from attacks.tools import clamp
from attacks.tools import normalize_by_pnorm

from .base import Attack
from .base import LabelMixin
from model import resnet


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

weight_path = 'model/resnet18.pt'

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class GradientSignAttack_d(Attack, LabelMixin):
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
        super(GradientSignAttack_d, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None, sa_b=None):
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
        # import pdb;pdb.set_trace()
        epsilon = (self.eps / 255.) / std
        delta = torch.zeros_like(x).cuda()
        delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())
        delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(), epsilon[1][0][0].item())
        delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(), epsilon[2][0][0].item())
        delta.requires_grad = True
        predict = resnet.resnet18(num_classes=self.num_classes)
        pretrained_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        predict_dict= predict.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in predict_dict and 'fc' not in k)}
        predict_dict.update(pretrained_dict)
        predict.load_state_dict(predict_dict,strict=False)

        xadv =x.requires_grad_()
        predict= predict.cuda()
        outputs = predict(xadv+delta)


        if isinstance(outputs,tuple):
            logits= outputs[-2]
        else:
            logits=outputs
        loss = self.loss_fn(logits,y.cuda())
        loss=-loss
        loss.backward()
        grad =x.grad.detach()
        delta.data =delta + epsilon * torch.sign(grad)
        delta = delta.detach()
        if type(sa_b) != type(None):
            a=sa_b
            delta.data = torch.mul(delta.data, a.cuda())
        delta = clamp(delta, -epsilon.cuda(),epsilon.cuda())
        xadv = clamp(xadv + delta, lower_limit.cuda(),upper_limit.cuda())
        return xadv


FGSM_delta = GradientSignAttack_d
