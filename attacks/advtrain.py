import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


meanstd_dataset = {
    'cifar10': [[0.49139968, 0.48215827, 0.44653124],
                [0.24703233, 0.24348505, 0.26158768]],
    'mnist': [[0.13066051707548254],
                [0.30810780244715075]],
    'fashionmnist': [[0.28604063146254594],
                [0.35302426207299326]],
    'imagenet': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
    'tinyimagenet': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
    'tinyimagenet_64': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
}


class Normalize(nn.Module):
    def __init__(self, dataset, input_channels, input_size):
        super(Normalize, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels

        self.mean, self.std = meanstd_dataset[dataset.lower()]
        self.mean = torch.Tensor(np.array(self.mean)[:, np.newaxis, np.newaxis]).cuda()
        self.mean = self.mean.expand(self.input_channels, self.input_size, self.input_size).cuda()
        self.std = torch.Tensor(np.array(self.std)[:, np.newaxis, np.newaxis]).cuda()
        self.std = self.std.expand(self.input_channels, self.input_size, self.input_size).cuda()
        # self.mean, self.std = self.get_meanstd(dataset)

    # def get_meanstd(self, dataset):
    #     mean, std = meanstd_dataset[dataset.lower()]
    #     mean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis]).cuda()
    #     mean = mean.expand(self.input_channels, self.input_size, self.input_size).cuda()
    #     std = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis]).cuda()
    #     std = std.expand(self.input_channels, self.input_size, self.input_size).cuda()
    #     return mean, std
    
    def forward(self, input):
        device = input.device
        output = input.sub(self.mean.to(device)).div(self.std.to(device))
        return output
    
    # def __call__(self, input):
    #     return self.forward(input)


class freetrain:
    def __init__(self, batch_size, input_channels, input_size, eps=0.31, n_repeats=8, eps_iter=0.01):
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_repeats = n_repeats
        self.input_size = input_size
        self.input_channels = input_channels
        self.global_noise_data = torch.zeros([batch_size, input_channels, input_size, input_size]).cuda()
        # self.normalize = Normalize(dataset, input_channels, input_channels)
        # self.mean, self.std = self.get_meanstd(dataset)
        # mean, std = get_meanstd(dataset, input_channels, input_size)

    
    def get_advinput(self, input):
        # Ascend on the global noise
        self.noise_batch = Variable(self.global_noise_data[0: input.size(0)], requires_grad=True).cuda()
        in1 = input + self.noise_batch
        in1.clamp_(0, 1.0)
        # in1 = self.normalize(in1)
        # in1.sub_(self.mean).div_(self.std)
        return in1 # self.normalize(in1)

    def attack(self, gradz):
        return self.eps_iter * torch.sign(gradz)

    def update_noise(self, input):
        # Update the noise for the next iteration
        pert = self.attack(self.noise_batch.grad)
        self.global_noise_data[0: input.size(0)] += pert.data
        self.global_noise_data.clamp_(-self.eps, self.eps)

