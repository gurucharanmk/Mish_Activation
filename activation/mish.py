import torch
import torch.nn as nn
import torch.nn.functional as F



class Mish(nn.Module):
    '''https://arxiv.org/abs/1908.08681v1
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input *( torch.tanh(F.softplus(input)))