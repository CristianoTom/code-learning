# 调包
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

def recognize(ls):
    """识别网络结构,将网络结构转换为线性层的输入输出"""
    l = []
    for i in range(len(ls)-1):
        l.append([ls[i], ls[i+1]])
    return l

class LinearNet(nn.Module):
    """定义线性网络块"""
    def __init__(self, n_input, in_output):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_input, in_output)

    def forward(self, x):
        y = self.linear(x)
        return y

def get_net(net_structure):
    net_structure = recognize(net_structure)
    net = nn.Sequential()
    for i in range(len(net_structure)):
        net.add_module('linear%d' % i, LinearNet(net_structure[i][0], 
                                                 net_structure[i][1]))
    return net

def load_array(data_arrays, batch_size, is_train=True): 
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)