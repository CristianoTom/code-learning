# 调包
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import pandas as pd
from pathlib import Path


def read_csv(file_name):
    """读取csv文件,返回特征和标签"""
    PATH = Path(__file__).parent.resolve()
    data = pd.read_csv(PATH / 'data' / file_name, header=None)
    data = data.values
    features = torch.tensor(data[:, :], dtype=torch.float32)
    return features

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


def train(net, features, labels, batch_size, num_epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    net.to(device)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    data_iter = load_array((features, labels), batch_size)
    for epoch in range(num_epochs):
        net.train()
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        with torch.no_grad():
            l = loss(net(features.to(device)), labels.to(device))
        print(f'epoch {epoch + 1}, loss {l:f}')

def use_net(net, test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device).eval()
    test = test.to(device)
    with torch.no_grad():
        return net(test)
