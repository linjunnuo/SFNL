'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-22 13:56:51
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-22 16:13:11
FilePath: /SFNL/test.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''
import syft as sy
import torch
import numpy as np
import torch
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import syft as sy
import time
hook = sy.TorchHook(torch)
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),           # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 将张量标准化到[-1, 1]的范围
])

# 下载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class SplitNN:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        
    def forward(self, x):
        a = []
        remote_a = []
        
        a.append(models[0](x))
        if a[-1].location == models[1].location:
            remote_a.append(a[-1].detach().requires_grad_())
        else:
            remote_a.append(a[-1].detach().move(models[1].location).requires_grad_())

        i=1    
        while i < (len(models)-1):
            
            a.append(models[i](remote_a[-1]))
            if a[-1].location == models[i+1].location:
                remote_a.append(a[-1].detach().requires_grad_())
            else:
                remote_a.append(a[-1].detach().move(models[i+1].location).requires_grad_())
            
            i+=1
        
        a.append(models[i](remote_a[-1]))
        self.a = a
        self.remote_a = remote_a
        
        return a[-1]
    
    def backward(self):
        a=self.a
        remote_a=self.remote_a
        optimizers = self.optimizers
        
        i= len(models)-2   
        while i > -1:
            if remote_a[i].location == a[i].location:
                grad_a = remote_a[i].grad.copy()
            else:
                grad_a = remote_a[i].grad.copy().move(a[i].location)
            a[i].backward(grad_a)
            i-=1

    
    def zero_grads(self):
        for opt in optimizers:
            opt.zero_grad()
        
    def step(self):
        for opt in optimizers:
            opt.step()

torch.manual_seed(0)  # Define our model segments
input_size = 784
hidden_sizes = [128, 640]
output_size = 10
models = [
    nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
    ),
    nn.Sequential(
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
    ),
    nn.Sequential(
                nn.Linear(hidden_sizes[1], output_size),
                nn.LogSoftmax(dim=1)
    )
]
# Create optimisers for each segment and link to them
optimizers = [
    optim.SGD(model.parameters(), lr=0.03,)
    for model in models
]

# create some workers
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
claire = sy.VirtualWorker(hook, id="claire")

# Send Model Segments to model locations
model_locations = [alice, bob, claire]
for model, location in zip(models, model_locations):
    model.send(location)
splitNN =  SplitNN(models, optimizers)

def train(x, target, splitNN):
    
    #1) Zero our grads
    splitNN.zero_grads()
    
    #2) Make a prediction
    pred = splitNN.forward(x)
    
    #3) Figure out how much we missed by
    criterion = nn.NLLLoss()
    loss = criterion(pred, target)
    
    #4) Backprop the loss on the end layer
    loss.backward()
    
    #5) Feed Gradients backward through the network
    splitNN.backward()
    
    #6) Change the weights
    splitNN.step()
    
    return loss

# ---------------------------------------------------------------------------- #


for i in range(10):
    running_loss = 0
    for images, labels in trainloader:
        images = images.send(models[0].location)
        images = images.view(images.shape[0], -1)
        labels = labels.send(models[-1].location)
        loss = train(images, labels, splitNN)
        running_loss += loss.get()

    else:
        print("Epoch {} - Training loss: {}".format(i, running_loss/len(trainloader)))