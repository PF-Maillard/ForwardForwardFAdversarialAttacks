import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import Utils

#Layer object for FF training

class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)

            loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold,g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

   
##MNIST models

class Net_MNIST(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.Layer1 = Layer(784, 500).cuda()
        self.Layer2 = Layer(500, 500).cuda()

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            z = Utils.overlay_y_on_x(x, label)
            goodness = []
            z = self.Layer1(z)
            goodness += [z.pow(2).mean(1)]
            z = self.Layer2(z)
            goodness += [z.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = self.Layer1.train(x_pos, x_neg)
        h_pos, h_neg = self.Layer2.train(h_pos, h_neg)

    
class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MyCNN_MNIST(nn.Module):
    def __init__(self):
        super(MyCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # Fewer filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # Fewer filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)  # Fewer neurons
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Reshape inputs to [batch_size, 1, 28, 28]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
##CIFAR10 models
   
class Net_CIFAR10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer1 = Layer(3072, 1024).cuda()
        self.Layer2 = Layer(1024, 1024).cuda()
        self.Layer3 = Layer(1024, 512).cuda()
        self.Layer4 = Layer(512, 256).cuda()
        
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            z = Utils.overlay_y_on_x(x, label)
            goodness = []
            z = self.Layer1(z)
            goodness += [z.pow(2).mean(1)]
            z = self.Layer2(z)
            goodness += [z.pow(2).mean(1)]
            z = self.Layer3(z)
            goodness += [z.pow(2).mean(1)]
            z = self.Layer4(z)
            goodness += [z.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = self.Layer1.train(x_pos, x_neg)
        h_pos, h_neg = self.Layer2.train(h_pos, h_neg)
        h_pos, h_neg = self.Layer3.train(h_pos, h_neg)
        h_pos, h_neg = self.Layer4.train(h_pos, h_neg)   
    
class MLP_CIFAR10(nn.Module):
    def __init__(self):
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class MyCNN_CIFAR10(nn.Module):
    def __init__(self):
        super(MyCNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.Flatten = nn.Flatten()
        
    def forward(self, x):
        x = x.view(-1, 3, 32, 32) 
        
        x = self.relu(self.conv1(x)) 
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.pool(self.relu(self.conv6(x)))
        x = self.Flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x