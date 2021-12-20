import numpy as np
import matplotlib.pyplot as plt


import torch.nn.functional as F
from torch import nn, optim
import torch
from torch import optim
from torchvision import datasets, transforms

# use GPU for computation if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim) # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

        self.criterion = nn.MSELoss()
        
    # 编码过程
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    # 整个前向传播过程：编码-》解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst

    def cost_function(self,X):
      X = X.view(-1, 28*28)
      X = X.to(device)
      X_recon = self.forward(X)
      loss = self.criterion(X, X_recon)
      return loss

    def fit(self, X, optimizer):
      # move to the device
      X = X.to(device)

      # reset the gradient information
      optimizer.zero_grad()

      # compute loss
      loss = self.cost_function(X)

      # perform backpropagation
      loss.backward()

      # optimize the weight
      optimizer.step()

      return loss