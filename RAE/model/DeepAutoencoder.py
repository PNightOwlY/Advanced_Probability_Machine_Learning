import torch.nn.functional as F
from torch import nn, optim
import torch
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DAE(nn.Module):
  def __init__(self):
    super(DAE,self).__init__()

  # encoder layers
    self.enc1 = nn.Linear(784, 625)
    self.enc2 = nn.Linear(625, 400)
    self.enc3 = nn.Linear(400, 225)        
    self.enc4 = nn.Linear(225, 100)
    
    # decoder layers
    self.dec1 = nn.Linear(100, 255)
    self.dec2 = nn.Linear(255, 400)
    self.dec3 = nn.Linear(400, 625)
    self.dec4 = nn.Linear(625, 784)        

    self.criterion = nn.MSELoss()
  
  def encoder(self, x):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.enc3(x))
    x = F.relu(self.enc4(x))

    return x
  
  def decoder(self, x):
    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))

    return x

  def forward(self, x):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.enc3(x))
    x = F.relu(self.enc4(x))
    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    
    return x
  
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
