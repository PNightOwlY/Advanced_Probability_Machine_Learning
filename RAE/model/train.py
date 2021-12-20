
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch import nn, optim
import torch
from torch import optim
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_data, model, optimizer, EPOCH=10):
  model = model.to(device)

  train_loss = []
  test_loss = []

  for k in range(EPOCH):
    for data in train_data:
      data = data.to(device)
      # compute loss
      data = data.view(-1,28*28).float()
      
      loss = model.fit(data, optimizer)
      
      train_loss.append(loss)

  plt.plot(train_loss)