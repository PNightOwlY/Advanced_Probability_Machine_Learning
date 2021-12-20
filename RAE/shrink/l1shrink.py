import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def shrink(epsilon, x):
  y = torch.zeros((x.shape))
  y = y.to(device)

  y[x > epsilon] = -1
  y[x < -epsilon] = +1

  x = x + y * epsilon
  y[y != 0] = 1

  return x.mul(y)