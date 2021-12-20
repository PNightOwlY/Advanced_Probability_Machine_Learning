import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def shrink(epsilon, x):
  y = torch.zeros((x.shape))
  y = y.to(device)

  norm = torch.norm(x, 2, 0)
  for i in range(x.shape[1]):
    if norm[i] > epsilon:
      y[:, i] = x[:,i] - epsilon * x[:,i] / norm[i]
    else:
      y[:, i] = 0.
  
  return y