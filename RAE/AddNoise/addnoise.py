import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_noise_uniform_normal(X, corNum=10):
  X = X.view(-1,28*28).float()
  index = torch.randint(0,X.shape[1],(len(X), corNum))
  index = index.to(device)

  for i in range(len(X)):
    X[i, index[i]] = torch.randn(len(index[i])).to(device)
  return X
 


def add_noise_fixed(X):
    X = X.float()
    index = torch.tensor([100,200,300,400,500,600,700])
    X[:, index] = torch.randn(len(index)).to(device)
    return X