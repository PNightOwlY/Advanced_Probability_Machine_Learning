import torch.nn.functional as F
from torch import nn, optim
from torch import optim
from torchvision import datasets, transforms
from shrink import l1shrink as SHR
from shrink import l21shrink as SHR2
import DAE
import train


class RDAE(object):
  def __init__(self, lambda_=1.0, error=1.0e-7):
    super(RDAE,self).__init__()
    self.lambda_ = lambda_
    self.error = error
    self.AE = DAE()

  def fit(self, X):
    self.AE.to(device)
    optimizer = optim.Adam(self.AE.parameters(), lr=LEARNING_RATE)
    error = self.error
    lambda_ = self.lambda_

    L = torch.zeros(X.shape)
    S = torch.zeros(X.shape)

    LS0 = L + S

    XFnorm = torch.norm(X, 'fro')
    mu = (len(X))/(4.0 * torch.norm(X, 1))

    # move to the device
    X = X.to(device)
    L = L.to(device)
    S = S.to(device)
    LS0 = LS0.to(device)

    # add noise to X
    # X = addnoise.add_noise_uniform_normal(X, 100)

    for i in range(5):
      # updata L
      L = X - S

      # generate train data
      train_datas = torch.utils.data.DataLoader(L, batch_size = BATCH_SIZE, shuffle=False)
      
      train(train_datas, self.AE, optimizer)
      
      with torch.no_grad():
        # get optimized L
        L = self.AE(L.float())
        
        # S = shrink_anomly(lambda_/mu, (X-L))
        S = SHR.shrink(lambda_/mu, (X-L))

      # break criterion 1: the L and S are close enough to X
        c1 = torch.norm(X-L-S, 'fro') / XFnorm

      # break criterion 2: there is no changes for L and S
        c2 = torch.min(mu, torch.sqrt(mu)) * torch.norm(LS0 - L - S) / XFnorm

        if c1 < error and c2 < error:
          print("early break")
          break
      LS0 = L + S
    return L, S