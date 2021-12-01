import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn


import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils.black_scholes import black_scholes_put
import torch.optim as optim


class PutNet(nn.Module):
    """

    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!

    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()
        self.l1 = nn.Linear(5,20)
        self.l2 = nn.Linear(20,20)
        self.l3 = nn.Linear(20,20)
        self.out = nn.Linear(20,1)

        self.x_scaler = { 'mean': torch.zeros(5), 'std': torch.ones(5) }
        self.y_scaler = { 'mean': torch.zeros(1), 'std': torch.ones(1) }

    def transform(self, x, scaler):
        mu = scaler['mean']
        std = scaler['std']
        return ((x - mu)/ std)

    def inverse_transform(self, x, scaler):
        mu = scaler['mean']
        std = scaler['std']
        return( x * std + mu)

    def fit_scalers(self, X , y):
        mu_x = X.mean(0)
        std_x = X.std(0)

        mu_y = y.mean()
        std_y = y.std()

        self.x_scaler['mean'] = mu_x
        self.x_scaler['std'] = std_x
        self.y_scaler['mean'] = mu_y
        self.y_scaler['std'] = std_y


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.out(x)
        return (x)



if __name__ == "__main__":



    # create model
    mm = PutNet()

    # load dataset
    dd = pd.read_csv("bs-put-1k.csv")


    # set up training
    X = torch.Tensor(dd[['S','K','T','r','sigma']].to_numpy())
    y = torch.Tensor(dd[['value']].to_numpy())


    mm.fit_scalers(X,y)

    with torch.no_grad():
        X = mm.transform(X, mm.x_scaler)
        y = mm.transform(y, mm.y_scaler)


    X = Variable(X)
    y = Variable(y)
    loss = torch.nn.MSELoss()
    optimizer = optim.SGD(mm.parameters(), lr = 1e-3, momentum=0.9)


    # train
    for i in range(500):

        # TODO: modify to account for dataset size
        y_hat_i = mm(X)
        y_i = y

        # calculate training loss
        training_loss = loss(y_hat_i, y_i)

        # take a step.
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        # check validation loss
        with torch.no_grad():
            # TODO: use a proper validation set
            validation_loss = loss(mm(X), y)

        print(f"Iteration: {i} | Training Loss: {training_loss} | Validation Loss {validation_loss} ")


    torch.save(mm.state_dict(), "simple-model.pt")

    y_unscaled = mm.inverse_transform(y, scaler= mm.y_scaler)
    y_hat_unscaled = mm.inverse_transform(y_hat_i, scaler=mm.y_scaler)
