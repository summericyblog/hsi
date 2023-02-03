import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


def read_data():
    fp = 'data/hsi_ma.csv'
    X_columns = ['vol', 'open_ret', 'day_ret', 'ret', 'ma_close_z_5', 'ma_open_z_5', 'strong_5', 'ma_close_z_10', 'ma_open_z_10', 'strong_10', 'ma_close_z_20', 'ma_open_z_20', 'strong_20', 'ma_close_z_60', 'ma_open_z_60', 'strong_60', 'ma_close_z_180', 'ma_open_z_180', 'strong_180']
    Y_columns = ['open_ret', 'day_ret', 'ret']
    usecols = ['Date'] + X_columns + Y_columns
    df = pd.read_csv(fp, encoding='utf8', usecols=usecols)
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)
    X = df[X_columns]
    Y = df[Y_columns].shift(-1)
    X.drop(X.tail(1).index, inplace=True) 
    Y.drop(Y.tail(1).index, inplace=True) 
    # Y = df[Y_columns].shift(0)
    return (X, Y)


def sign_ret(y, d):
    new_y = y.copy()
    new_y[new_y > 0] = 1
    new_y[new_y < 0] = -1
    new_y[(new_y >= -d) & (new_y <= d)] = 0
    new_y = new_y.astype(np.int32)
    return new_y



class Net(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        k1 = 30
        k2 = 50
        k3 = 18
        self.l1 = nn.Linear(in_dim, k1)
        self.l2 = nn.Linear(k1, k2)
        self.l3 = nn.Linear(k2, k3)
        self.l4 = nn.Linear(k3, out_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        ret = self.l1(x)
        ret = F.relu(ret)
        ret = self.l2(ret)
        ret = F.relu(ret)
        ret = self.l3(ret)
        ret = F.relu(ret)
        ret = self.l4(ret)
        return ret


def train():
    X, Y_origin = read_data()
    Y_origin = Y_origin['open_ret'] * 100
    Y = Y_origin
    # Y = sign_ret(Y_origin, 0.004)
    Y_origin = torch.tensor(Y_origin.values).view(-1, 1).to(torch.float32)
    X = torch.tensor(X.values)
    Y = torch.tensor(Y.values).view(-1, 1).to(torch.float32)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    in_dim = X.shape[1]
    out_dim = 1
    net = Net(in_dim, out_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 1000000
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = net(X_train)
        loss = criterion(pred, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            pred[pred > 0] = 1.0
            pred[pred < 0] = -1.0
            ret = torch.mul(pred, Y_train)
            pos = (ret > 0).sum() / ret.shape[0]
            ret = ret.mean()
            test_pred = net(X_test)
            test_pred[test_pred > 0] = 1.0
            test_pred[test_pred < 0] = -1.0
            ret_test = torch.mul(test_pred, Y_test)
            pos_test = (ret_test > 0).sum() / ret_test.shape[0]
            ret_test = ret_test.mean()
            print(loss.item(), ret.item(), ret_test.item(), pos, pos_test)



if __name__ == '__main__':
    train()