import torch
import torch.nn as nn
import torch.nn.functional as F


num_points = 50

mean = torch.zeros((1), device='cuda')
epsilon = torch.eye(num_points, device='cuda') * 1e-6

def make_points(batch_size):
    x = torch.rand((batch_size, num_points, 1), device='cuda')
    K = torch.exp(-1 / 2 * ((x.view(batch_size, 1, -1) - x) / 0.1) ** 2) + epsilon
    y = torch.distributions.multivariate_normal.MultivariateNormal(mean, K).sample()
    return torch.cat((x,y.view(batch_size, num_points, 1)), 2).transpose(1,0)


class SimpleModel(nn.Module):

    def __init__(self, obs_size, num_outputs, model_config):
        nn.Module.__init__(self)
        self.obs_size = obs_size
        self.prelstm = nn.ModuleList()
        lstminp = self.obs_size
        if model_config['prelstm']:
            self.prelstm.append(nn.Linear(self.obs_size, model_config['prelstm'][0], bias=True))
            lstminp = model_config['prelstm'][-1]
            for i in range(0, len(model_config['prelstm']) - 1):
                self.prelstm.append(nn.Linear(model_config['prelstm'][i], model_config['prelstm'][i + 1], bias=True))
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.lstm = nn.LSTM(lstminp, self.rnn_hidden_dim)

        self.postlstm = nn.ModuleList()
        lstmout = self.rnn_hidden_dim
        if model_config['postlstm']:
            self.postlstm.append(nn.Linear(self.rnn_hidden_dim+1, model_config['postlstm'][0], bias=True))
            lstmout = model_config['postlstm'][-1]
            for i in range(0, len(model_config['postlstm']) - 1):
                self.postlstm.append(nn.Linear(model_config['postlstm'][i], model_config['postlstm'][i + 1], bias=True))
        self.fcout = nn.Linear(lstmout, num_outputs)

    def forward(self, series):
        x = series[:-1]
        targetx = series[1:, :, :1]
        for layer in self.prelstm:
            x = F.relu(layer(x))
        x, h = self.lstm(x, None)
        x = torch.cat((x, targetx), dim=2)
        for layer in self.postlstm:
            x = F.relu(layer(x))
        out = self.fcout(x)
        return out#, h




def train(episodes, batch_size, model, opt):
    for i in range(episodes):
        model.zero_grad()
        data = make_points(batch_size)
        out = model(data)
        loss = F.mse_loss(out, data[1:, :, 1:])
        loss.backward()
        opt.step()
        if i%1000==0:
            with torch.no_grad():
                print('Epsisode: ',i,' Loss: ', loss)
                print((out-data[1:, :, 1:]).abs().mean(dim=1).squeeze())
