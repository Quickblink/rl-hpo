import torch.nn as nn
import torch
import torch.nn.functional as F





class MyModel(nn.Module):
    """The default RNN model for QMIX."""

    def __init__(self, obs_size, num_outputs, model_config):
        nn.Module.__init__(self)
        #model_config = model_config['custom_options']  # print('HELLO', model_config)
        self.obs_size = obs_size
        self.prelstm = nn.ModuleList()
        lstminp = self.obs_size
        if model_config['prelstm']:
            self.prelstm.append(nn.Linear(self.obs_size, model_config['prelstm'][0], bias=True))
            lstminp = model_config['prelstm'][-1]
            for i in range(0, len(model_config['prelstm']) - 1):
                self.prelstm.append(nn.Linear(model_config['prelstm'][i], model_config['prelstm'][i + 1], bias=True))
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        # self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.lstm = nn.LSTM(lstminp, self.rnn_hidden_dim)

        self.postlstm = nn.ModuleList()
        lstmout = self.rnn_hidden_dim
        if model_config['postlstm']:
            self.postlstm.append(nn.Linear(self.rnn_hidden_dim, model_config['postlstm'][0], bias=True))
            lstmout = model_config['postlstm'][-1]
            for i in range(0, len(model_config['postlstm']) - 1):
                self.postlstm.append(nn.Linear(model_config['postlstm'][i], model_config['postlstm'][i + 1], bias=True))

        lstmout = 32
        self.ll1 = nn.Linear(self.rnn_hidden_dim, 64)
        self.ll2 = nn.Linear(64, 32)
        self.ll3 = nn.Linear(64, 32)
        self.fvar = nn.Linear(32, 1)
        self.fcout = nn.Linear(lstmout, num_outputs)
        self.valuef = nn.Linear(lstmout, 1)


    def forward(self, x, hidden_state):
        #x = input_dict["obs_flat"].float()
        bsz = x.shape[0]
        # x = nn.functional.relu(self.fc1(input_dict["obs_flat"].float()))
        for layer in self.prelstm:
            x = F.relu(layer(x))
        x, h = self.lstm(x.view(1, bsz, self.rnn_hidden_dim), hidden_state)
        for layer in self.postlstm:
            x = F.relu(layer(x.view(bsz, -1)))
        # no ReLu activation in the output layer
        x = F.relu(self.ll1(x.view(bsz, -1)))
        v = F.relu(self.ll3(x))
        x = F.relu(self.ll2(x))
        a = torch.sigmoid(self.fcout(x))
        var = torch.sigmoid(self.fvar(x))
        v = self.valuef(v)
        return a, v, h, var


