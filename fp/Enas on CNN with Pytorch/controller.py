import torch
from torch import nn
import torch.nn.functional as F


class ControllerNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, num_layer_types):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_layer_types = num_layer_types
        #####
        self.fc1 = nn.Linear(self.hidden_size, self.num_layer_types)
        self.fc2 = nn.Linear(self.hidden_size, self.num_layers)
        #
        self.lstm = nn.LSTM(input_size=self.num_layers + self.num_layer_types, hidden_size=self.hidden_size,
                            num_layers=1)
        ####

    def forward(self):
        # initialize zero inputs
        # torch.autograd.set_detect_anomaly(True)
        previous_h = torch.zeros(1, 1, self.hidden_size)
        previous_c = torch.zeros(1, 1, self.hidden_size)
        previous_soft_1 = torch.zeros(1, 1, self.num_layer_types)
        previous_soft_2 = torch.zeros(1, 1, self.num_layers)
        #
        soft_1_list = []
        soft_2_list = []
        #
        n = (self.num_layers * 2) # first layer connected to input (no other layer connected to input) & the last layer is (global average pooling + softmax layer)
        for i in range(n):
            input = torch.cat((previous_soft_1, previous_soft_2), 2)
            out, (previous_h, previous_c) = self.lstm(input, (previous_h, previous_c))
            out = out.view(-1, self.hidden_size)
            ##### block one --> choosing layer type #####
            previous_soft_1 = F.softmax(self.fc1(out), dim=1).reshape(1, 1, -1)
            soft_1_list.append(previous_soft_1)
            ##### block two --> choosing connections to previous layers #####
            sigmoid_const = int(i / 2)
            previous_soft_2 = torch.log(1 / (1 + sigmoid_const * torch.exp(-self.fc2(out)))).reshape(1, 1, -1)
            # previous_soft_2 = F.logsigmoid(self.fc2(out)).reshape(1, 1, -1)
            soft_2_list.append(previous_soft_2)
        soft_1 = torch.cat(soft_1_list, dim=0)
        soft_2 = torch.cat(soft_2_list, dim=0)
        return soft_1, soft_2
