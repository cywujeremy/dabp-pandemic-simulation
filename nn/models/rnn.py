import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    
    def __init__(self, num_layers, num_hidden, bidirectional=False):
        super(Network, self).__init__()
        
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        
        if bidirectional:
            self.bidirectional_multiplier = 2
        else:
            self.bidirectional_multiplier = 1
        
        self.rnn = nn.LSTM(input_size=3, 
                           hidden_size=self.num_hidden,
                           num_layers=self.num_layers,
                           dropout=0.5,
                           bidirectional=bidirectional)
        
        self.reg = nn.Sequential(*[nn.Linear(self.num_hidden * self.bidirectional_multiplier, 2048),
                                   nn.Dropout(0.3),
                                   nn.Linear(2048, 1)])
    
    def forward(self, X):
        
        out, _ = self.rnn(X)
        out = self.reg(out)
        
        return out