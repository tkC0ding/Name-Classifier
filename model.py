import torch
from torch import nn
import pickle

data_file = 'data/preprocessed.pkl'

# Loading data
with open(data_file, 'rb') as file:
    data = pickle.load(file)


# Bulding the RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, input, hidden):
        input_tensor = torch.cat([input, hidden], 1)

        h = self.input_layer(input_tensor)
        out = self.hidden_layer(h)
        return((out, h))
    