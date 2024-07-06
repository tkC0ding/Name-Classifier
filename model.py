import torch
from torch import nn
import pickle

data_file = 'data/preprocessed.pkl'

# Loading data
with open(data_file, 'rb') as file:
    data = pickle.load(file)

# Device

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Bulding the RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size+hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        nn.ModuleList([self.input_layer, self.output_layer, self.relu])
    
    def forward(self, x, hidden):
        input_tensor = torch.cat([x, hidden], 1)

        o = self.input_layer(input_tensor)
        h = self.relu(o)
        output = self.output_layer(h)

        return((output, h))
    
    def hidden_init(self):
        return(torch.zeros(1, self.hidden_size))
