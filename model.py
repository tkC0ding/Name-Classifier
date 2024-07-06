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
        self.input_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        nn.ModuleList([self.input_layer, self.hidden_layer, self.relu])
    
    def forward(self, input, hidden):
        input_tensor = torch.cat([input, hidden], 1)

        h = self.input_layer(input_tensor)
        out = self.hidden_layer(h)
        return((out, h))
    
    def hidden_init(self):
        return(torch.zeros(1, self.hidden_size))

input_size = 57
hidden_size = 128
output_size = data[0][1].size(1)

model = RNN(input_size, hidden_size, output_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

data = data[:20070]

batched_data = []
for i in range(2007):
    temp = []
    for j in range(10):
        temp.append(data[j])
    batched_data.append(temp)

