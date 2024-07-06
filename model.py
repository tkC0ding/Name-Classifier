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

model = RNN(57, 128, data[0][1].size(1))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

def train(model, optimizer, loss_fn):
    model.train()
    loss_sum = 0
    for item in data:
        hidden = model.hidden_init().to(device)
        X = item[0].to(device)
        Y = item[1].to(device)
        for inp in X:
            output, hidden = model(inp, hidden)
        loss = loss_fn(output, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    avg_loss = loss_sum/len(data)
    return(avg_loss)
