import torch
from torch import nn
import pickle
import matplotlib.pyplot as plt

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

model = RNN(55, 128, data[0][1].size(1)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

# helper function to train the model
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

# Training the model
epochs = 10
losses = []
for i in range(epochs):
    loss = train(model, optimizer, loss_fn)
    losses.append(loss)
    print(f"Epoch : {i+1}\tLoss : {loss}")

# Plotting the Loss
epochs_range = list(range(10))
fig, ax = plt.subplots(1, 1)
ax.plot(epochs, losses)
ax.set_xlabel('Epochs')
ax.set_ylabel('Losses')
ax.set_title('Monitoring the Loss')
fig.show()

# Saving the Model

torch.save(model.state_dict(), 'SavedModel/modddel_weights.pth')
torch.save(model, 'SavedModel/model.pth')