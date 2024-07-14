import torch
from torch import nn
import pickle
import string
import unicodedata
import os

data_file = 'data/preprocessed.pkl'

ALL_LETTERS = string.ascii_letters + '.,;'
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s):
    return ''.join(i for i in unicodedata.normalize('NFD', s) if unicodedata.category(i) and i in ALL_LETTERS)

def vectorizer(s):
    tensor = torch.zeros(len(s), 1, len(ALL_LETTERS))
    for i in range(len(s)):
        index = ALL_LETTERS.find(s[i])
        tensor[i][0][index] = 1
    return tensor

with open(data_file, 'rb') as file:
    data = pickle.load(file)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x, hidden):
        input_tensor = torch.cat([x, hidden], 1)
        o = self.input_layer(input_tensor)
        h = self.relu(o)
        output = self.output_layer(h)
        return output, h
    
    def hidden_init(self):
        return torch.zeros(1, self.hidden_size)

model = torch.load('SavedModel/model.pth', map_location=device)
model.eval()

categories = []
for filename in os.listdir('data/'):
    categories.append(filename.split('.')[0])

def predict(model, inp):
    inp = vectorizer(unicode_to_ascii(inp.strip())).to(device)
    hidden = model.hidden_init().to(device)
    for i in inp:
        output, hidden = model(i, hidden)
    score, prediction = torch.max(output, 1)
    print(f"{score.item()}\t{categories[prediction.item()]}")

while True:
    a = input("Enter the name : ")
    if(a != 'quit'):
        predict(model, a)
    else:
        break
