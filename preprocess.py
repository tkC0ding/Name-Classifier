import os
import string
import unicodedata
import torch
import random
import pickle

data_path = 'data/'
data_save_dir = 'data/'
ALL_LETTERS = string.ascii_letters + '.,;'
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s):
    return ''.join(i for i in unicodedata.normalize('NFD', s) if unicodedata.category(i) and i in ALL_LETTERS)

def vectorizer(s):
    tensor = torch.zeros(len(s), 1, len(ALL_LETTERS))
    for i in range(len(s)):
        index = ALL_LETTERS.find(s[i])
        tensor[i][0][index] = 1
    return(tensor)

def label_vectorizer(label_list, label):
    tensor = torch.zeros(1, len(label_list))
    index = label_list.index(label)
    tensor[0][index] = 1
    return(tensor)

categories = []
for item in os.listdir(data_path):
    categories.append(item.split('.')[0])

data = []
for filename in os.listdir(data_path):
    file = open(f'{data_path}{filename}')
    data_list = file.read().strip().split('\n')
    label = filename.split('.')[0]
    label_tensor = label_vectorizer(categories, label)
    for i in data_list:
        data.append((vectorizer(unicode_to_ascii(i)), label_tensor))

random.shuffle(data)

with open(f'{data_save_dir}preprocessed.pkl', 'wb') as file:
    pickle.dump(data, file)