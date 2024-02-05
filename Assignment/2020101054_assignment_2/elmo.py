#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import numpy as np
import  string
import nltk.data
import nltk
import time
import pandas as pd
import random
import numpy as np
import torch
import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import codecs
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from gensim import models
from torch.utils.data import random_split


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


# Load the dataset
file_path = '/kaggle/input/anlp-assignment-2/train.csv'

dataset = pd.read_csv(file_path)
text = dataset['Description']
label = dataset['Class Index']
len(dataset)
text[0]


# In[ ]:


train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]

val_dataset= dataset[train_size:]
type(val_dataset)


# In[ ]:


train_text = train_dataset['Description'].tolist()
train_label = train_dataset['Class Index'].tolist()

val_text = val_dataset['Description'].tolist()
val_label = val_dataset['Class Index'].tolist()
type(train_text)


# In[ ]:


val_text[0]


# In[ ]:


train_label[0]


# In[ ]:


def cleaning_data(text):
    #for text in lines:
    text = text.lower()
    
    text = text.encode('ascii', 'ignore').decode()
    
    text = re.sub("\'\w+", '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', text)
    text = re.sub("#[A-Za-z0-9_]+", '', text)
    text = re.sub("[0-9_]+", "", text)
    text = re.sub(r"[^\w\s\0-9.]+|\.(?=.*\.)", "", text)
    text = re.sub(r'[^\w\s]', " ", text)
    
    # Remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    
    text = re.sub('\s{2,}', " ", text)
    return text


# In[ ]:


def preprocessed(data):
    final = []
    for sent in (data):
        sentence = cleaning_data(sent)
        tokens = nltk.word_tokenize(sentence)
        final.append(tokens)
    return final


# In[ ]:


train_data = preprocessed(train_text)


# In[ ]:


val_data = preprocessed(val_text)


# In[ ]:





# In[ ]:


train_text.iloc[0]


# In[ ]:


def load_glove_model(glove_file):

    f = codecs.open(glove_file,'r', encoding='utf-8')
    glove_dict = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        glove_dict[word] = embedding
    glove_dict['<pad>'] = np.zeros(200)
    glove_dict['<s>'] = np.ones(200)
    glove_dict['</s>'] = np.ones(100)*2
    glove_dict['<unk>'] = np.ones(100)*3
    return glove_dict


# In[ ]:


glove_dict = load_glove_model('/kaggle/input/nlpword2vecembeddingspretrained/glove.6B.200d.txt')


# In[ ]:





# In[ ]:


vocab = torchtext.vocab.build_vocab_from_iterator(train_data, min_freq=2)


# In[ ]:


len(vocab)


# In[ ]:


vocab.append_token('<pad>')
vocab.append_token('<s>')
vocab.append_token('</s>')
vocab.append_token('<unk>')
vocab.set_default_index(vocab["<unk>"])


# In[ ]:


def get_data(dataset):
    data = []
    x = []
    for sent in dataset:
        if len(sent)!=0:
            sent = ['<s>'] +sent + ['</s>']
            w =[]
            for i in sent:
                if i not in glove_dict:
                    w.append('<unk>')
                else:
                    w.append(i)

            x.append(w)
    return x


# In[ ]:


train_x = get_data(train_data)


# In[ ]:


val_x = get_data(val_data)


# In[ ]:


class Data_set(Dataset):
    
    def __init__(self,X, vocab, Y=None, classify = False):
        self.X = X
        self.Y = Y
        self.vocab = vocab
        self.classify = classify
            
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self,idx):
        self.X[idx] = np.array(self.X[idx])
        x = []
        y = []
        if self.classify:
            y.append(self.Y[idx])
            
        for i in range(len(self.X[idx])-1):
            final = glove_dict[self.X[idx][i]]
            if not self.classify:
                y.append(self.vocab[self.X[idx][i+1]])
            x.append(final)

        x = np.array(x)
        y = np.array(y)
        return x, y


# In[ ]:


def Padding(data):
    x = []
    y = []
  
    for i in range(len(data)):
#         print(type(data[i][0]))  # Check the type
#         print(data[i][0])  # Print the content

        x.append(torch.tensor(data[i][0]))
        y.append(torch.tensor(data[i][1],dtype=torch.float32))
    inputs = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value = 0)
    lbl = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value = 0)
    return inputs, lbl


# def Padding(data):
#     x = []
#     y = []
#     for i in range(len(data)):
#         # Convert each numpy array in the list to a tensor separately
#         for arr in data[i][0]:
#             x.append(torch.tensor(arr, dtype=torch.float32))
#         y.append(torch.tensor(data[i][1], dtype=torch.float32))
#     inputs = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
#     targets = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0.0)
#     return inputs, targets


# In[ ]:


dataset_train = Data_set(train_x, vocab)


# In[ ]:


dataset_val = Data_set(val_x, vocab)


# In[ ]:


train_loader = DataLoader(dataset_train, batch_size=8, collate_fn = Padding)


# In[ ]:


val_loader = DataLoader(dataset_val, batch_size=8, collate_fn = Padding)


# In[ ]:


class eLMO(nn.Module):
    def __init__(self,input_dim, vocab_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.vocab_size = vocab_size
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, batch_first =True, bidirectional = True, num_layers =1)
        self.lstm2 = torch.nn.LSTM(2*self.hidden_dim, self.hidden_dim, batch_first =True, bidirectional = True, num_layers =1)
        self.fc = nn.Linear(self.input_dim, self.vocab_size)

    def forward(self, x):
        out1, (h_t1,c1) = self.lstm(x)
        out2, (h_t2,c2) = self.lstm2(out1)
        print(out2.shape)  # Print the shape of out2
        if len(out2.shape) == 2:  # Check if the sequence length is 1
            out2 = out2.unsqueeze(1)  # Add the sequence length dimension back
        pred1 = self.fc(out2[:,:,:self.input_dim])
        pred2 = self.fc(out2[:,:,self.input_dim:])
        return (out1, out2), (pred1, pred2)


# In[ ]:


input_dim = 200
learning_rate = 0.01
criterion = nn.NLLLoss()

model = eLMO(input_dim, len(vocab))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


def training(model, dataloader, optimizer, loss_fn, device, cls =False):
    e_loss = 0
    e_acc = 0
    
    model.train()
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if not cls:
            out, preds  = model(x.float())
            output1 = torch.permute(F.log_softmax(preds[0], dim = -1), (0, 2,1) )
            output2 = torch.permute(F.log_softmax(preds[1], dim = -1), (0, 2,1) )
            loss1 = loss_fn(output1, y)
            loss2 = loss_fn(output2, y)
            loss = (loss1 + loss2)/2
        else:
            out = model(x.float())
            loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        e_loss += loss.item()
    return e_loss/len(dataloader)


# In[ ]:


def validation(model, dataloader, loss_fn, device, cls =False):
    e_loss = 0
    e_acc = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if not cls:
                out, preds  = model(x.float())
                output1 = torch.permute(F.log_softmax(preds[0], dim = -1), (0, 2,1) )
                output2 = torch.permute(F.log_softmax(preds[1], dim = -1), (0, 2,1) )
                loss1 = loss_fn(output1, y)
                loss2 = loss_fn(output2, y)
                loss = (loss1 + loss2)/2
            else:
                out = model(x.float())
                loss = loss_fn(out, y)
            
            e_loss += loss.item()
    return e_loss/len(dataloader)


# In[ ]:


def training_steps(epochs, model, train_loader, val_loader, criterian, optimizer, device, cls = False):
    min_loss = np.inf
    train_losses = []
    val_losses = []
    x_s = []
    for epoch in range(epochs):
        x_s.append(epoch)
        start_time = time.time()
        train_loss = training(model, train_loader, optimizer, criterian, device,cls)
        train_losses.append(train_loss)
        train_time = time.time()
        val_loss = validation(model, val_loader, criterian, device,cls)
        val_losses.append(val_loss)
        
        torch.save(model, "Trained model.pth")
    return train_losses, val_losses


train_losses, val_losses = training_steps(20, model, train_loader, val_loader, criterion, optimizer, device)

np.save(np.array(train_losses))
np.save(np.array(val_losses))


# In[ ]:


torch.save(model, "/kaggle/working/Trained model.pth")


# # downstream task

# In[ ]:


# Load the dataset
file_path = '/kaggle/input/anlp-assignment-2/test.csv'

dataset_test = pd.read_csv(file_path)
test_text = dataset['Description'].to_list()
test_label = dataset['Class Index'].to_list()


# In[ ]:


def Padding_classify(data):
    x = []
    y = []
    for i in range(len(data)):
        x.append(torch.tensor(data[i][0]))
        y.append(data[i][1][0])
    inputs = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value = 0)
    return inputs, torch.tensor(y)


# In[ ]:


class Classify(nn.Module):
    def __init__(self,input_dim, model):
        super().__init__()
        self.input_dim = input_dim
        self.model = model
        self.s0 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.s1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.s2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.y = nn.Parameter(torch.ones(1, requires_grad=True))
        self.fc1 = nn.Linear(2*self.input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 5)
        
    def forward(self, x):
        out, pred = self.model(x)
        s0, s1, s2 = F.softmax(torch.tensor([self.s0,self.s1,self.s2]), dim = 0)
        elmo = self.y * (torch.cat((x,x),-1)*s0 + out[0]*s1 + out[1]*s2)
        emd = elmo.mean(dim = 1)
        out = self.fc1(emd)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# In[ ]:


model_classify = classify(200,model )
model_classify = model_classify.to(device)


# In[ ]:


dataset = Data_set(train_x, vocab,train_lbl, True )
dataset_val = Data_set(val_x, vocab, val_lbl, True )
cl_loader = DataLoader(dataset, batch_size=16, collate_fn = Padding_classify)
cl_val_loader = DataLoader(dataset_val, batch_size=16, collate_fn = Padding_classify)


# In[ ]:


# HYperParameters
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


train_losses, val_losses = training_steps(10, model_cls, cl_val_loader, cl_val_loader, criterion, optimizer, device, True)


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Evaluate the model on the test set
correct = 0
total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['text'].to(device)
        targets = batch['label'].to(device)
        outputs = Classify(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calculate Metrics
accuracy = accuracy_score(all_targets, all_preds)
micro_f1 = f1_score(all_targets, all_preds, average='micro')
print("Accuracy: ", accuracy)
print("Micro F1 Score: ", micro_f1)

# Confusion Matrix
conf_matrix = confusion_matrix(all_targets, all_preds)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

