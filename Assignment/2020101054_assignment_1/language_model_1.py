#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import numpy as np
import math


# In[4]:


# Read the text file (replace with your own path)
with open('/kaggle/input/auguste/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    text = file.read()


# In[5]:


# Tokenize and clean the text
tokens = text.lower().split()
tokens = [word.strip('.,!?;()[]{}"\'') for word in tokens]
tokens = [word for word in tokens if word]


# In[6]:


# Build vocabulary
word_counts = Counter(tokens)
vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
vocab_size = len(vocab)


# In[7]:


# Convert tokens to numerical indices
token_indices = [vocab.get(token, vocab.get('<UNK>')) for token in tokens]


# In[8]:


# Load pre-trained word2vec model
word2vec = KeyedVectors.load_word2vec_format('/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)


# In[9]:


# Create a weight matrix for words in training docs
embedding_dim = 300  # Assuming word2vec dim is 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in vocab.items():
    try:
        embedding_vector = word2vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        pass


# In[10]:


# ----- Create 5-gram Sequences -----

context_size = 4
sequences = np.array([token_indices[i:i + context_size + 1] for i in range(len(token_indices) - context_size)])
X = sequences[:, :-1]
y = sequences[:, -1]


# In[11]:


# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)  # 90% training, 10% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)  # 10k validation, 20k test


# In[12]:


# ----- Model Architecture -----

class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NeuralLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# In[13]:


# Initialize the model, loss, and optimizer
model = NeuralLM(vocab_size, embed_dim=embedding_dim, hidden_dim=300)
model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[14]:


model


# In[15]:


# Create DataLoader
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)


# In[16]:


# # Training loop
# for epoch in range(5):  # 5 epochs for demonstration, you can adjust this
#     for i, (x_batch, y_batch) in enumerate(train_loader):
#         # Forward pass
#         outputs = model(x_batch)
#         loss = criterion(outputs, y_batch)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Print loss every 100 batches
#         if (i + 1) % 100 == 0:
#             print(f"Epoch [{epoch+1}/5], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


# In[17]:


# Move model to GPU
if torch.cuda.is_available():
    model = model.cuda()

# # Inside your training loop
# if torch.cuda.is_available():
#     x_batch, y_batch = x_batch.cuda(), y_batch.cuda()


# In[18]:


for epoch in range(1):  # Reduced to 1 epoch for faster training
    for i, (x_batch, y_batch) in enumerate(train_loader):
        
        # Move data to GPU if available
        if torch.cuda.is_available():
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/1], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


# In[19]:


# def calculate_perplexity(data_loader):
#     total_loss = 0
#     total_count = 0
#     model.eval()
#     with torch.no_grad():
#         for x_batch, y_batch in data_loader:
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             total_loss += loss.item() * len(y_batch)
#             total_count += len(y_batch)
#     return math.exp(total_loss / total_count)



# In[25]:


def calculate_perplexity(data_loader, device='cuda:0'):
    total_loss = 0
    total_count = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)
    return math.exp(total_loss / total_count)


# In[26]:


# Create DataLoaders for validation and test sets
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
val_loader = DataLoader(val_data, batch_size=128)


# In[27]:


test_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_data, batch_size=128)


# In[28]:


# Calculate perplexity
val_perplexity = calculate_perplexity(val_loader)
test_perplexity = calculate_perplexity(test_loader)
train_perplexity = calculate_perplexity(train_loader)


# In[29]:


print(f"Validation Perplexity: {val_perplexity:.2f}")
print(f"Test Perplexity: {test_perplexity:.2f}")
print(f"Train Perplexity: {train_perplexity:.2f}")


# In[ ]:




