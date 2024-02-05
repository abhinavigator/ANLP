#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import numpy as np
import math


# In[2]:


# Read the text file (replace with your own path)
with open('/kaggle/input/auguste/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    text = file.read()


# In[3]:


# Tokenize and clean the text
tokens = text.lower().split()
tokens = [word.strip('.,!?;()[]{}"\'') for word in tokens]
tokens = [word for word in tokens if word]


# In[4]:


# Build vocabulary
word_counts = Counter(tokens)
vocab = {word: i for i, (word, _) in enumerate(word_counts.items())}
vocab_size = len(vocab)


# In[5]:


# Convert tokens to numerical indices
token_indices = [vocab.get(token, vocab.get('<UNK>')) for token in tokens]


# In[6]:


# Load pre-trained word2vec model
word2vec = KeyedVectors.load_word2vec_format('/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin', binary=True)


# In[7]:


# Create a weight matrix for words in training docs
embedding_dim = 300  # Assuming word2vec dim is 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in vocab.items():
    try:
        embedding_vector = word2vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        pass


# In[8]:


# ----- Create 5-gram Sequences -----

context_size = 4
sequences = np.array([token_indices[i:i + context_size + 1] for i in range(len(token_indices) - context_size)])
X = sequences[:, :-1]
y = sequences[:, -1]


# In[9]:


# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)  # 90% training, 10% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)  # 10k validation, 20k test


# In[10]:


vocab_size


# In[11]:


# Create DataLoader
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)


# In[12]:


train_data


# In[13]:


# Step 1: Define the Model Architecture

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LanguageModel, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM Layer(s)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Linear Layer
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        
        return out, hidden



# In[14]:


vocab_size
embedding_dim = 300
hidden_dim = 256
num_layers = 1


# In[15]:


model = LanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)


# In[20]:


if torch.cuda.is_available():
    model = model.cuda()


# In[27]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    model = model.cuda()

for epoch in range(5):
#     hidden = None
    
    for batch_X, batch_y in train_loader:
#         batch_size = batch_X.size(0)
        
#         hidden = (torch.zeros(num_layers, batch_size, hidden_dim).to(batch_X.device),
#                   torch.zeros(num_layers, batch_size, hidden_dim).to(batch_X.device))
        
        
        
        
        if torch.cuda.is_available():
            batch_X , batch_y = batch_X.cuda(), batch_y.cuda()
            
        batch_size = batch_X.size(0)
        
        hidden = (torch.zeros(num_layers, batch_size, hidden_dim).to(batch_X.device),
                  torch.zeros(num_layers, batch_size, hidden_dim).to(batch_X.device))
        optimizer.zero_grad()
        
        out, hidden = model(batch_X, hidden)
        
        hidden = (hidden[0].detach(), hidden[1].detach())
        
        # Compute loss
        loss = criterion(out[:, -1, :], batch_y)  # Use output corresponding to last token in sequence
        
#         loss = criterion(out.squeeze(1), batch_y)
        
        loss.backward()
        
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# In[28]:


# Create DataLoaders for validation and test sets
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
val_loader = DataLoader(val_data, batch_size=128)


# In[29]:


test_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_data, batch_size=128)


# In[34]:





# In[36]:


import math

def calculate_perplexity(data_loader, model, criterion, device='cuda:0'):
    model.eval()
    total_loss = 0
    total_count = 0
    model.to(device)
    with torch.no_grad():
        hidden = None
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Initialize hidden state dynamically based on batch size
            batch_size = batch_X.size(0)
            hidden = (torch.zeros(num_layers, batch_size, hidden_dim).to(batch_X.device),
                      torch.zeros(num_layers, batch_size, hidden_dim).to(batch_X.device))
            
            out, hidden = model(batch_X, hidden)
            
            # Note: No need to squeeze 'out' here
            loss = criterion(out[:, -1, :], batch_y)  # Use output corresponding to last token in sequence
            
            total_loss += loss.item() * len(batch_y)
            total_count += len(batch_y)
            
    return math.exp(total_loss / total_count)  # Fixed typo: "lotal_count" to "total_count"


# In[40]:


# Calculate perplexity
val_perplexity = calculate_perplexity(val_loader, model,criterion )
test_perplexity = calculate_perplexity(test_loader, model, criterion)
train_perplexity = calculate_perplexity(train_loader, model , criterion)


# In[41]:


print(f"Validation Perplexity: {val_perplexity:.2f}")
print(f"Test Perplexity: {test_perplexity:.2f}")
print(f"Train Perplexity: {train_perplexity:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:




