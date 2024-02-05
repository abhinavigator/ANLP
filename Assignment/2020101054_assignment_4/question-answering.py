#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import nn
import random
import json

from torch.utils.data import DataLoader
from torch.nn import DataParallel


# In[ ]:





# In[ ]:





# In[2]:


# squad_dict


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


import torch
from torch.utils.data import Dataset
import json
from transformers import GPT2Tokenizer

class QADataset(Dataset):
    def __init__(self, filename, tokenizer, max_length , fraction = 0.1):
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.answers = []

        # Load and parse the dataset
        with open(filename, 'r') as file:
            data = json.load(file)
        
        num_articles = int(len(data['data'])*fraction)
        data['data'] = data['data'][:num_articles]
            

        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text'] if not qa['is_impossible'] else '[NO ANSWER]'

                    # Format the input
                    input_text = f"Context: {context} Question: {question} Answer:"
                    self.inputs.append(input_text)
                    self.answers.append(answer)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        answer_text = self.answers[idx]
        input_text = f"[QUESTION] {input_text} [ANSWER]"

        # Encode the inputs and answers
        input_encoding = self.tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        answer_encoding = self.tokenizer.encode_plus(answer_text, add_special_tokens=True, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)

        # Return as a dictionary
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': answer_encoding['input_ids'].flatten()
        }

# # Usage
# fraction = 0.1
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # tokenizer.pad_token = tokenizer.eos_token
# max_length = 512  # or whatever the model max length is
# dataset = QADataset('/kaggle/input/squad-20/train-v2.0.json', tokenizer, max_length)


# In[4]:


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

special_tokens = {'additional_special_tokens': ['[QUESTION]', '[ANSWER]']}
tokenizer.add_special_tokens(special_tokens)

# Usage
fraction = 0.1

# tokenizer.pad_token = tokenizer.eos_token
max_length = 512  # or whatever the model max length is
dataset = QADataset('/kaggle/input/squad-20/train-v2.0.json', tokenizer, max_length)


# In[5]:


len(dataset)


# In[6]:


val_dataset = QADataset('/kaggle/input/squad-20/dev-v2.0.json', tokenizer, max_length)


# In[7]:


len(val_dataset)


# In[9]:


batch_size = 8  # or any suitable batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




# In[10]:


val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# In[11]:


model = GPT2LMHeadModel.from_pretrained('gpt2')


model.resize_token_embeddings(len(tokenizer))


# In[12]:


if torch.cuda.device_count()>1:
    print("using", torch.cuda.device_count(),"GPUs!")
    model = DataParallel(model)


# In[13]:


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# In[14]:


model = model.to(device)


# In[15]:


from torch.optim import AdamW

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
number_of_epochs = 1  # Define the number of epochs


# In[ ]:


# for epoch in range(number_of_epochs):
#     model.train()
#     total_loss = 0  # Initialize total loss for each epoch
#     total_batches = len(dataloader)  
#     for i, batch in enumerate(dataloader):
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         total_loss += loss.item()  # Accumulate the loss
#         loss.backward()
#         optimizer.step()

#         if (i + 1) % 100 == 0:  # Print loss every 10 batches
#             print(f'Epoch {epoch + 1}/{number_of_epochs}, Batch {i + 1}/{total_batches}, Loss: {loss.item():.4f}')

#     # Print the average loss after each epoch
#     average_loss = total_loss / len(dataloader)
#     print(f'End of Epoch {epoch + 1}/{number_of_epochs}, Average Loss: {average_loss:.4f}')


# In[ ]:


for epoch in range(number_of_epochs):
    model.train()
    total_loss = 0
    total_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Check if the loss is a scalar; if not, take the mean
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{number_of_epochs}, Batch {i + 1}/{total_batches}, Loss: {loss.item():.4f}')

    average_loss = total_loss / total_batches
    print(f'End of Epoch {epoch + 1}/{number_of_epochs}, Average Loss: {average_loss:.4f}')


# In[ ]:


# def validate(model, dataloader, device):
#     model.eval()  # Set the model to evaluation mode
#     total_loss = 0
#     total_batches = len(dataloader)

#     with torch.no_grad():  # Disable gradient calculation
#         for i, batch in enumerate(dataloader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()

#             if (i + 1) % 10 == 0:  # Print validation loss every 10 batches
#                 print(f'Validation Batch {i + 1}/{total_batches}, Loss: {loss.item():.4f}')

#     average_loss = total_loss / total_batches
#     return average_loss


# In[ ]:


# Save the model
model.module.save_pretrained('/kaggle/working/model.pth')
tokenizer.save_pretrained('/kaggle/working/tokenizer.pth')


# In[ ]:





# In[ ]:


# generated_text


# In[17]:


model = GPT2LMHeadModel.from_pretrained('/kaggle/working/model.pth')
# tokenizer = GPT2Tokenizer.from_pretrained('/kaggle/working/tokenizer.pth')
# from torch.utils.data import DataLoader

# # Assuming validation_dataset is an instance of a Dataset class similar to your training dataset
# validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# In[18]:


model = model.to(device)


# In[20]:


# Example: Generating text and comparing with ground truth (e.g., for BLEU score)
from nltk.translate.bleu_score import corpus_bleu

references = []
candidates = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Generating text
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50)

        # Decode the outputs to text
        decoded_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
        decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in batch['labels']]

        candidates.extend(decoded_preds)
        references.extend([[label.split()] for label in decoded_labels])

# Calculate BLEU score
bleu_score = corpus_bleu(references, [candidate.split() for candidate in candidates])
print(f'BLEU score: {bleu_score}')


# In[ ]:





# In[ ]:




