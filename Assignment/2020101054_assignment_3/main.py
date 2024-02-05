#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from transformer import Transformer # this is the transformer.py file
import torch
import numpy as np


# In[2]:


import sys
sys.path.append("/kaggle/input/check-transformer")

from transformer import Transformer


# In[3]:


# from transformer_test import Transformer


# In[4]:


train_english_file = '/kaggle/input/assignment-3-dataset/ted-talks-corpus/train.en'
train_french_file ='/kaggle/input/assignment-3-dataset/ted-talks-corpus/train.fr'


# In[5]:


valid_english_file = '/kaggle/input/assignment-3-dataset/ted-talks-corpus/dev.en'
valid_french_file = '/kaggle/input/assignment-3-dataset/ted-talks-corpus/dev.fr'


# In[6]:


test_english_file = '/kaggle/input/assignment-3-dataset/ted-talks-corpus/test.en'
test_french_file = '/kaggle/input/assignment-3-dataset/ted-talks-corpus/test.fr'


# In[7]:


train_french_file


# In[8]:


valid_french_file


# 

# In[9]:


START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'


# In[10]:


english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]


french_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     ':', '<', '=', '>', '?', '@',
                     '[', '\\', ']', '^', '_', '`', 
                     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                     'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                     'y', 'z', 
                     'à', 'â', 'ç', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ',
                     'œ', 'æ', '€',  # Additional French-specific characters
                     '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]


# In[11]:


index_to_french = {k:v for k,v in enumerate(french_vocabulary)}
french_to_index = {v:k for k,v in enumerate(french_vocabulary)}
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}


# In[ ]:





# In[12]:


with open(train_english_file, 'r') as file:
    train_english_sentences = file.readlines()
with open(train_french_file, 'r') as file:
    train_french_sentences = file.readlines()


# In[13]:


with open(valid_english_file, 'r') as file:
    valid_english_sentences = file.readlines()
with open(valid_french_file, 'r') as file:
    valid_french_sentences = file.readlines()


# In[14]:


with open(test_english_file, 'r') as file:
    test_english_sentences = file.readlines()
with open(test_french_file, 'r') as file:
    test_french_sentences = file.readlines()


# In[15]:


# valid_english_sentences


# In[16]:


# train_french_sentences


# In[17]:


train_english_sentences = [sentence.rstrip('\n').lower() for sentence in train_english_sentences]
train_french_sentences = [sentence.rstrip('\n').lower() for sentence in train_french_sentences]


# In[18]:


valid_english_sentences = [sentence.rstrip('\n').lower() for sentence in valid_english_sentences]
valid_french_sentences = [sentence.rstrip('\n').lower() for sentence in valid_french_sentences]


# In[19]:


test_english_sentences = [sentence.rstrip('\n').lower() for sentence in test_english_sentences]
test_french_sentences = [sentence.rstrip('\n').lower() for sentence in test_french_sentences]


# In[20]:


train_english_sentences[:10]


# In[21]:


valid_english_sentences[:10]


# In[22]:


train_french_sentences[:10]


# In[ ]:





# In[23]:


import numpy as np
PERCENTILE = 97
print( f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in train_english_sentences], PERCENTILE)}" )
print( f"{PERCENTILE}th percentile length french: {np.percentile([len(x) for x in train_french_sentences], PERCENTILE)}" )


# In[24]:


len(train_french_sentences)


# In[25]:


len(test_french_sentences)


# In[26]:


max_sequence_length = 350

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

valid_sentence_indicies = []
for index in range(len(train_french_sentences)):
    train_french_sentence, train_english_sentence = train_french_sentences[index], train_english_sentences[index]
    if is_valid_length(train_french_sentence, max_sequence_length) \
      and is_valid_length(train_english_sentence, max_sequence_length) \
      and is_valid_tokens(train_french_sentence, french_vocabulary):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(train_english_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")


# In[27]:


max_sequence_length = 350

# Function to check if tokens in a sentence are valid in the given vocabulary
def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

# Function to check if the length of a sentence is valid
def is_valid_length(sentence, max_sequence_length):
    return len(sentence) < (max_sequence_length - 1)  # need to re-add the end token so leaving 1 space

# Function to filter sentences and create a valid sentence indices list
def filter_valid_sentences(sentences, vocab, max_length):
    valid_sentence_indices = []
    for index in range(len(sentences)):
        sentence = sentences[index]
        if is_valid_length(sentence, max_length) and is_valid_tokens(sentence, vocab):
            valid_sentence_indices.append(index)
    return valid_sentence_indices

# Filter the training sentences for valid indices
valid_train_sentence_indices = filter_valid_sentences(train_french_sentences, french_vocabulary, max_sequence_length)

# Filter the validation sentences for valid indices
valid_validation_sentence_indices = filter_valid_sentences(valid_french_sentences, french_vocabulary, max_sequence_length)

print(f"Number of training sentences: {len(train_french_sentences)}")
print(f"Number of valid training sentences: {len(valid_train_sentence_indices)}")

print(f"Number of validation sentences: {len(valid_french_sentences)}")
print(f"Number of valid validation sentences: {len(valid_validation_sentence_indices)}")


# In[28]:


test_valid_sentence_indices = filter_valid_sentences(test_french_sentences, french_vocabulary, max_sequence_length)


# In[29]:


print(f"Number of test sentences: {len(test_french_sentences)}")
print(f"Number of test valid sentences: {len(test_valid_sentence_indices)}")


# In[ ]:





# In[30]:


# if the sentence is more than 350 it is not a valid sentence


# In[31]:


train_french_sentences = [train_french_sentences[i] for i in valid_sentence_indicies]
train_english_sentences = [train_english_sentences[i] for i in valid_sentence_indicies]


# In[32]:


valid_french_sentences = [valid_french_sentences[i] for i in valid_validation_sentence_indices]
valid_english_sentences = [valid_english_sentences[i] for i in valid_validation_sentence_indices]


# In[33]:


test_french_sentences = [test_french_sentences[i] for i in test_valid_sentence_indices]
test_english_sentences = [test_english_sentences[i] for i in test_valid_sentence_indices]


# In[34]:


test_english_sentences[:3]


# In[35]:


valid_french_sentences[:2]


# In[ ]:





# In[36]:


train_french_sentences[:3]


# In[37]:


import torch

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 2
max_sequence_length = 350
french_vocab_size = len(french_vocabulary)

transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          french_vocab_size,
                          english_to_index,
                          french_to_index,
                          START_TOKEN, 
                          END_TOKEN, 
                          PADDING_TOKEN)


# In[38]:


transformer


# In[39]:


from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):

    def __init__(self, train_english_sentences, train_french_sentences):
        self.train_english_sentences = train_english_sentences
        self.train_french_sentences = train_french_sentences

    def __len__(self):
        return len(self.train_english_sentences)

    def __getitem__(self, idx):
        return self.train_english_sentences[idx], self.train_french_sentences[idx]


# In[40]:


# dataloader for validation set

class TextDataset(Dataset):

    def __init__(self, valid_english_sentences, valid_french_sentences):
        self.valid_english_sentences = valid_english_sentences
        self.valid_french_sentences = valid_french_sentences

    def __len__(self):
        return len(self.valid_english_sentences)

    def __getitem__(self, idx):
        return self.valid_english_sentences[idx], self.valid_french_sentences[idx]


# In[41]:


# dataloader for test set 
class TextDataset(Dataset):

    def __init__(self, test_english_sentences, test_french_sentences):
        self.test_english_sentences = test_english_sentences
        self.test_french_sentences = test_french_sentences

    def __len__(self):
        return len(self.test_english_sentences)

    def __getitem__(self, idx):
        return self.test_english_sentences[idx], self.test_french_sentences[idx]


# In[42]:


dataset = TextDataset(train_english_sentences, train_french_sentences)


# In[43]:


validation_dataset = TextDataset(valid_english_sentences, valid_french_sentences)


# In[44]:


test_dataset = TextDataset(test_english_sentences, test_french_sentences)


# In[45]:


len(test_dataset)


# In[46]:


len(dataset)


# In[47]:


len(validation_dataset)


# In[48]:


validation_dataset[0]


# In[49]:


train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)


# In[50]:


validation_loader = DataLoader(validation_dataset, batch_size)


# In[51]:


test_loader =  DataLoader(test_dataset, batch_size)


# In[52]:


for batch_num, batch in enumerate(iterator):
#     print(batch)
    if batch_num > 3:
        break


# In[53]:


from torch import nn

criterian = nn.CrossEntropyLoss(ignore_index=french_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[54]:


NEG_INFTY = -1e9

def create_masks(eng_batch, french_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, french_sentence_length = len(eng_batch[idx]), len(french_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      french_chars_to_padding_mask = np.arange(french_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, french_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, french_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, french_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


# Modify mask such that the padding tokens cannot look ahead.
# In Encoder, tokens before it should be -1e9 while tokens after it should be -inf.

# In[55]:


# transformer.train()
# transformer.to(device)
# total_loss = 0
# num_epochs = 2

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch}")
#     iterator = iter(train_loader)
#     for batch_num, batch in enumerate(iterator):
#         transformer.train()
#         eng_batch, french_batch = batch
#         encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
#         optim.zero_grad()
#         french_predictions = transformer(eng_batch,
#                                      french_batch,
#                                      encoder_self_attention_mask.to(device), 
#                                      decoder_self_attention_mask.to(device), 
#                                      decoder_cross_attention_mask.to(device),
#                                      enc_start_token=False,
#                                      enc_end_token=False,
#                                      dec_start_token=True,
#                                      dec_end_token=True)
#         labels = transformer.decoder.sentence_embedding.batch_tokenize(french_batch, start_token=False, end_token=True)
#         loss = criterian(
#             french_predictions.view(-1, french_vocab_size).to(device),
#             labels.view(-1).to(device)
#         ).to(device)
#         valid_indicies = torch.where(labels.view(-1) == french_to_index[PADDING_TOKEN], False, True)
#         loss = loss.sum() / valid_indicies.sum()
#         loss.backward()
#         optim.step()
#         #train_losses.append(loss.item())
#         if batch_num % 100 == 0:
#             print(f"Iteration {batch_num} : train loss {loss.item()}")
# #             print(f"English: {eng_batch[0]}")
# #             print(f"french Translation: {french_batch[0]}")
#             french_sentence_predicted = torch.argmax(french_predictions[0], axis=1)
#             predicted_sentence = ""
#             for idx in french_sentence_predicted:
#               if idx == french_to_index[END_TOKEN]:
#                 break
#               predicted_sentence += index_to_french[idx.item()]
# #             print(f"french Prediction: {predicted_sentence}")


#             transformer.eval()
#             french_sentence = ("",)
#             eng_sentence = ("should we go to the mall?",)
#             for word_counter in range(max_sequence_length):
#                 encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, french_sentence)
#                 predictions = transformer(eng_sentence,
#                                           french_sentence,
#                                           encoder_self_attention_mask.to(device), 
#                                           decoder_self_attention_mask.to(device), 
#                                           decoder_cross_attention_mask.to(device),
#                                           enc_start_token=False,
#                                           enc_end_token=False,
#                                           dec_start_token=True,
#                                           dec_end_token=False)
#                 next_token_prob_distribution = predictions[0][word_counter] # not actual probs
#                 next_token_index = torch.argmax(next_token_prob_distribution).item()
#                 next_token = index_to_french[next_token_index]
#                 french_sentence = (french_sentence[0] + next_token, )
#                 if next_token == END_TOKEN:
#                   break
            
# #             print(f"Evaluation translation (should we go to the mall?) : {french_sentence}")
# #             print("-------------------------------------------")


# In[56]:


# transformer.train()
# transformer.to(device)
# total_train_loss = 0
# total_valid_loss = 0
# num_epochs = 1

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch}")
    
#     # Training phase
#     iterator = iter(train_loader)
#     for batch_num, batch in enumerate(iterator):
#         transformer.train()
#         eng_batch, french_batch = batch
#         encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
#         optim.zero_grad()
#         french_predictions = transformer(eng_batch,
#                                          french_batch,
#                                          encoder_self_attention_mask.to(device), 
#                                          decoder_self_attention_mask.to(device), 
#                                          decoder_cross_attention_mask.to(device),
#                                          enc_start_token=False,
#                                          enc_end_token=False,
#                                          dec_start_token=True,
#                                          dec_end_token=True)
#         labels = transformer.decoder.sentence_embedding.batch_tokenize(french_batch, start_token=False, end_token=True)
#         loss = criterian(
#             french_predictions.view(-1, french_vocab_size).to(device),
#             labels.view(-1).to(device)
#         ).to(device)
#         valid_indicies = torch.where(labels.view(-1) == french_to_index[PADDING_TOKEN], False, True)
#         loss = loss.sum() / valid_indicies.sum()
#         loss.backward()
#         optim.step()
#         total_train_loss += loss.item()
        
#         if batch_num % 100 == 0:
#             print(f"Iteration {batch_num} : train loss {loss.item()}")
    
#     # Validation phase
#     transformer.eval()
#     with torch.no_grad():
#         valid_iterator = iter(validation_loader)
#         for batch_num, batch in enumerate(valid_iterator):
#             eng_batch, french_batch = batch
#             encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
#             french_predictions = transformer(eng_batch,
#                                              french_batch,
#                                              encoder_self_attention_mask.to(device), 
#                                              decoder_self_attention_mask.to(device), 
#                                              decoder_cross_attention_mask.to(device),
#                                              enc_start_token=False,
#                                              enc_end_token=False,
#                                              dec_start_token=True,
#                                              dec_end_token=True)
#             labels = transformer.decoder.sentence_embedding.batch_tokenize(french_batch, start_token=False, end_token=True)
#             valid_indicies = torch.where(labels.view(-1) == french_to_index[PADDING_TOKEN], False, True)
#             valid_loss = criterian(
#                 french_predictions.view(-1, french_vocab_size).to(device),
#                 labels.view(-1).to(device)
#             ).to(device)
#             valid_loss = valid_loss.sum() / valid_indicies.sum()
#             total_valid_loss += valid_loss.item()

#     avg_train_loss = total_train_loss / len(train_loader)
#     avg_valid_loss = total_valid_loss / len(validation_loader)
    
#     print(f"Avg Train Loss: {avg_train_loss:.4f}")
#     print(f"Avg Validation Loss: {avg_valid_loss:.4f}")


# In[58]:


transformer.train()
transformer.to(device)
total_train_loss = 0
total_valid_loss = 0
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    
    # Training phase
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, french_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
        optim.zero_grad()
        french_predictions = transformer(eng_batch,
                                         french_batch,
                                         encoder_self_attention_mask.to(device), 
                                         decoder_self_attention_mask.to(device), 
                                         decoder_cross_attention_mask.to(device),
                                         enc_start_token=False,
                                         enc_end_token=False,
                                         dec_start_token=True,
                                         dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(french_batch, start_token=False, end_token=True)
        loss = criterian(
            french_predictions.view(-1, french_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == french_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        total_train_loss += loss.item()
        
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : train loss {loss.item()}")
    
    # Validation phase
    transformer.eval()
    with torch.no_grad():
        valid_iterator = iter(validation_loader)
        for batch_num, batch in enumerate(valid_iterator):
            eng_batch, french_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
            french_predictions = transformer(eng_batch,
                                             french_batch,
                                             encoder_self_attention_mask.to(device), 
                                             decoder_self_attention_mask.to(device), 
                                             decoder_cross_attention_mask.to(device),
                                             enc_start_token=False,
                                             enc_end_token=False,
                                             dec_start_token=True,
                                             dec_end_token=True)
            labels = transformer.decoder.sentence_embedding.batch_tokenize(french_batch, start_token=False, end_token=True)
            valid_indicies = torch.where(labels.view(-1) == french_to_index[PADDING_TOKEN], False, True)
            valid_loss = criterian(
                french_predictions.view(-1, french_vocab_size).to(device),
                labels.view(-1).to(device)
            ).to(device)
            valid_loss = valid_loss.sum() / valid_indicies.sum()
            total_valid_loss += valid_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_valid_loss = total_valid_loss / len(validation_loader)
    
    print(f"Avg Train Loss: {avg_train_loss:.4f}")
    print(f"Avg Validation Loss: {avg_valid_loss:.4f}")


# In[59]:


torch.save(transformer.state_dict(), '/kaggle/working/model_weights.pth')


# In[60]:


transformer.load_state_dict(torch.load('/kaggle/working/model_weights.pth'))


# In[61]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformer.to(device)


# In[62]:


# evaluate the model on test set 

transformer.eval()  # Set the model to evaluation mode

total_test_loss = 0

with torch.no_grad():  # Ensure no gradients are computed during testing
    test_iterator = iter(test_loader)  # Assuming you have a test_loader similar to train_loader and validation_loader
    for batch_num, batch in enumerate(test_iterator):
        eng_batch, french_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
        french_predictions = transformer(eng_batch,
                                         french_batch,
                                         encoder_self_attention_mask.to(device), 
                                         decoder_self_attention_mask.to(device), 
                                         decoder_cross_attention_mask.to(device),
                                         enc_start_token=False,
                                         enc_end_token=False,
                                         dec_start_token=True,
                                         dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(french_batch, start_token=False, end_token=True)
        valid_indicies = torch.where(labels.view(-1) == french_to_index[PADDING_TOKEN], False, True)
        test_loss = criterian(
            french_predictions.view(-1, french_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        test_loss = test_loss.sum() / valid_indicies.sum()
        total_test_loss += test_loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Avg Test Loss: {avg_test_loss:.4f}")


# In[ ]:


# from nltk.translate.bleu_score import corpus_bleu

# def compute_bleu(model, data_loader, device):
#     model.eval()
#     all_references = []  # List of list of references
#     all_predictions = []  # List of predictions

#     with torch.no_grad():
#         for batch in data_loader:
#             eng_batch, french_batch = batch
#             encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
#             french_predictions = model(eng_batch,
#                                        french_batch,
#                                        encoder_self_attention_mask.to(device), 
#                                        decoder_self_attention_mask.to(device), 
#                                        decoder_cross_attention_mask.to(device),
#                                        enc_start_token=False,
#                                        enc_end_token=False,
#                                        dec_start_token=True,
#                                        dec_end_token=True)
            
#             # Convert model outputs to actual words (This will depend on how your model works. Adjust as needed.)
#             pred_sentences = ...  # Convert `french_predictions` to list of predicted sentences
#             ref_sentences = ...  # Convert `french_batch` to list of actual sentences
            
#             all_predictions.extend(pred_sentences)
#             all_references.extend([[ref] for ref in ref_sentences])  # `corpus_bleu` expects a list of list of references

#     # Compute BLEU score
#     bleu_score = corpus_bleu(all_references, all_predictions)
#     return bleu_score


# In[63]:


from nltk.translate.bleu_score import corpus_bleu


# In[64]:


def compute_bleu(model, data_loader, device):
    model.eval()
    all_references = []  # List of list of references
    all_predictions = []  # List of predictions

    with torch.no_grad():
        for batch in data_loader:
            eng_batch, french_batch = batch
#             print(french_batch[0][:10])

            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, french_batch)
            french_predictions = model(eng_batch,
                                       french_batch,
                                       encoder_self_attention_mask.to(device), 
                                       decoder_self_attention_mask.to(device), 
                                       decoder_cross_attention_mask.to(device),
                                       enc_start_token=False,
                                       enc_end_token=False,
                                       dec_start_token=True,
                                       dec_end_token=True)
            
            # Convert model outputs to actual words 
            pred_sentences = convert_predictions_to_sentences(french_predictions)  # Implement this function
            ref_sentences = convert_tensor_to_sentences(french_batch)  # Implement this function
            
            all_predictions.extend(pred_sentences)
            all_references.extend([[ref] for ref in ref_sentences])  # `corpus_bleu` expects a list of list of references

    # Compute BLEU score
    bleu_score = corpus_bleu(all_references, all_predictions)
    return bleu_score


index_to_word = {index: word for word, index in french_to_index.items()}
def convert_predictions_to_sentences(predictions_tensor):
    # Convert logits to word indices
    _, predicted_indices = predictions_tensor.max(dim=-1)
    sentences = []
    for pred in predicted_indices:
        sentence = [index_to_word[token_idx.item()] for token_idx in pred]  # Convert tensor to integer and then lookup word
        sentences.append(sentence)
    return sentences


def convert_tensor_to_sentences(tensor):
    # If the tensor already contains words rather than indices, return them as is
    if isinstance(tensor[0][0], str):
        return tensor
    # Otherwise, convert using the index_to_word dictionary
    sentences = []
    for seq in tensor:
        sentence = [index_to_word[token_idx] for token_idx in seq]
        sentences.append(sentence)
    return sentences


# In[ ]:





# In[65]:


train_bleu = compute_bleu(transformer, train_loader, device)
test_bleu = compute_bleu(transformer, test_loader, device)
validation_bleu = compute_bleu(transformer, validation_loader, device)

print(f"Train BLEU Score: {train_bleu:.4f}")
print(f"Test BLEU Score: {test_bleu:.4f}")
print(f"Validation BLEU Score: {validation_bleu:.4f}")


# In[ ]:





# In[ ]:




