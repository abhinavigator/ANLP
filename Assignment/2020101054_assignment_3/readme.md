Machine Translation using Transformer

This code for implementing a machine translation model using the Transformer architecture. The model translates English sentences into French sentences.


Dataset:

The machine translation model is trained on the TED Talks corpus, which consists of English and French parallel sentences.

The dataset is organized into three main parts:

* Training set: Used to train the model.
* Validation set: Used to validate the model during training and tune hyperparameters.
* Test set: Used to evaluate the model after training.


Training:

The model is using the Transformer architecture with specified parameters:

* d_model: 512
* ffn_hidden: 2048
* num_heads: 8
* drop_prob: 0.1
* num_layers: 2
* max_sequence_length: 350

The training process involves minimizing the CrossEntropyLoss, ignoring padding tokens.


Evaluation:

The model is evaluated using BLEU (Bilingual Evaluation Understudy) score, which measures the similarity between the predicted translations and the ground truth.

BLEU Score for:

Training set: {0.0395}
Validation set: {0.0335}
Test set: {0.0370}




here is the link of saved model for this assignment :

https://drive.google.com/file/d/1qB8JpthOI-9HNWIsU5RXiBi4R18lUXxT/view?usp=drive_link