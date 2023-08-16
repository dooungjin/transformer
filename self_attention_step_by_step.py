from transformers import MobileBertTokenizer, MobileBertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load pre-trained tokenizer and model for MobileBERT
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertModel.from_pretrained("google/mobilebert-uncased")

hidden_size = 512  
projection_size = 256  # Size for query, key, and value projections

# Input sentence
input_sentence = "I am a boy"

# Tokenize the input and get token IDs
token_ids = tokenizer.encode(input_sentence, add_special_tokens=True, return_tensors="pt")
# tensor([[ 101, 1045, 2572, 1037, 2879,  102]])

"""
101 is the token ID of the special token [CLS], which stands for "classification" and is used to mark the beginning of a sequence.
1045 corresponds to the token ID of the word "I" in the vocabulary.
2572 corresponds to the token ID of the word "am" in the vocabulary.
1037 corresponds to the token ID of the word "a" in the vocabulary.
2879 corresponds to the token ID of the word "boy" in the vocabulary.
102 is the token ID of the special token [SEP], which stands for "separator" and is used to mark the end of a sentence
"""

# Create positional encodings
max_seq_length = token_ids.size(1)
position_ids = torch.arange(0, max_seq_length).unsqueeze(0)

# Get model embeddings with positional encodings
with torch.no_grad():
    embeddings = model(input_ids=token_ids, position_ids=position_ids).last_hidden_state
# torch.Size([1, 6, 512])

# 512 by 256
W_q = nn.Parameter(torch.randn(hidden_size, projection_size))
W_k = nn.Parameter(torch.randn(hidden_size, projection_size))
W_v = nn.Parameter(torch.randn(hidden_size, projection_size))

# Calculate query, key, and value matrices
query_matrices = torch.matmul(embeddings, W_q)
key_matrices = torch.matmul(embeddings, W_k)
value_matrices = torch.matmul(embeddings, W_v)

# Apply normalization (optional)
query_matrices = F.normalize(query_matrices, dim=-1)
key_matrices = F.normalize(key_matrices, dim=-1)
value_matrices = F.normalize(value_matrices, dim=-1)

# Calculate attention scores
attention_scores = torch.matmul(query_matrices, key_matrices.transpose(-2, -1))

# Apply softmax to get attention weights
attention_weights = F.softmax(attention_scores, dim=-1)

# Calculate the weighted sum of value vectors using attention weights
attended_values = torch.matmul(attention_weights, value_matrices)
